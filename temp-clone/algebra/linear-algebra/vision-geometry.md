# Linear Algebra in 3D Vision Geometry

## 1. 목적
3D vision 논문의 수식을 "선형대수 연산 블록"으로 분해해 읽고 구현하는 가이드.

## 2. 핵심 식
- Camera projection: `x ~ K [R|t] X`
- Epipolar constraint: `x'^T F x = 0`
- Essential matrix: `E = [t]_x R`
- Triangulation: 여러 view의 선형 제약 `AX=0` 풀기
- Bundle adjustment: reprojection error 최소화

## 3. 연구자 관점
- 기하 제약을 학습 신호로 넣는 방법:
  - hard constraint (정확한 구조 강제)
  - soft penalty (loss 항으로 유도)
- 선형해는 초기값, 비선형 최적화는 refinement로 분리하는 설계가 재현성 높다.
- rank/정규직교성/스케일 모호성 처리를 명시해야 주장 타당성이 생긴다.

## 4. 엔지니어 관점
- 수치 안정화 우선순위:
  1. 좌표 정규화 (Hartley normalization 류)
  2. SVD 기반 최소제곱 해
  3. 제약 재투영(rank-2, det(R)=1 등)
  4. robust loss + RANSAC
- 좌표계 정의(world/camera/image) 혼동은 버그의 주원인이다.

## 5. 주요 모듈별 실전 포인트

### 5.1 DLT와 SVD
- `AX=0` 최소해는 SVD에서 가장 작은 singular value에 대응하는 벡터.
- CV 예시: homography, fundamental matrix 초기 추정.
- 논문 등장: "closed-form initialization via SVD".

### 5.2 Fundamental/Essential Matrix
- `F`는 rank-2 제약 필수.
- `E`는 singular values 패턴 제약(이론적 구조)이 중요.
- CV 예시: two-view pose recovery.
- 논문 등장: geometric consistency loss + rank constraint.

### 5.3 Triangulation
- 여러 view ray의 교차를 최소제곱으로 계산.
- 작은 baseline에서는 수치적으로 불안정해 depth noise가 급증.
- 논문 등장: uncertainty-aware triangulation.

### 5.4 Bundle Adjustment (BA)
- 반복적 선형화(가우스-뉴턴/LM)와 Jacobian 블록 구조 활용.
- sparse block 행렬 계산이 속도/메모리 병목.
- 논문 등장: differentiable BA, implicit differentiation.

## 6. 자주 헷갈리는 포인트
- 선형해가 최종해가 아니다: 대개 refinement 단계가 필요.
- scale/sign ambiguity 처리 없이 성능 비교하면 지표 해석이 틀어진다.
- reprojection error만 낮다고 geometry가 항상 올바른 것은 아니다.
- outlier가 많은 데이터에서 순수 least squares는 쉽게 붕괴한다.

## 7. 실험 아이디어

```python
import torch

# rank-2 enforcement toy
F = torch.randn(3, 3)
U, S, Vh = torch.linalg.svd(F)
S[-1] = 0.0
F_rank2 = U @ torch.diag(S) @ Vh
print("rank(F_rank2):", int(torch.linalg.matrix_rank(F_rank2)))

# orthogonality projection for R
R = torch.randn(3, 3)
U, _, Vh = torch.linalg.svd(R)
R_proj = U @ Vh
if torch.linalg.det(R_proj) < 0:
    U[:, -1] *= -1
    R_proj = U @ Vh
print("det(R_proj):", float(torch.linalg.det(R_proj)))
```

## 8. 논문 읽기 체크리스트
1. 선형화가 어디서 발생하는가?
2. 어떤 제약을 hard로 강제하고 어떤 제약을 soft로 두는가?
3. SVD/eigen 후 제약 재투영 단계가 명시되어 있는가?
4. 강건성(outlier, low-texture, small-baseline) 평가가 충분한가?
5. 학습 기반 모듈과 기하 모듈의 경계(gradient 흐름)가 명확한가?

## 9. 딥러닝과 결합되는 지점
- NeRF/GS 계열: pose uncertainty가 렌더링 품질에 직접 영향.
- Multi-view depth: photometric loss + geometric consistency 동시 사용.
- Learned feature matching: descriptor metric과 epipolar 제약의 결합.

## 10. 연결 문서
- [fundamentals](./fundamentals.md)
- [matrix-calculus](./matrix-calculus.md)
- [eigens-and-svd](./eigens-and-svd.md)
