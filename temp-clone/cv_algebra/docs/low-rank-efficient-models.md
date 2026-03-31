# Low-rank Efficient Vision Models

## 1. 목적
low-rank 근사를 이용해 비전 모델의 연산량/메모리를 줄이면서  
정확도 손실을 관리하는 실전 절차를 정리한다.

## 2. 핵심 질문
- 어떤 레이어가 low-rank 근사에 민감한가?
- rank를 어떻게 선택해야 정확도-속도 균형이 맞는가?
- LoRA/adapter류와 구조적 분해를 언제 구분해서 써야 하는가?

## 3. 기본 아이디어
- 큰 행렬 `W`를 `W ≈ A B`로 분해하면 파라미터/계산량 감소
- SVD 기반이면 최적 rank-k 근사를 직접 얻을 수 있음
- Conv도 커널 차원/채널 차원으로 factorization 가능

## 4. 연구자 관점
- 저랭크 가정은 데이터/표현의 내재적 저차원 구조 가정과 연결된다.
- rank 제약은 일반화 향상에도 기여할 수 있지만 과도하면 underfitting 발생.
- layer별 spectrum이 서로 달라 균일 rank 정책은 비효율적일 수 있다.

## 5. 엔지니어 관점
- 압축은 반드시 실제 장비 latency로 검증해야 한다.
- FLOPs 감소가 latency 감소를 보장하지 않는다.
- 정적 분해(배포 전)와 동적 adaptation(학습 중)을 분리해 운영한다.

## 6. CV 예시
- Mobile/Edge 환경에서 backbone projection 레이어 저랭크화.
- Segmentation head 경량화 시 rank를 낮추되 boundary 품질 유지 점검.
- ViT에서 MLP/attention projection의 rank sweep으로 병목 제거.

## 7. 딥러닝 예시
- LoRA: `ΔW = BA` 형태의 저랭크 업데이트만 학습.
- Truncated SVD compression: 사전학습 weight를 rank-k로 근사 후 재학습.
- Tensor decomposition: CP/Tucker로 convolution 압축.

## 8. 논문에서 보이는 형태
- "low-rank adaptation"
- "factorized projection"
- "rank-constrained training"
- "tensor decomposition for CNN acceleration"

## 9. 구현 체크리스트
1. 대상 레이어 선정 기준 정의(파라미터 비중/시간 비중)
2. rank 후보군을 로그 스케일로 스윕
3. 정확도/latency/메모리 동시 기록
4. 근사 후 짧은 재학습(finetune) 단계 포함
5. 배포 엔진 최적화 가능성(연산 fusion) 확인
6. 장치별 성능(서버/GPU/모바일) 분리 측정

## 10. 미니 실험 아이디어
```python
import torch

def compress_linear(W, k):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    A = U[:, :k] @ torch.diag(S[:k])
    B = Vh[:k, :]
    return A, B

W = torch.randn(2048, 2048)
for k in [32, 64, 128, 256, 512]:
    A, B = compress_linear(W, k)
    Wk = A @ B
    err = torch.norm(W - Wk) / (torch.norm(W) + 1e-12)
    params_ratio = (A.numel() + B.numel()) / W.numel()
    print(f"k={k}, rel_err={float(err):.5f}, param_ratio={params_ratio:.3f}")
```

## 11. 자주 나오는 오해
- "rank만 낮추면 다 빨라진다": 메모리 접근 패턴 때문에 느려질 수도 있다.
- "한 번 압축하면 끝": 근사 후 재학습 단계가 성능 복구에 중요하다.
- "모든 레이어 동일 rank": 레이어별 민감도 차이를 무시하면 비효율적.

## 12. 추천 실험 순서
1. Baseline 학습/추론 측정
2. 레이어별 spectrum 분석
3. 민감한 레이어 rank sweep
4. best trade-off 조합 선택
5. 재학습 후 재측정

## 13. 연결 문서
- [허브 문서](./linear-algebra-for-cv.md)
- [SVD/PCA 문서](./eigens-svd-pca.md)
- [Convolution operator 문서](./convolution-as-operator.md)
