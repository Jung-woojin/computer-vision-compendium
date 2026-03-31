# Eigens and SVD for Vision/DL Architecture Design

## 1) 왜 이 문서가 필요한가
고유값/특이값은 모델의 "안정성, 정보 보존, 압축 가능성"을 동시에 보여주는 가장 강력한 진단 도구다.  
이 문서는 다음 질문에 답한다.
- 지금 레이어는 정보를 얼마나 보존하는가?
- 학습이 불안정한 이유가 스펙트럼 폭주 때문인가?
- low-rank 근사로 속도를 올려도 정확도가 유지되는가?

## 2) Eigen vs SVD 핵심 비교
- Eigen decomposition: `A = VΛV^{-1}` (보통 정사각/대각화 가능 가정)
- SVD: `A = UΣV^T` (임의 행렬 가능, 실전 기본값)
- 실전 규칙:
  - 대칭 PSD 행렬(공분산, Hessian 근사): eigen 해석이 직관적
  - 일반 weight/activation 행렬: SVD가 더 안정적

## 3) 연구자 관점 / 엔지니어 관점

### 연구자 관점
- 고유값 분포는 동역학을 설명한다.
  - 큰 고유값: 빠른 변화축, 과민감 가능
  - 작은 고유값: 무시되는 축, 정보 소실 가능
- 스펙트럼 tail은 잡음/세부 정보 trade-off를 보여준다.
- rank 제약은 일반화와 효율성을 동시에 다루는 구조적 priors다.

### 엔지니어 관점
- full SVD를 매스텝 계산하면 병목이다.
- 모니터링 목적이면 top-k singular values만 추적해도 충분하다.
- 실전에서는 다음 세 가지를 로그로 남기면 좋다.
  - spectral norm (`σ_max`)
  - nuclear norm (`Σσ_i`)
  - effective rank

## 4) CV 예시
- ViT projection matrix의 singular spectrum을 epoch별로 추적해 feature collapse 징후 탐지.
- Low-rank attention 근사로 메모리 사용량 감소.
- Super-resolution에서 operator rank 제한이 고주파 복원 성능에 미치는 영향 분석.
- 3D vision에서 essential/fundamental matrix를 SVD 제약(rank=2)으로 정규화.

## 5) 논문에서 자주 나오는 패턴
- "spectral regularization", "spectral normalization"
- "low-rank adaptation", "rank-constrained optimization"
- "nuclear norm minimization"
- "whitening / decorrelation via eigendecomposition"
- "rank-2 constraint for fundamental matrix"

## 6) 자주 헷갈리는 포인트
- `σ_i`가 작다고 무조건 쓸모없는 정보는 아니다.
- rank를 낮추면 일반화가 좋아질 수 있지만 underfitting 위험도 증가.
- spectral norm만 제어하면 모든 안정성 문제가 해결되지는 않는다.
- 학습 중 스펙트럼 변화는 optimizer, normalization, augmentation의 영향도 받는다.

## 7) 실험 아이디어 (PyTorch)

```python
import torch

def effective_rank(W: torch.Tensor, eps: float = 1e-12) -> float:
    s = torch.linalg.svdvals(W)
    p = s / (s.sum() + eps)
    erank = torch.exp(-(p * torch.log(p + eps)).sum())
    return float(erank)

W = torch.randn(768, 768)
sv = torch.linalg.svdvals(W)
print("top-5 singular values:", sv[:5].tolist())
print("spectral norm:", float(sv[0]))
print("effective rank:", effective_rank(W))

# rank-k approximation error
k = 64
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
Wk = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
rel_err = torch.norm(W - Wk) / torch.norm(W)
print("rank-k relative error:", float(rel_err))
```

## 8) 논문 읽기 체크리스트
- 이 논문이 제어하려는 것은 `σ_max`(안정성)인가 `rank`(압축/일반화)인가?
- 제약이 train-time 전용인지, inference-time 가속으로도 이어지는가?
- 분해 후 fine-tuning(예: LoRA) 시 성능 회복 비용은 얼마인가?
- 스펙트럼 분석이 단일 레이어인지, 네트워크 전체인지?

## 9) 구현할 때 권장 루틴
1. 기준 모델 학습
2. 핵심 레이어 top singular values 추적
3. rank-k 압축 후보 탐색
4. 정확도-속도-메모리 trade-off 곡선 작성
5. 논문 주장과 실제 스펙트럼 변화를 함께 검증

## 10) 다음 문서 연결
- [fundamentals](C:/Users/ust21/linear-algebra-cv-dl-notes/linear-algebra/fundamentals.md)
- [matrix-calculus](C:/Users/ust21/linear-algebra-cv-dl-notes/linear-algebra/matrix-calculus.md)
- [optimization-geometry](C:/Users/ust21/linear-algebra-cv-dl-notes/linear-algebra/optimization-geometry.md)
