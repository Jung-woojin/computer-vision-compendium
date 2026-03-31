# 컴퓨터비전/딥러닝 아키텍처 연구자를 위한 선형대수 Fundamentals

## 1. 이 문서의 목적
이 문서는 "수학 교과서 요약"이 아니라, 다음 상황에서 바로 참조할 수 있는 실전 노트다.
- 논문을 읽을 때: 수식이 왜 필요한지, 모델 설계에서 어떤 의사결정을 유도하는지 파악
- 모델을 설계할 때: 표현 공간, 연산 구조, 최적화 안정성을 선형대수 관점으로 점검
- 디버깅할 때: 성능 저하를 rank 붕괴, conditioning 악화, 거리/정규화 미스매치로 해석

핵심 원칙:
- 증명보다 직관
- 직관만이 아니라 수식
- 수식만이 아니라 PyTorch 구현 감각

## 2. 왜 이 개념이 비전/딥러닝에 중요한가
- CNN: convolution/1x1 conv/group conv/depthwise conv는 모두 선형 연산자 구조 설계 문제다.
- Representation learning: 임베딩 품질은 거리(norm/distance), 각도(orthogonality), 유효차원(rank)으로 결정된다.
- Optimization: gradient descent는 국소 선형화, 2차 근사(Hessian) 위에서 움직이는 알고리즘이다.
- Metric learning: cosine/L2/Mahalanobis 선택이 곧 귀납편향(inductive bias) 선택이다.
- Transformer: Q/K/V projection, attention map, MLP projection은 연속된 선형변환 조합이다.
- 3D Vision: triangulation, pose, epipolar 제약은 선형화 + SVD + 최소제곱으로 구현된다.

## 3. 핵심 개념

아래는 각 주제를 "직관 / 연구자 관점 / 엔지니어 관점 / CV 예시 / 논문에서의 등장"으로 정리한 실전 카드다.

### 3.1 System of Linear Equations (`Ax=b`)
- 직관: 제약 조건들의 교집합을 찾는 문제.
- 연구자 관점: 해 존재성/유일성은 `rank(A)`와 `rank([A|b])`로 판정한다.
- 엔지니어 관점: 역행렬을 직접 계산하지 말고 `solve/lstsq`를 사용한다.
- CV 예시: PnP 초기화, 선형 삼각측량(triangulation), DLT.
- 논문 등장: `A^T A x = A^T b` 형태의 normal equation.

### 3.2 Vector Spaces, Basis, Dimension
- 직관: 데이터가 실제로 살아있는 방향의 집합.
- 연구자 관점: "고차원 표현"보다 "유효 차원"이 일반화와 압축 가능성을 좌우한다.
- 엔지니어 관점: 채널 수를 늘려도 실제 정보 차원이 늘지 않을 수 있다.
- CV 예시: ViT feature가 클래스별로 몇 개의 주축에서 분리되는지 PCA로 확인.
- 논문 등장: intrinsic dimension, subspace regularization, manifold 가정.

### 3.3 Linear Independence
- 직관: 벡터들이 서로 대체 불가능한 새 정보를 주는가.
- 연구자 관점: collapse는 독립성 상실 문제로 해석 가능하다.
- 엔지니어 관점: batch feature covariance의 off-diagonal가 커지면 중복 표현 가능성.
- CV 예시: self-supervised pretraining에서 representation collapse 탐지.
- 논문 등장: redundancy reduction (Barlow Twins 류), decorrelation loss.

### 3.4 Matrix Multiplication as Transformation
- 직관: 행렬 곱은 변환의 합성(회전/스케일/혼합/투영).
- 연구자 관점: layer stack을 "연산자(operator)의 연쇄"로 본다.
- 엔지니어 관점: shape/stride/layout이 의미를 결정한다 (`[B, N, D] @ [D, D]`).
- CV 예시: 1x1 conv = 채널 공간 선형변환, attention의 Q/K/V projection.
- 논문 등장: token mixing, projection head, linear probe.

### 3.5 Rank and Null Space
- 직관: rank는 보존된 정보 차원, null space는 소거된 방향.
- 연구자 관점: 모델 병목이 정보 보존 실패인지(rank) 확인한다.
- 엔지니어 관점: weight의 effective rank를 로그로 추적해 과압축을 조기 발견.
- CV 예시: 저해상도 복원에서 high-frequency 성분이 null space로 빠져 디테일 손실.
- 논문 등장: low-rank adaptation(LoRA), bottleneck design, spectral pruning.

### 3.6 Orthogonality and Projection
- 직관: 직교는 정보 중복 최소화, 투영은 관심 성분 분리.
- 연구자 관점: nuisance factor 제거를 projection으로 모델링.
- 엔지니어 관점: 정규직교 제약은 학습 안정성과 gradient 분산 제어에 유리.
- CV 예시: 배경/조명/identity 성분 분리.
- 논문 등장: orthogonal regularization, subspace projection head.

### 3.7 Eigenvalues and Eigenvectors
- 직관: 변환의 고정축과 증폭률.
- 연구자 관점: 동역학 안정성(수렴/발산), 표현 방향성의 핵심 지표.
- 엔지니어 관점: 반복 업데이트 연산의 spectral radius를 점검.
- CV 예시: diffusion/graph message passing 안정성 분석.
- 논문 등장: spectral norm constraint, Laplacian eigen analysis.

### 3.8 Diagonalization
- 직관: 연산자를 "축별 독립 스케일링"으로 분해해 해석.
- 연구자 관점: 해석 가능성/계산 단순화/근사해 도출에 유리.
- 엔지니어 관점: 일반행렬은 대각화 불가할 수 있으니 Schur/SVD 우회가 실전적.
- CV 예시: 공분산 대각화 기반 whitening.
- 논문 등장: second-order 근사, preconditioning 해석.

### 3.9 SVD
- 직관: 임의 행렬을 입력 회전-스케일-출력 회전으로 분해.
- 연구자 관점: 저랭크 구조, 노이즈 분리, 정보축 해석의 표준.
- 엔지니어 관점: full SVD는 비싸므로 truncated/randomized 접근 검토.
- CV 예시: LoRA, low-rank attention, 3D geometric fitting.
- 논문 등장: nuclear norm, low-rank regularization, robust decomposition.

### 3.10 Positive Definite Matrices (SPD/PD)
- 직관: 모든 방향에서 에너지가 양수인 안정적 곡면.
- 연구자 관점: covariance/metric/Hessian 근사의 기본 구조.
- 엔지니어 관점: SPD 보장을 위해 `A = L L^T + eps I` 파라미터화 사용.
- CV 예시: Mahalanobis metric learning, covariance pooling.
- 논문 등장: Gaussian modeling, Riemannian learning on SPD manifold.

### 3.11 Quadratic Forms (`x^T A x`)
- 직관: 방향별 패널티/에너지 측정.
- 연구자 관점: regularizer를 방향 선택적으로 설계 가능.
- 엔지니어 관점: 손실항을 quadratic으로 보면 조건수/학습률 선택 근거가 생긴다.
- CV 예시: optical flow smoothness, graph Laplacian regularization.
- 논문 등장: Tikhonov regularization, energy minimization.

### 3.12 Norms and Distances
- 직관: "가깝다/멀다"의 정의가 학습 목표를 바꾼다.
- 연구자 관점: metric 선택이 representation geometry를 결정한다.
- 엔지니어 관점: normalize 위치(pre/post projection)가 성능에 큰 영향.
- CV 예시: Face recognition에서 cosine margin, re-ID에서 triplet distance.
- 논문 등장: contrastive, triplet, ArcFace, proxy-based losses.

### 3.13 Matrix Factorization
- 직관: 복잡한 연산을 작은 의미 단위로 분해.
- 연구자 관점: 생성요인(disentangled factor)을 해석하거나 제약한다.
- 엔지니어 관점: 파라미터/연산량 감소와 배포 효율성 향상.
- CV 예시: convolution kernel factorization (k x k -> k x 1 + 1 x k).
- 논문 등장: CP/Tucker decomposition, low-rank adapters.

### 3.14 Matrix Calculus Basics
- 직관: 벡터/행렬 변수에 대한 미분 문법.
- 연구자 관점: 논문의 gradient 유도는 설계 가정(대칭성, 정규화)을 드러낸다.
- 엔지니어 관점: autograd 결과를 shape와 transpose 규칙으로 sanity check.
- CV 예시: custom loss 구현에서 broadcasting 오류로 gradient 왜곡.
- 논문 등장: supplementary의 미분 유도식, closed-form updates.

### 3.15 Jacobian / Hessian Intuition
- 직관: Jacobian=민감도 지도, Hessian=곡률 지도.
- 연구자 관점: sharp/flat minima, robustness, generalization 연결.
- 엔지니어 관점: full Hessian 대신 HVP, top-eigen, trace 근사를 쓴다.
- CV 예시: 입력 Jacobian norm 제어로 adversarial robustness 개선.
- 논문 등장: SAM, curvature regularization, second-order optimizer 분석.

### 3.16 PCA and Low-rank Approximation
- 직관: 중요한 축만 남겨 노이즈/중복을 제거.
- 연구자 관점: representation의 통계 구조를 요약해 모델 개선 포인트를 찾는다.
- 엔지니어 관점: 차원 축소로 시각화/캐싱/검색 속도 개선.
- CV 예시: embedding retrieval에서 PCA whitening 전후 recall 비교.
- 논문 등장: feature analysis, model compression pre-analysis.

### 3.17 Optimization and Gradient-based Learning Connection
- 직관: 학습은 1차(gradient)와 2차(곡률) 정보를 번갈아 쓰는 근사 반복.
- 연구자 관점: conditioning, preconditioning, spectrum이 수렴 속도/해 품질을 좌우.
- 엔지니어 관점: LR, WD, normalization은 모두 선형대수적 스케일 조정.
- CV 예시: 대규모 배치에서 warmup 없이 divergence 발생 -> 스펙트럼 스케일 미스매치.
- 논문 등장: AdamW 해석, natural gradient 근사, K-FAC 계열.

## 4. 수식 직관

### 4.1 세 줄 요약 수식
- 선형 시스템: `x* = argmin_x ||Ax-b||_2^2 = (A^T A)^{-1}A^T b` (full rank 가정)
- 투영: `P = A(A^T A)^{-1}A^T`, `x_proj = Px`, `r = x - Px`는 `col(A)`에 직교
- SVD: `A = UΣV^T`, `rank-k` 근사: `A_k = U_k Σ_k V_k^T`

### 4.2 왜 이 수식이 중요한가
- `A^T A`가 ill-conditioned면 작은 노이즈도 해를 크게 흔든다.
- 투영/직교 분해는 "신호 vs nuisance" 분리를 코드 수준에서 가능하게 한다.
- SVD 절단(truncation)은 압축/가속/일반화 사이의 trade-off를 조절한다.

### 4.3 Jacobian/Hessian 직관 수식
- Jacobian: `J_f(x) = ∂f/∂x` (입력의 미소 변화가 출력으로 어떻게 전달되는지)
- Hessian: `H_L(θ) = ∂^2 L / ∂θ^2` (손실 지형의 국소 곡률)
- 2차 근사: `L(θ+Δ) ≈ L(θ) + g^TΔ + 1/2 Δ^T H Δ`

## 5. 딥러닝/컴퓨터비전 연결

### CNN
- Conv는 선형연산 + 비선형의 반복이며, 채널 혼합은 사실상 행렬곱이다.
- 1x1 conv는 feature basis를 재정의하는 projection 연산으로 볼 수 있다.

### Representation Learning
- 좋은 representation은 class 신호는 유지(rank 유지), nuisance는 줄이는(projection) 구조를 갖는다.
- collapse 탐지는 covariance eigen-spectrum으로 빠르게 가능하다.

### Optimization
- 학습 실패를 "optimizer 이슈"로만 보지 말고 condition number 문제로 재해석해야 한다.
- normalization(BN/LN)은 사실상 스케일 정렬(preconditioning)에 가깝다.

### Metric Learning
- cosine vs L2 vs Mahalanobis는 embedding 공간의 기하학을 다르게 강제한다.
- 임베딩 정규화는 각도 정보 중심 학습을 유도한다.

### Transformer
- attention score `QK^T / sqrt(d)`는 Gram-like 연산이며, 분산 스케일링이 핵심.
- low-rank attention 근사는 메모리/연산량 병목 해결의 표준 패턴이다.

### 3D Vision
- Essential/Fundamental matrix 추정은 SVD 제약(`rank=2`)이 중요하다.
- BA는 반복적인 선형화와 Jacobian 블록 구조 활용 문제다.

## 6. 자주 헷갈리는 포인트
- "역행렬 쓰면 깔끔"은 착각이다: 수치 안정성은 `solve/lstsq/cholesky`가 훨씬 낫다.
- "차원 크면 표현력 좋다"는 반만 맞다: 유효 차원과 잡음 차원을 구분해야 한다.
- "SVD = 고유분해"는 오해다: 대칭행렬 특수 케이스에서만 유사하다.
- "Hessian은 못 구하니 의미 없다"는 오해다: HVP/근사 고유값만으로도 유익하다.
- "PCA는 구식"이 아니다: 해석/압축/전처리에서 여전히 강력한 baseline이다.

## 7. 작은 예제 또는 numpy/pytorch 실험 아이디어

### 7.1 Effective rank 모니터링
```python
import torch

W = torch.randn(1024, 512)
s = torch.linalg.svdvals(W)
p = s / (s.sum() + 1e-12)
effective_rank = torch.exp(-(p * torch.log(p + 1e-12)).sum())
print("effective rank:", float(effective_rank))
```

### 7.2 Projection으로 nuisance 제거
```python
import torch

# nuisance basis (예: 조명 방향)라고 가정
A = torch.randn(256, 8)
P = A @ torch.linalg.inv(A.T @ A) @ A.T
I = torch.eye(256)
P_perp = I - P

x = torch.randn(256)
x_clean = P_perp @ x
print("removed energy:", float(torch.norm(P @ x)))
```

### 7.3 Hessian-vector product (HVP) 장난감 실험
```python
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)
)
x = torch.randn(16, 32)
y = torch.randint(0, 10, (16,))
loss_fn = torch.nn.CrossEntropyLoss()

loss = loss_fn(model(x), y)
params = [p for p in model.parameters() if p.requires_grad]
g = torch.autograd.grad(loss, params, create_graph=True)
v = [torch.randn_like(p) for p in params]
gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
hvp = torch.autograd.grad(gv, params)
print("hvp norms:", [float(t.norm()) for t in hvp])
```

## 8. 논문 읽기 연결 포인트
- `spectral`, `subspace`, `orthogonal`, `low-rank`, `conditioning`, `curvature`는 핵심 수학 신호어다.
- 방법론에서 projection/factorization이 나오면 목적은 대개 둘 중 하나다: 안정화 또는 효율화.
- 부록의 미분식은 구현에서 가장 좋은 unit test 힌트다.
- 3D 비전 논문에서 SVD/eigen은 "제약을 만족하는 해 구성"에 자주 쓰인다.
- 아키텍처 논문에서 행렬 형태를 먼저 정리하면 실험 실패 원인을 빠르게 좁힐 수 있다.

## 9. GitHub에 같이 링크할 후속 문서
이 문서는 허브 문서다. 아래 문서로 분할해 심화한다.

- `linear-algebra/fundamentals.md`
  - 개념 전체 지도
  - 실험 quick-start
- `linear-algebra/eigens-and-svd.md`
  - 고유값/특이값 스펙트럼 해석
  - low-rank 설계 패턴과 구현
- `linear-algebra/matrix-calculus.md`
  - matrix calculus 규칙
  - Jacobian/Hessian 실전 계산
- `linear-algebra/optimization-geometry.md`
  - condition number, preconditioning, sharpness
- `linear-algebra/metric-and-representation.md`
  - 거리, 정규화, 임베딩 기하
- `linear-algebra/vision-geometry.md`
  - epipolar/pose/triangulation, BA 선형화

## 10. 핵심 질문 5개
1. 현재 모델의 성능 병목은 표현력 부족(rank)인가, 최적화 불안정(conditioning)인가?
2. 내가 택한 metric(L2/cosine/Mahalanobis)은 task semantics와 진짜로 정합적인가?
3. feature 공간에서 어떤 방향이 signal이고 어떤 방향이 nuisance인지 투영 관점으로 설명 가능한가?
4. 학습률/정규화/초기화 설정이 Jacobian/Hessian 관점에서 어떤 스케일을 강제하는가?
5. 이 논문의 기여를 "선형변환 + 제약 + 분해"로 환원하면 본질이 무엇인가?
