# Optimization Geometry for Deep Vision Models

## 1. 목적
학습이 안 될 때 "옵티마이저 바꿔보자" 수준에서 끝내지 않고,  
선형대수(스펙트럼, 조건수, 곡률) 관점으로 원인을 분해하기 위한 실전 가이드.

## 2. 핵심 프레임
- 1차 정보: gradient `g = ∇L(θ)`
- 2차 정보: Hessian `H = ∇^2L(θ)`
- 국소 근사: `L(θ+Δ) ≈ L(θ) + g^TΔ + 1/2 Δ^T H Δ`
- 목표: `g`의 방향성과 `H`의 스케일을 맞춰 안정적 업데이트를 수행

## 3. 연구자 관점
- Generalization은 단순히 train loss가 아니라 landscape geometry와 연결된다.
- Sharp minima/flat minima 논쟁은 Hessian 스펙트럼 분포를 봐야 실체가 보인다.
- Normalization/skip connection은 function class뿐 아니라 optimization geometry도 바꾼다.
- Curvature가 크고 anisotropic하면 동일 LR로는 일부 축에서 발산/일부 축에서 정체가 동시 발생.

## 4. 엔지니어 관점
- 장애 대응 우선순위:
  1. 학습률 스케일(LR, warmup, scheduler)
  2. gradient norm 분포(layer-wise)
  3. normalization 위치(BN/LN/RMSNorm)
  4. mixed precision overflow/underflow
  5. loss scale imbalance (multi-task일수록 중요)
- 단기 안정화 루틴:
  - gradient clipping
  - warmup 증가
  - weight decay 재조정
  - unstable head만 별도 LR 설정

## 5. 자주 쓰는 선형대수 개념

### 5.1 Condition Number
- `kappa(H)=lambda_max/lambda_min`
- 의미: 경사하강이 "찌그러진 타원 계곡"에서 얼마나 힘든지.
- CV 예시: 대규모 ViT pretrain에서 초기 conditioning 악화로 loss spike.

### 5.2 Preconditioning
- 업데이트 전 좌표계를 바꿔서 등방성에 가깝게 만듦.
- 실전 대응:
  - Adam/Adafactor 계열의 adaptive scaling
  - BN/LN 기반 feature scale 정렬
  - K-FAC류 2차 근사

### 5.3 Spectrum Tracking
- 상위 Hessian eigenvalue 추적은 LR 상한 추정에 유용.
- gradient covariance spectrum은 noise-driven regime 판단에 도움.

## 6. CV/딥러닝 연결
- CNN: 깊어질수록 Jacobian 스케일 제어 실패 시 gradient 소실/폭발.
- Transformer: residual branch 스케일 및 norm 위치가 학습 안정성 핵심.
- Detection/Segmentation: multi-head loss scaling 불균형이 conditioning 악화 유발.
- Diffusion: timestep별 gradient scale이 달라 스케줄링 설계가 중요.

## 7. 논문에서 어떻게 등장하나
- "loss landscape analysis"
- "sharpness-aware training (SAM)"
- "curvature regularization"
- "natural gradient / approximate Fisher"
- "adaptive preconditioning"

논문을 읽을 때는 다음을 확인:
- geometry 관련 주장이 실제 측정값(top eigenvalue, trace 등)으로 뒷받침되는가
- 안정화 기법이 train-time trick인지 inference 효율까지 이어지는가
- 배치 스케일 변화에 대한 재현성 결과가 있는가

## 8. 실험 아이디어

```python
import torch

def hvp(loss, params, v):
    g = torch.autograd.grad(loss, params, create_graph=True)
    gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
    hv = torch.autograd.grad(gv, params, retain_graph=True)
    return hv

# top curvature proxy
model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
x = torch.randn(64, 128)
y = torch.randint(0, 10, (64,))
loss = torch.nn.CrossEntropyLoss()(model(x), y)
params = [p for p in model.parameters() if p.requires_grad]
v = [torch.randn_like(p) for p in params]
hv = hvp(loss, params, v)
num = sum((h * vv).sum() for h, vv in zip(hv, v))
den = sum((vv * vv).sum() for vv in v)
rayleigh = (num / den).item()
print("curvature proxy:", rayleigh)
```

## 9. 실전 체크리스트
1. 학습 초반 1~3 epoch에서 loss spike가 있는가?
2. layer-wise gradient norm 분포가 극단적으로 치우치는가?
3. 학습률을 반으로 줄였을 때 안정성은 좋아지지만 수렴이 너무 느린가?
4. warmup 길이를 늘리면 문제 재현이 사라지는가?
5. BN/LN 위치를 바꾸면 같은 hyperparameter에서 곡선이 안정화되는가?

## 10. 연결 문서
- [fundamentals](./fundamentals.md)
- [matrix-calculus](./matrix-calculus.md)
- [eigens-and-svd](./eigens-and-svd.md)
