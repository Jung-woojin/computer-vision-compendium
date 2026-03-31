# Optimization Curvature

## 1. 목적
학습 안정성 문제를 "하이퍼파라미터 감"이 아니라  
Hessian/곡률/조건수 관점으로 해석하고 실험으로 검증한다.

## 2. 핵심 질문
- 왜 같은 모델인데 어떤 설정에서는 폭주하고 어떤 설정에서는 수렴하는가?
- sharpness를 줄이면 언제 일반화가 좋아지는가?
- curvature proxy를 실험 루프에 어떻게 넣을 것인가?

## 3. 핵심 개념
- Gradient `g = ∇L(θ)`: 1차 정보
- Hessian `H = ∇²L(θ)`: 2차 곡률 정보
- 로컬 근사: `L(θ+Δ) ≈ L + g^TΔ + 1/2 Δ^THΔ`
- Condition number: 큰 값일수록 anisotropic landscape

## 4. 연구자 관점
- 최적화 경로는 loss landscape geometry의 결과다.
- sharp/flat 논의는 Hessian spectrum을 동반할 때만 의미가 생긴다.
- normalization, residual, optimizer는 function class뿐 아니라 geometry를 바꾼다.

## 5. 엔지니어 관점
- 실전 점검 순서:
  1. 학습률/워밍업
  2. gradient norm 분포
  3. mixed precision overflow 여부
  4. 손실항 스케일 균형
  5. top curvature proxy
- 학습 불안정 시 LR만 줄이지 말고 곡률 지표를 같이 보자.

## 6. CV 예시
- 대규모 ViT pretrain에서 warmup 길이 부족 -> 초기 곡률 영역에서 발산.
- Detection에서 multi-head loss 불균형 -> gradient 방향 충돌 + conditioning 악화.
- Diffusion 모델에서 timestep별 곡률 편차 -> 스케줄링 전략 필요.

## 7. 딥러닝 예시
- SAM: sharp한 영역을 회피하는 방향으로 업데이트.
- Adaptive optimizer: 좌표축별 스케일 보정(일종의 preconditioning).
- Gradient clipping: 급격한 curvature 구간에서 업데이트 폭 제한.

## 8. 논문에서 보이는 형태
- "sharpness-aware optimization"
- "Hessian trace / top eigenvalue"
- "curvature regularization"
- "second-order approximation"

## 9. 구현 체크리스트
1. gradient norm과 loss를 동일 step에서 기록
2. 최소한 주기적으로 Hessian top-eig proxy 측정
3. warmup 길이 변경 실험을 독립 변수로 분리
4. 혼합정밀도 사용 시 overflow 로그 확인
5. batch size 변경 시 LR scaling rule 맹신 금지

## 10. 미니 실험 아이디어
```python
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(128, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)
x = torch.randn(64, 128)
y = torch.randint(0, 10, (64,))
loss = torch.nn.CrossEntropyLoss()(model(x), y)
params = [p for p in model.parameters() if p.requires_grad]

g = torch.autograd.grad(loss, params, create_graph=True)
v = [torch.randn_like(p) for p in params]
gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
hvp = torch.autograd.grad(gv, params)

num = sum((h * vi).sum() for h, vi in zip(hvp, v))
den = sum((vi * vi).sum() for vi in v)
print("rayleigh quotient (curvature proxy):", float(num / (den + 1e-12)))
```

## 11. 자주 나오는 오해
- "좋은 optimizer로 바꾸면 해결": 곡률 원인을 안 보면 같은 문제 재발.
- "Hessian은 너무 비싸서 무의미": 근사 지표(top eig/trace/HVP)만으로도 충분히 유용.
- "flat minima면 항상 좋다": 데이터 분포/정규화/모델 용량과 함께 해석해야 한다.

## 12. 연결 문서
- [허브 문서](./linear-algebra-for-cv.md)
- [행렬 미분/곡률 맥락](./eigens-svd-pca.md)
- [저랭크 효율화](./low-rank-efficient-models.md)
