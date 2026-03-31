# 챕터 6: 심화 주제 및 실전 응용

## 📚 개요

이 챕터에서는 머신러닝과 딥러닝에서 중요한 심화 주제와 실전 응용 사례들을 다룹니다.

---

## 📖 주요 내용

### 1. 베이지안 통계

#### 1.1 베이즈 정리
```python
P(θ|D) = P(D|θ) · P(θ) / P(D)
```
- **P(θ|D)**: 사후확률 (posterior)
- **P(D|θ)**: 우도 (likelihood)
- **P(θ)**: 사전확률 (prior)
- **P(D)**: 증거 (evidence)

#### 1.2 MAP (Maximum a Posteriori)
```python
θ_MAP = argmax_θ P(θ|D)
        = argmax_θ [P(D|θ) · P(θ)]
```

---

### 2. 정규화와의 연결

#### 2.1 Ridge 회귀 = MAP with Gaussian Prior
```python
Prior: θ ~ N(0, σ²·I)
Log prior: log P(θ) ∝ -||θ||²/(2σ²)

Cost: L(θ) = ||y - Xθ||² + λ||θ||²
```

#### 2.2 Lasso 회귀 = MAP with Laplace Prior
```python
Prior: θ ~ Laplace(0, b)
Log prior: log P(θ) ∝ -||θ||₁/b

Cost: L(θ) = ||y - Xθ||² + λ||θ||₁
```

---

### 3. 확률적 경사하강법 (SGD)

#### 3.1 기본 알고리즘
```python
for epoch in range(n_epochs):
    for batch in data:
        gradient = compute_gradient(batch)
        θ = θ - α · gradient
```

#### 3.2 모멘텀 SGD
```python
v = γ·v - α·∇L(θ)
θ = θ + v
```

---

### 4. 최적화 팁

#### 4.1 학습률 선택
- 너무 작음: 수렴 느림
- 너무 큼: 발산/진동
- 권장: 0.001 ~ 0.1 (Adam)

#### 4.2 배치 크기
- 작음: 빠른 업데이트 but 노이즈
- 큼: 안정적 but 느림
- 권장: 32, 64, 128, 256

#### 4.3 정규화 전략
- **Dropout**: 과적합 방지
- **BatchNorm**: 학습 안정화
- **Weight decay**: L2 정규화

---

### 5. 과적합 방지 전략

```
1. 데이터 증강 (Data Augmentation)
   - 회전, 이동, 크기변환
   - 색상 변형

2. 정규화 (Regularization)
   - L1, L2 weight decay
   - Dropout

3. 모델 단순화
   - 레이어 감소
   - 파라미터 감소

4. 조기종료 (Early Stopping)
   - validation loss 모니터링
   -最佳 시점에서 중단
```

---

## 🔧 실습 가이드

### 실습 1: Ridge vs Lasso
```python
from sklearn.linear_model import Ridge, Lasso
import numpy as np

# 데이터 생성
X = np.random.randn(100, 20)
y = X[:, :3].sum(axis=1) + np.random.randn(100) * 0.1

# Ridge (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
ridge_coef = ridge.coef_

# Lasso (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
lasso_coef = lasso.coef_

print(f"Ridge non-zeros: {np.sum(ridge_coef != 0)}")
print(f"Lasso non-zeros: {np.sum(lasso_coef != 0)}")
```

### 실습 2: Learning Rate Schedule
```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

model = YourModel()

# StepLR
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train(model)
    scheduler.step()
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()}")
```

### 실습 3: Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

---

## 📊 Regularization 효과 비교

| 기법 | 과적합 감소 | 학습 속도 | 계산량 | 권장 |
|------|----------|------|-----|----|
| L2 (Ridge) | 중간 | 빠름 | 없음 | 기본 |
| L1 (Lasso) | 중간 | 빠름 | 없음 | feature selection |
| Dropout | 높음 | 느림 | 없음 | 신경망 |
| BatchNorm | 중간 | 빠름 | 낮음 | 권장 |
| Data Aug | 높음 | 보통 | 중간 | 필수 |

---

## 📝 추가 학습 자료

- **핵심:** 6 장 전체 Jupyter 노트북 (chapter6.ipynb)
- **데이터:** data/ 디렉토리 확인
- **심화:** 베이지안 추론, MCMC, Variational Inference

---

## 🎯 학습 완료!

이제 `essential-mathematics-for-ai` 의 6 개 챕터 모두 완료되었습니다!

[시작하기: README.md](../README.md)

*최종 수정일: 2026 년 3 월*
