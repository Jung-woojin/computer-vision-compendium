# 챕터 3: 최적화 (Optimization)

## 📚 개요

이 챕터에서는 머신러닝 모델 학습의 핵심인 최적화 기법들을 다룹니다. 경사하강법부터 최신 최적화 알고리즘까지 폭넓게 다룹니다.

---

## 📖 주요 내용

### 1. 경사하강법 (Gradient Descent)

#### 1.1 기본 공식
```python
θ = θ - α · ∇J(θ)
```
- **θ**: 모델 파라미터
- **α**: 학습률 (learning rate)
- **∇J(θ)**: 비용 함수의 기울기

#### 1.2 변형들

**Mini-batch GD:**
- 배치 GD 보다 빠름
- 메모리 효율적
- 노이즈 존재

**Stochastic GD:**
- 매 샘플마다 업데이트
- 매우 빠르지만 불안정
- 수렴 진동

**모멘텀 (Momentum):**
```python
v = γ·v + α·∇J(θ)
θ = θ - v
```

**Adam optimizer:**
- 적응적 학습률
- 1 차, 2 차 모멘텀 사용
- 현재 가장 널리 사용

---

### 2. 최적화 알고리즘 비교

| 알고리즘 | 학습률 | 수렴속도 | 노이즈 | 사용도 |
|----------|--------|---------|--------|--------|
| Batch GD | 고정 | 느림 | 없음 | 기본 |
| Mini-batch GD | 고정 | 보통 | 있음 | 권장 |
| SGD | 고정 | 빠름 | 많음 | 단순작업 |
| Momentum | 고정 | 빠름 | 적음 | 권장 |
| Adam | 적응 | 매우 빠름 | 적음 | 가장 권장 |

---

### 3. 학습률 스케줄링

#### 3.1 학습률 감쇠
```python
α(t) = α₀ / (1 + γ·t)
```

#### 3.2 Step Decay
```python
α(t) = α₀ · γᵏ  (k: 에폭 카운트)
```

#### 3.3 Warmup
초기 학습률 낮게 시작 → 점진적 증가 → 감소

---

### 4. 구현 예제

#### 4.1 경사하강법 구현 (NumPy)
```python
import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def fit(self, X, y, n_epochs=1000):
        m, n = X.shape
        theta = np.zeros(n)
        
        for epoch in range(n_epochs):
            gradient = (2/m) * X.T.dot(X.dot(theta) - y)
            theta = theta - self.lr * gradient
        
        return theta
```

#### 4.2 Adam optimizer
```python
import torch
from torch.optim import Adam

model = YourModel()
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

---

## 🔧 실습 가이드

### 실습 1: 경사하강법 시각화
```python
import matplotlib.pyplot as plt

# 2D cost function
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = (X-2)**2 + (Y+1)**2

# 경사 하강 추적
path = []
theta = np.array([0, 0])
for i in range(100):
    grad = np.array([2*(theta[0]-2), 2*(theta[1]+1)])
    theta = theta - 0.1 * grad
    path.append(theta.copy())

path = np.array(path)
plt.contour(X, Y, Z)
plt.plot(path[:,0], path[:,1], 'r-', lw=2)
plt.show()
```

### 실습 2: 다양한 옵티마이저 비교
```python
from torch.optim import SGD, Adam, RMSprop

# 옵티마이저 테스트
optimizers = {
    'SGD': SGD(model.parameters(), lr=0.01),
    'Adam': Adam(model.parameters(), lr=0.001),
    'RMSprop': RMSprop(model.parameters(), lr=0.01)
}

for name, optimizer in optimizers.items():
    # 학습 루프
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
```

---

## 📝 추가 학습 자료

- **핵심:** 3 장 전체 Jupyter 노트북 (chapter3.ipynb)
- **시각화:** confusion matrix 이미지 포함
- **데이터:** data/ 디렉토리 확인

---

## 🎯 다음 챕터

[챕터 4: 행렬 분해 (Matrix Decompositions)](../chapter04/README.md)

*최종 수정일: 2026 년 3 월*
