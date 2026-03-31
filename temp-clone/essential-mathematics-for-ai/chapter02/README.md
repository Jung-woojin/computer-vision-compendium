# 챕터 2: 확률분포와 추정 (Probability Distributions & Estimation)

## 📚 개요

이 챕터에서는 인공지능과 머신러닝에서 필수적인 확률분포와 통계적 추정 방법을 다룹니다.

---

## 📖 주요 내용

### 1. 기본 확률분포

#### 1.1 정규분포 (Normal/Gaussian Distribution)
```python
f(x|μ,σ²) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
```
- **μ (mean)**: 평균
- **σ² (variance)**: 분산

#### 1.2 이항분포 (Binomial Distribution)
```python
P(X=k) = C(n,k) · pᵏ · (1-p)ⁿ⁻ᵏ
```

#### 1.3 포아송분포 (Poisson Distribution)
```python
P(X=k) = (λᵏ · e⁻λ) / k!
```

---

### 2. 다변량 정규분포 (Bivariate Normal)

**공분산 행렬 Σ:**
```
Σ = [[σ₁²,     ρσ₁σ₂],
     [ρσ₁σ₂,   σ₂²  ]]
```

**2D 샘플링 시뮬레이션 코드:**
```python
import numpy as np
import matplotlib.pyplot as plt

mu = [0, 0]
Sigma = [[1, 0.5], [0.5, 1]]

samples = np.random.multivariate_normal(mu, Sigma, 1000)
plt.scatter(samples[:,0], samples[:,1])
plt.show()
```

---

### 3. 함수 적합 (Fitting Functions)

#### 3.1 최소제곱법 (Least Squares)

**비용 함수:**
```python
J(θ) = Σᵢ(yᵢ - f(xᵢ;θ))²
```

**해:**
```python
θ = (XᵀX)⁻¹Xᵀy
```

#### 3.2 정규화 (Regularization)

**Ridge (L2):**
```python
J(θ) = Σᵢ(yᵢ - f(xᵢ;θ))² + λ||θ||²
```

**Lasso (L1):**
```python
J(θ) = Σᵢ(yᵢ - f(xᵢ;θ))² + λ||θ||₁
```

---

### 4. 최대우도추정 (Maximum Likelihood Estimation)

**로그 우도 함수:**
```python
log L(θ|X) = Σᵢ log P(xᵢ|θ)
```

**ML 추정량:**
```python
θ_ML = argmax_θ log L(θ|X)
```

---

## 🔧 실습 가이드

### 실습 1: 정규분포 샘플링과 시각화
```python
# 샘플 생성
samples = np.random.normal(loc=0, scale=1, size=1000)

# 히스토그램과 PDF 오버레이
plt.hist(samples, bins=50, density=True, alpha=0.6)
x = np.linspace(-4, 4, 100)
plt.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-x**2/2))
```

### 실습 2: 선형회귀 구현
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 생성
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# 모델 학습
model = LinearRegression()
model.fit(X, y)

print(f"절편: {model.intercept_}")
print(f"기울기: {model.coef_[0]}")
```

---

## 📝 추가 학습 자료

- **필독:** 2 장 전체 Jupyter 노트북 (chapter2_*.ipynb)
- **시각화:** plots/ 디렉토리 참고
- **데이터:** data/ 디렉토리 확인

---

## 🎯 다음 챕터

[챕터 3: 최적화 (Optimization)](../chapter03/README.md)

*최종 수정일: 2026 년 3 월*
