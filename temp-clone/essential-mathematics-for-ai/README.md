# 인공지능 수학 Deepdive 📐🧠

**완전 분석: 수학 이론, 증명, 코드 구현 및 실전 응용**

> 🔥 **핵심 철학**: "수학은 머신러닝의 언어입니다. 이론적 배경을 이해해야 모델의 본질을 이해할 수 있습니다."

---

## 📚 목차

- [개요](#-개요)
- [Chapter 2: 확률분포와 추정](#-chapter-2-확률분포와-추정)
- [Chapter 3: 최적화](#-chapter-3-최적화)
- [Chapter 4: 행렬 분해](#-chapter-4-행렬-분해)
- [Chapter 5: 선형대수 심화](#-chapter-5-선형대수-심화)
- [Chapter 6: 심화 주제](#-chapter-6-심화-주제)
- [실전 도구](#-실전-도구)
- [PhD 연구 가이드](#-phd-연구-가이드)

---

## 🎯 개요

본 레포지토리는 **수학 이론**과 **코드 구현**의 연결을 통해 머신러닝/딥러닝의 본질을 이해합니다.

**학습 목표:**
1. **수학적 엄밀성**: 증명과 정리를 통한 깊은 이해
2. **코드 구현**: NumPy, PyTorch 로 직접 구현
3. **시각화**: Jupyter Notebook 으로 직관적 이해
4. **실전 응용**: 실제 문제에 적용

---

## 📖 Chapter 2: 확률분포와 추정

### 2.1 확률분포 이론

#### 정규분포 (Normal Distribution)

**확률밀도함수:**
```
f(x) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))

where:
- μ: mean (기대값)
- σ²: variance (분산)
- σ: standard deviation
```

**통계적 성질:**
- **Expected Value**: E[X] = μ
- **Variance**: Var(X) = σ²
- **Standard Deviation**: std(X) = σ

**Python 구현:**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class NormalDistribution:
    """정규분포 클래스"""
    
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    def pdf(self, x):
        """Probability Density Function"""
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
    
    def cdf(self, x):
        """Cumulative Distribution Function"""
        return 0.5 * (1 + stats.erf((x - self.mu) / (self.sigma * np.sqrt(2))))
    
    def sample(self, n_samples=1000):
        """Sample from normal distribution"""
        return np.random.normal(self.mu, self.sigma, n_samples)
    
    def log_pdf(self, x):
        """Log probability density function"""
        return -0.5 * np.log(2 * np.pi * self.sigma**2) - \
               (x - self.mu)**2 / (2 * self.sigma**2)
    
    def __repr__(self):
        return f"Normal(μ={self.mu}, σ={self.sigma})"
    
    def plot(self, x_range=None, n_points=1000):
        """Plot PDF and CDF"""
        if x_range is None:
            x_range = (self.mu - 3*self.sigma, self.mu + 3*self.sigma)
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = self.pdf(x)
        y_cdf = self.cdf(x)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # PDF
        axes[0].plot(x, y, 'b-', linewidth=2)
        axes[0].set_title('Probability Density Function')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(self.mu, color='r', linestyle='--', label=f'mean={self.mu}')
        axes[0].legend()
        
        # CDF
        axes[1].plot(x, y_cdf, 'g-', linewidth=2)
        axes[1].set_title('Cumulative Distribution Function')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('F(x)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# 시각화
normal = NormalDistribution(mu=0, sigma=1)
normal.plot()
plt.show()
```

#### 다변량 정규분포 (Multivariate Normal Distribution)

**확률밀도함수:**
```
f(x) = (1/(√((2π)^d |Σ|))) · exp(-0.5 · (x-μ)^T Σ^(-1) (x-μ))

where:
- μ: d×1 mean vector
- Σ: d×d covariance matrix
- d: dimension
- |Σ|: determinant of Σ
```

**Python 구현:**
```python
class MultivariateNormal:
    """다변량 정규분포"""
    
    def __init__(self, mu, Sigma):
        """
        Args:
            mu: d×1 mean vector
            Sigma: d×d covariance matrix (symmetric positive-definite)
        """
        self.mu = np.array(mu)
        self.Sigma = np.array(Sigma)
        self.d = len(mu)
        
        # Validate covariance matrix
        assert self.Sigma.shape == (self.d, self.d)
        assert np.allclose(self.Sigma, self.Sigma.T), "Covariance matrix must be symmetric"
        assert np.all(np.linalg.eigvalsh(self.Sigma) > 0), "Covariance matrix must be positive-definite"
        
        # Precompute inverse and determinant
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.Sigma_det = np.linalg.det(self.Sigma)
    
    def pdf(self, x):
        """
        Probability density function
        
        Args:
            x: d×1 vector
        
        Returns:
            PDF value
        """
        x = np.atleast_1d(x)
        diff = x - self.mu
        
        exponent = -0.5 * np.dot(diff, np.dot(self.Sigma_inv, diff))
        pre_factor = 1 / np.sqrt((2 * np.pi)**self.d * self.Sigma_det)
        
        return pre_factor * np.exp(exponent)
    
    def log_pdf(self, x):
        """Log probability density function"""
        x = np.atleast_1d(x)
        diff = x - self.mu
        
        exponent = -0.5 * np.dot(diff, np.dot(self.Sigma_inv, diff))
        log_pre_factor = -0.5 * (self.d * np.log(2 * np.pi) + np.log(self.Sigma_det))
        
        return log_pre_factor + exponent
    
    def sample(self, n_samples=1000):
        """Generate samples using Cholesky decomposition"""
        L = np.linalg.cholesky(self.Sigma)
        z = np.random.randn(n_samples, self.d)
        samples = self.mu + np.dot(z, L.T)
        return samples
    
    def eigendecomposition(self):
        """Eigenvalue decomposition of covariance matrix"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.Sigma)
        return eigenvalues, eigenvectors
    
    def plot_ellipse(self, n_points=100, scale=1):
        """Plot 2D confidence ellipse"""
        assert self.d == 2, "Only works for 2D case"
        
        eigenvalues, eigenvectors = self.eigendecomposition()
        
        angles = np.linspace(0, 2*np.pi, n_points)
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Transform to ellipse
        scale_factors = np.sqrt(eigenvalues) * scale
        ellipse_x = x * scale_factors[0]
        ellipse_y = y * scale_factors[1]
        
        # Rotate
        rotated_x = (ellipx * eigenvectors[0,0] - ellipse_y * eigenvectors[0,1])
        rotated_y = (ellipx * eigenvectors[1,0] + ellipse_y * eigenvectors[1,1])
        
        # Shift to mean
        ellipse_x = rotated_x + self.mu[0]
        ellipse_y = rotated_y + self.mu[1]
        
        return ellipse_x, ellipse_y, eigenvalues, eigenvectors
    
    def plot(self):
        """Plot 2D distribution"""
        assert self.d == 2, "Only works for 2D case"
        
        n_samples = 10000
        samples = self.sample(n_samples)
        
        ellipse_x, ellipse_y, eigvals, eigvecs = self.plot_ellipse(scale=2)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1, label='Samples')
        plt.plot(ellipse_x, ellipse_y, 'r-', linewidth=2, label='2σ ellipse')
        plt.plot(self.mu[0], self.mu[1], 'r*', markersize=15, label='Mean')
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title(f'Multivariate Normal\nμ={self.mu}, Covariance eigenvalues: {eigvals}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

# 시각화
mu = np.array([0, 0])
Sigma = np.array([[1, 0.5], [0.5, 1]])
mvn = MultivariateNormal(mu, Sigma)
mvn.plot()
```

#### 함수 적합 (Function Fitting)

**Least Squares:**
```
min_w ||Xw - y||²

where:
- X: n×d design matrix
- w: d×1 weights
- y: n×1 target
```

**해:**
```
ŵ = (XᵀX)^(-1) Xᵀy
```

**Regularization (Ridge Regression):**
```
min_w ||Xw - y||² + λ||w||²

해: ŵ = (XᵀX + λI)^(-1) Xᵀy
```

```python
class LeastSquares:
    """Least Squares Fitting"""
    
    def fit(self, X, y):
        """
        Fit least squares model
        
        Args:
            X: n×d design matrix
            y: n×1 target vector
        
        Returns:
            w: d×1 weights
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Normal equation
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)
        w = XtX_inv @ X.T @ y
        
        self.weights = w.flatten()
        self.intercept = y.mean() - X.mean() @ w
        return self
    
    def predict(self, X):
        """Predict using fitted model"""
        X = np.array(X)
        return X @ self.weights + self.intercept
    
    def score(self, X, y):
        """R² score"""
        y = np.array(y)
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        
        r2 = 1 - (ss_res / ss_tot)
        return r2

class RidgeRegression:
    """Ridge Regression with regularization"""
    
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: regularization strength
        """
        self.alpha = alpha
    
    def fit(self, X, y):
        """
        Fit ridge regression
        
        Args:
            X: n×d design matrix
            y: n×1 target vector
        
        Returns:
            w: d×1 weights
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n, d = X.shape
        
        # Regularized normal equation
        XtX = X.T @ X + self.alpha * np.eye(d)
        XtX_inv = np.linalg.pinv(XtX)
        w = XtX_inv @ X.T @ y
        
        self.weights = w.flatten()
        self.intercept = y.mean() - X.mean() @ w
        return self
    
    def score(self, X, y):
        """R² score"""
        y = np.array(y)
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        
        r2 = 1 - (ss_res / ss_tot)
        return r2

# 사용 예시
X = np.random.randn(100, 3)
y = 2*X[:, 0] - 3*X[:, 1] + 0.5*X[:, 2] + 1 + np.random.randn(100) * 0.5

ols = LeastSquares()
ols.fit(X, y)
print(f"OLS R²: {ols.score(X, y):.4f}")
print(f"OLS weights: {ols.weights}")

ridge = RidgeRegression(alpha=1.0)
ridge.fit(X, y)
print(f"Ridge R²: {ridge.score(X, y):.4f}")
print(f"Ridge weights: {ridge.weights}")
```

---

### 2.2 최대우도추정 (Maximum Likelihood Estimation)

**개념:** 파라미터 θ 가 주어졌을 때, 관측 데이터의 우도 (likelihood) 를 최대화하는 θ 찾음

**Likelihood 함수:**
```
L(θ) = P(X | θ) = ∏ᵢ₌₁ⁿ P(xᵢ | θ)
```

**Log-Likelihood:**
```
ℓ(θ) = log L(θ) = Σᵢ₌₁ⁿ log P(xᵢ | θ)
```

**Normal Distribution MLE:**
```
ℓ(μ, σ²) = -n/2·log(2π) - n/2·log(σ²) - 1/(2σ²)·Σᵢ(xᵢ - μ)²

∂ℓ/∂μ = Σᵢ(xᵢ - μ) = 0  →  μ̂ = Σᵢxᵢ / n = x̄

∂ℓ/∂σ² = -n/(2σ²) + 1/(2σ⁴)·Σᵢ(xᵢ - μ)² = 0  →  σ̂² = Σᵢ(xᵢ - μ̂)² / n
```

```python
class MaximumLikelihoodEstimator:
    """Maximum Likelihood Estimator"""
    
    def fit_normal(self, data):
        """
        MLE for Normal distribution
        
        Args:
            data: 1D array of samples
        
        Returns:
            μ̂: estimated mean
            σ̂²: estimated variance
        """
        data = np.array(data)
        n = len(data)
        
        # MLE for mean
        mu_hat = np.mean(data)
        
        # MLE for variance (biased estimator)
        sigma2_hat = np.sum((data - mu_hat)**2) / n
        
        return mu_hat, sigma2_hat
    
    def fit_logistic(self, X, y):
        """
        MLE for Logistic Regression
        
        Args:
            X: n×d design matrix
            y: n×1 binary labels
        
        Returns:
            w: learned weights
        """
        X = np.array(X)
        y = np.array(y)
        
        # Initialize weights
        w = np.zeros(X.shape[1])
        
        # Gradient descent
        learning_rate = 0.1
        n_samples, n_features = X.shape
        
        for epoch in range(1000):
            # Linear predictions
            z = np.dot(X, w)
            
            # Sigmoid
            predictions = 1 / (1 + np.exp(-z))
            
            # Gradient
            gradient = (1/n_samples) * np.dot(X.T, (predictions - y))
            
            # Update
            w = w - learning_rate * gradient
        
        self.weights = w
        return w

# 사용 예시
np.random.seed(42)
data = np.random.normal(5, 2, 1000)

mle = MaximumLikelihoodEstimator()
mu_hat, var_hat = mle.fit_normal(data)

print(f"True μ = 5, μ̂ = {mu_hat:.4f}")
print(f"True σ² = 4, σ̂² = {var_hat:.4f}")
```

---

## 📊 Chapter 3: 최적화

### 3.1 경사하강법 (Gradient Descent)

**수학적 배경:**
```
θ_new = θ_old - η · ∇f(θ_old)

where:
- η: learning rate
- ∇f: gradient of loss function
```

**경사하강법 variants:**

1. **Batch GD:** 전체 데이터로 gradient 계산
2. **Stochastic GD (SGD):** 한 샘플로 gradient 계산
3. **Mini-batch GD:** 일부 데이터로 gradient 계산

```python
class GradientDescent:
    """Gradient Descent Optimizer"""
    
    def __init__(self, learning_rate=0.01):
        """
        Args:
            learning_rate: 기본 학습률
        """
        self.learning_rate = learning_rate
    
    def update(self, weights, gradients):
        """
        Update weights using gradient descent
        
        Args:
            weights: current weights
            gradients: gradient of loss
        
        Returns:
            new_weights: updated weights
        """
        new_weights = weights - self.learning_rate * gradients
        return new_weights
    
    def __repr__(self):
        return f"GradientDescent(learning_rate={self.learning_rate})"

class SGD:
    """Stochastic Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        Args:
            learning_rate: 학습률
            momentum: 모멘텀 (0~1)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, weights, gradients):
        """
        SGD with momentum
        
        Args:
            weights: current weights
            gradients: gradient
        
        Returns:
            new_weights: updated weights
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)
        
        # Velocity update
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        
        # Weight update
        new_weights = weights + self.velocity
        
        return new_weights
    
    def reset(self):
        """Reset velocity"""
        self.velocity = None

class Adam:
    """Adam Optimizer"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Args:
            learning_rate: 학습률
            beta1: 첫 번째 모멘트 감가계수
            beta2: 두 번째 모멘트 감가계수
            eps: 수치적 안정성
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None
    
    def update(self, weights, gradients):
        """
        Adam optimization
        
        Args:
            weights: current weights
            gradients: gradient
        
        Returns:
            new_weights: updated weights
        """
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update weights
        new_weights = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return new_weights
    
    def reset(self):
        """Reset internal state"""
        self.t = 0
        self.m = None
        self.v = None

# 사용 예시
def quadratic(x):
    """Test function: f(x) = (x-2)²"""
    return (x - 2)**2

def gradient(x):
    """Gradient: f'(x) = 2(x-2)"""
    return 2 * (x - 2)

# GD
gd = GradientDescent(learning_rate=0.1)
x = np.array([10.0])
for i in range(50):
    grad = gradient(x)
    x = gd.update(x, grad)
    if i % 10 == 0:
        print(f"Step {i}: x = {x[0]:.4f}, f(x) = {quadratic(x[0]):.4f}")

# SGD
sgd = SGD(learning_rate=0.1, momentum=0.9)
x = np.array([10.0])
for i in range(50):
    grad = gradient(x)
    x = sgd.update(x, grad)
    if i % 10 == 0:
        print(f"SGD Step {i}: x = {x[0]:.4f}, f(x) = {quadratic(x[0]):.4f}")
```

### 3.2 학습률 스케줄링

**Learning Rate Scheduling:**

1. **Step Decay:** 일정 epochs 마다 학습률 감소
2. **Exponential Decay:** 지수적으로 감소
3. **Cosine Annealing:** 코사인 곡선으로 감소

```python
import numpy as np

class LearningRateScheduler:
    """Learning Rate Scheduler"""
    
    def __init__(self, initial_lr=0.01, final_lr=1e-6):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
    
    def get_lr(self, epoch, max_epochs):
        """Get learning rate for epoch"""
        raise NotImplementedError

class StepLR(LearningRateScheduler):
    """Step Learning Rate Scheduler"""
    
    def __init__(self, initial_lr=0.01, step_size=30, gamma=0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch, max_epochs=None):
        """
        LR = initial_lr * gamma^(epoch // step_size)
        """
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))

class ExponentialLR(LearningRateScheduler):
    """Exponential Learning Rate Scheduler"""
    
    def __init__(self, initial_lr=0.01, decay_rate=0.95):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
    
    def get_lr(self, epoch, max_epochs=None):
        """
        LR = initial_lr * decay_rate^epoch
        """
        return self.initial_lr * (self.decay_rate ** epoch)

class CosineAnnealingLR(LearningRateScheduler):
    """Cosine Annealing Learning Rate Scheduler"""
    
    def __init__(self, initial_lr=0.01, final_lr=1e-6):
        super().__init__(initial_lr, final_lr)
    
    def get_lr(self, epoch, max_epochs):
        """
        LR = final_lr + 0.5 * (initial_lr - final_lr) * (1 + cos(π * epoch / max_epochs))
        """
        if max_epochs is None:
            raise ValueError("max_epochs required for CosineAnnealingLR")
        
        cosine_schedule = 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
        lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * cosine_schedule
        
        return lr

# 사용 예시
max_epochs = 100

lr_schedulers = [
    ("StepLR", StepLR(step_size=30, gamma=0.1)),
    ("ExponentialLR", ExponentialLR(decay_rate=0.95)),
    ("CosineAnnealingLR", CosineAnnealingLR())
]

plt.figure(figsize=(12, 6))

for name, scheduler in lr_schedulers:
    lrs = [scheduler.get_lr(i, max_epochs) for i in range(max_epochs)]
    plt.plot(lrs, label=name)

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedulers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

---

## 📐 Chapter 4: 행렬 분해

### 4.1 LU 분해

**개념:** A = L · U

```
A = [a₁₁ a₁₂] = [1   0] · [a₁₁  a₁₂]
    [a₂₁ a₂₂]   [l₂₁ 1]   [0    a₂₂']
```

**Python 구현:**
```python
import numpy as np

def lu_decomposition(A):
    """
    LU Decomposition without pivoting
    
    Args:
        A: n×n matrix
    
    Returns:
        L: n×n lower triangular matrix
        U: n×n upper triangular matrix
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Initialize L with diagonal elements
    np.fill_diagonal(L, 1)
    
    # LU decomposition
    for i in range(n):
        # Upper triangular part
        for j in range(i, n):
            U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])
        
        # Lower triangular part
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i]
    
    return L, U

# 사용 예시
A = np.array([[4, 3],
              [6, 3]])

L, U = lu_decomposition(A)
print("Original A:")
print(A)
print("\nL matrix:")
print(L)
print("\nU matrix:")
print(U)
print("\nL @ U:")
print(L @ U)
```

### 4.2 QR 분해

**개념:** A = Q · R

- **Q**: orthogonal matrix (QᵀQ = I)
- **R**: upper triangular matrix

**Gram-Schmidt Process:**
```
u₁ = v₁
e₁ = u₁ / ||u₁||

u₂ = v₂ - proj_{e₁}(v₂)
e₂ = u₂ / ||u₂||

...

uₙ = vₙ - Σᵢ₌₁ⁿ⁻¹ proj_{eᵢ}(vₙ)
eₙ = uₙ / ||uₙ||

Q = [e₁ e₂ ... eₙ]
R = QᵀA
```

```python
def gram_schmidt(A):
    """
    QR Decomposition using Gram-Schmidt process
    
    Args:
        A: m×n matrix
    
    Returns:
        Q: m×n orthogonal matrix
        R: n×n upper triangular matrix
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    # Copy columns of A
    A_cols = [A[:, i] for i in range(n)]
    
    # Gram-Schmidt process
    for j in range(n):
        # u = v - Σ proj
        u = A_cols[j].copy()
        
        for i in range(j):
            R[i, j] = np.dot(A_cols[j], Q[:, i])
            u -= R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]
    
    return Q, R

# 사용 예시
A = np.array([[1, 1],
              [1, 0],
              [1, 1]])

Q, R = gram_schmidt(A)
print("Original A:")
print(A)
print("\nQ matrix:")
print(Q)
print("\nR matrix:")
print(R)
print("\nQ @ R:")
print(Q @ R)
```

### 4.3 SVD (특이값 분해)

**개념:** A = U · Σ · Vᵀ

- **U**: m×m orthogonal matrix (left singular vectors)
- **Σ**: m×n diagonal matrix (singular values)
- **Vᵀ**: n×n orthogonal matrix (right singular vectors transpose)

```python
def svd_decomposition(A, k=None):
    """
    Singular Value Decomposition with optional truncation
    
    Args:
        A: m×n matrix
        k: optional truncation rank (low-rank approximation)
    
    Returns:
        U: m×m orthogonal matrix
        S: singular values (diagonal of Σ)
        Vt: n×n orthogonal matrix (transpose)
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    if k is not None and k < len(S):
        # Truncated SVD
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]
    
    return U, S, Vt

def low_rank_approximation(A, k):
    """
    Low-rank approximation using SVD
    
    Args:
        A: m×n matrix
        k: target rank
    
    Returns:
        A_approx: k-rank approximation of A
    """
    U, S, Vt = svd_decomposition(A, k)
    
    # Reconstruct matrix
    S_diag = np.diag(S)
    A_approx = U @ S_diag @ Vt
    
    return A_approx, U, S, Vt

# 사용 예시: 이미지 압축
from PIL import Image

def compress_image(image_path, k):
    """
    Compress image using SVD
    
    Args:
        image_path: path to image file
        k: number of singular values to keep
    
    Returns:
        compressed_image: compressed image
    """
    img = np.array(image.open(image_path))
    
    # Convert to grayscale
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    
    m, n = img.shape
    
    # Compute SVD
    U, S, Vt = svd_decomposition(img, k)
    
    # Reconstruct
    S_diag = np.diag(S)
    img_approx = U @ S_diag @ Vt
    
    return img_approx, U, S, Vt

# 사용 예시
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

U, S, Vt = svd_decomposition(A)
print("Singular values:", S)

# Low-rank approximation
k = 2
A_approx, U_k, S_k, Vt_k = low_rank_approximation(A, k)
print(f"\nApproximation with k={k}:")
print(A_approx)
print(f"Error: {np.linalg.norm(A - A_approx):.4f}")
```

### 4.4 Eigenvalue Decomposition

**개념:** A = P · Λ · P⁻¹

- **P**: eigenvectors columns
- **Λ**: diagonal matrix of eigenvalues

```python
def eigenvalue_decomposition(A):
    """
    Eigenvalue decomposition
    
    Args:
        A: n×n square matrix
    
    Returns:
        eigenvalues: array of eigenvalues
        eigenvectors: matrix of eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    return eigenvalues, eigenvectors

# 사용 예시
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = eigenvalue_decomposition(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
```

---

## 🎨 Chapter 5: 선형대수 심화 & CV 응용

### 5.1 이미지 필터링

**Gaussian Filter:**

```
G(x, y) = (1/(2πσ²)) · exp(-(x² + y²)/(2σ²))
```

```python
def gaussian_filter(size=5, sigma=1.0):
    """
    Generate Gaussian filter kernel
    
    Args:
        size: filter size (must be odd)
        sigma: standard deviation
    
    Returns:
        kernel: Gaussian filter kernel
    """
    assert size % 2 == 1, "Filter size must be odd"
    
    center = size // 2
    
    # Create coordinate grid
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x -= center
    y -= center
    
    # Gaussian function
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize
    
    return kernel

def apply_convolution(image, kernel):
    """
    Apply 2D convolution
    
    Args:
        image: 2D image (grayscale)
        kernel: 2D filter kernel
    
    Returns:
        output: convolved image
    """
    H, W = image.shape
    kH, kW = kernel.shape
    
    padH = kH // 2
    padW = kW // 2
    
    # Padding
    padded = np.pad(image, pad_width=((padH, padH), (padW, padW)), mode='reflect')
    
    # Convolution
    output = np.zeros_like(image)
    
    for i in range(H):
        for j in range(W):
            output[i, j] = np.sum(padded[i:i+kH, j:j+kW] * kernel)
    
    return output

# 사용 예시
image = np.random.rand(100, 100)  # Test image
kernel = gaussian_filter(size=5, sigma=1.0)
output = apply_convolution(image, kernel)
```

**Laplacian Filter:**

```
L = [ 0  -1   0 ]
    [ -1   4  -1 ]
    [ 0  -1   0 ]
```

```python
def laplacian_filter():
    """
    Laplacian filter kernel
    
    Returns:
        kernel: 3×3 Laplacian filter
    """
    return np.array([[ 0, -1,  0],
                     [-1,  4, -1],
                     [ 0, -1,  0]], dtype=np.float32)

def apply_laplacian(image):
    """
    Apply Laplacian filter for edge detection
    
    Args:
        image: 2D image
    
    Returns:
        edges: edge-detected image
    """
    kernel = laplacian_filter()
    edges = apply_convolution(image, kernel)
    return np.abs(edges)
```

### 5.2 가버 필터 (Gabor Filter)

**수식:**
```
G(x, y; λ, θ, φ, σ, γ) = exp(-π²·(x'² + γ²·y'²) / (2·σ²))
                         · cos(2π·x'/λ + φ)

where:
  x' = x·cosθ + y·sinθ
  y' = -x·sinθ + y·cosθ
```

```python
def gabor_filter(frequency=0.1, theta=0, sigma=7, phase=0, gamma=1.0):
    """
    Generate Gabor filter kernel
    
    Args:
        frequency: frequency of sinusoidal factor (cycles/pixel)
        theta: orientation in radians
        sigma: standard deviation of Gaussian envelope
        phase: phase offset in radians
        gamma: spatial aspect ratio
    
    Returns:
        kernel: Gabor filter kernel
    """
    size = int(np.ceil(2 * np.pi * sigma / frequency)) + 1
    if size % 2 == 0:
        size += 1
    
    half = size // 2
    
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)
    
    # Rotation
    theta_rad = theta
    x_rot = X * np.cos(theta_rad) + Y * np.sin(theta_rad)
    y_rot = -X * np.sin(theta_rad) + Y * np.cos(theta_rad)
    
    # Gabor function
    kernel = np.exp(-np.pi**2 * (x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2))
    kernel *= np.cos(2 * np.pi * frequency * x_rot + phase)
    kernel = kernel / np.abs(kernel).max()
    
    return kernel

# 사용 예시
gabor = gabor_filter(frequency=0.1, theta=np.pi/4, sigma=7, phase=0, gamma=1.0)
```

---

## 🧪 Chapter 6: 심화 주제 & 실전 응용

### 6.1 베이지안 통계

**베이즈 정리:**
```
P(θ|X) = P(X|θ) · P(θ) / P(X)

where:
- P(θ|X): posterior probability
- P(X|θ): likelihood
- P(θ): prior probability
- P(X): evidence (marginal likelihood)
```

**MAP (Maximum A Posteriori):**
```
θ_MAP = argmax_θ P(θ|X)
      = argmax_θ P(X|θ) · P(θ)
```

```python
class BayesianRegression:
    """Bayesian Linear Regression"""
    
    def __init__(self, prior_mean=0, prior_var=1):
        """
        Bayesian linear regression
        
        Args:
            prior_mean: prior mean for weights
            prior_var: prior variance for weights
        """
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.posterior_mean = None
        self.posterior_var = None
    
    def fit(self, X, y):
        """
        Fit Bayesian linear regression
        
        Args:
            X: n×d design matrix
            y: n×1 target
        """
        n, d = X.shape
        
        # Prior precision
        prior_precision = 1 / self.prior_var
        
        # Data precision
        data_precision = n / 1.0  # Assume σ² = 1 for simplicity
        
        # Posterior precision
        posterior_precision = prior_precision + data_precision
        
        # Prior mean
        prior_mean = np.full(d, self.prior_mean)
        
        # Likelihood mean (MLE)
        likelihood_mean = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Posterior mean (weighted average)
        self.posterior_mean = (prior_precision * prior_mean + data_precision * likelihood_mean) / posterior_precision
        
        # Posterior variance
        self.posterior_var = 1 / posterior_precision
    
    def predict(self, X):
        """
        Predict using posterior mean
        
        Args:
            X: n×d design matrix
        
        Returns:
            predictions: n×1 predicted values
        """
        return X @ self.posterior_mean
    
    def predictive_variance(self, X):
        """
        Predictive variance
        
        Args:
            X: n×d design matrix
        
        Returns:
            variance: predictive variance for each prediction
        """
        return np.ones(X.shape[0]) * self.posterior_var

# 사용 예시
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] - 3*X[:, 1] + 0.5*X[:, 2] + 1 + np.random.randn(100) * 0.5

bayes_reg = BayesianRegression(prior_mean=0, prior_var=1.0)
bayes_reg.fit(X, y)
print(f"Posterior mean: {bayes_reg.posterior_mean}")
print(f"Posterior variance: {bayes_reg.posterior_var:.4f}")

# Predictions
y_pred = bayes_reg.predict(X)
r2 = bayes_reg.score(X, y)
print(f"R²: {r2:.4f}")
```

### 6.2 정규화와의 연결

**Ridge Regression (L2 Regularization):**

```
min_w ||Xw - y||² + λ||w||²

等价于 Bayesian Regression with Gaussian prior
```

**Lasso (L1 Regularization):**

```
min_w ||Xw - y||² + λ||w||₁

等价于 Bayesian Regression with Laplace prior
```

```python
class BayesianRidgeRegression(BayesianRegression):
    """Bayesian Ridge Regression (L2 regularization)"""
    
    def fit(self, X, y, reg_strength=1.0):
        """
        Fit Bayesian Ridge Regression
        
        Args:
            X: n×d design matrix
            y: n×1 target
            reg_strength: regularization strength (λ)
        """
        n, d = X.shape
        
        # Prior precision (inverse of variance)
        prior_precision = reg_strength
        
        # Data precision
        data_precision = n
        
        # Posterior precision
        posterior_precision = prior_precision * np.eye(d) + X.T @ X
        
        # Prior mean
        prior_mean = np.zeros(d)
        
        # Likelihood mean
        likelihood_mean = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Posterior mean
        self.posterior_mean = np.linalg.solve(posterior_precision, prior_precision * prior_mean + X.T @ y)
        
        # Posterior covariance
        self.posterior_cov = np.linalg.inv(posterior_precision)
        
        return self
    
    def predict(self, X):
        """Predict using posterior mean"""
        return X @ self.posterior_mean

# 사용 예시
ridge = BayesianRidgeRegression()
ridge.fit(X, y, reg_strength=1.0)
print(f"Ridge coefficients: {ridge.posterior_mean}")
```

---

## 📚 참고 자료

### 필수 서적
1. **Goodfellow et al.**, "Deep Learning"
2. **Bishop**, "Pattern Recognition and Machine Learning"
3. **Gonzalez & Woods**, "Digital Image Processing"
4. **Strang**, "Linear Algebra and Its Applications"

### 참고 문헌
1. **Canny**, "A Computational Approach to Edge Detection"
2. **Marr & Hildreth**, "Theory of Edge Detection"
3. **He et al.**, "Deep Residual Learning"
4. **Dosovitskiy et al.**, "An Image is Worth 16x16 Words"

---

## 🚀 실전 도구

### NumPy 수학 유틸리티

```python
class MathUtilities:
    """수학 계산 유틸리티"""
    
    @staticmethod
    def compute_gradient(func, x, epsilon=1e-8):
        """Compute gradient using finite differences"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
        return grad
    
    @staticmethod
    def eigenvalues_symmetric(A):
        """Compute eigenvalues of symmetric matrix"""
        return np.linalg.eigvalsh(A)
    
    @staticmethod
    def svd_rank(A, threshold=1e-10):
        """Compute effective rank using SVD"""
        U, S, Vt = np.linalg.svd(A)
        rank = np.sum(S > threshold)
        return rank
```

---

## 🎓 PhD 연구 가이드

### 연구 주제 후보

#### 1. Bayesian Deep Learning
**문제**: 딥러닝의 불확실성 정량화

**연구 방향**:
- Variational Inference 기반 불확실성
- Monte Carlo Dropout
- Bayesian Neural Networks

#### 2. Optimizer Design
**문제**: 최적화 알고리즘의 한계

**연구 방향**:
- Adaptive learning rate design
- Second-order optimization
- Non-convex optimization theory

#### 3. Linear Algebra in Deep Learning
**문제**: 행렬 분해의 딥러닝 적용

**연구 방향**:
- SVD-based compression
- Eigenvalue dynamics in training
- Matrix factorization in neural networks

---

## 📝 라이선스

MIT License

---

*최종 업데이트: 2026-03-31*
*수학 이론과 코드의 완벽한 연결을 위한 심화 자료*
