# 챕터 4: 행렬 분해 (Matrix Decompositions)

## 📚 개요

이 챕터에서는 행렬을 더 간단한 구성요소로 분해하는 방법들을 다룹니다. 기계학습, 데이터 분석, 수치 해석에서 핵심적인 도구들입니다.

---

## 📖 주요 내용

### 1. LU 분해 (LU Decomposition)

**수식:**
```
A = LU
```
- **L**: 하삼각행렬 (Lower triangular)
- **U**: 상삼각행렬 (Upper triangular)

**Python 구현:**
```python
import numpy as np
from scipy.linalg import lu

A = np.array([[2, 1], [1, 3]])
P, L, U = lu(A)
```

**응용:**
- 연립방정식 해결
- 행렬식 계산

---

### 2. QR 분해 (QR Decomposition)

**수식:**
```
A = QR
```
- **Q**: 직교행렬 (QᵀQ = I)
- **R**: 상삼각행렬

**최소제곱법 해:**
```python
# Ax = b 에 대한 최소제곱해
x = R_inv @ Q.T @ b
```

---

### 3. 특이값 분해 (SVD) ⭐

**수식:**
```
A = UΣVᵀ
```
- **U**: m×m 직교행렬 (왼쪽 특이벡터)
- **Σ**: m×n 대각행렬 (특이값 σ₁ ≥ σ₂ ≥ ... ≥ 0)
- **V**: n×n 직교행렬 (오른쪽 특이벡터)

**Python:**
```python
U, Sigma, Vt = np.linalg.svd(A)
```

**응용:**
- 차원 축소 (PCA)
- 이미지 압축
- 노이즈 제거
- 가역행렬 (pseudoinverse)

**이미지 압축 예제:**
```python
# 원본 이미지
img = np.array(pil_image)

# SVD
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# k 개의 특이값만 사용
k = 50
img_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

---

### 4. 대각화 (Diagonalization)

**수식:**
```
A = PDP⁻¹
```
- **P**: 고유벡터 행렬
- **D**: 고유값 대각행렬

**조건:** 행렬이 n 개의 선형독립 고유벡터를 가져야 함

---

### 5. 코시 분해 (Cholesky Decomposition)

**수식:**
```
A = LLᵀ
```
- **A**: 대칭 양정치행렬
- **L**: 하삼각행렬

**Python:**
```python
L = np.linalg.cholesky(A)
```

**장점:**
- 계산 효율적
- 수치적 안정성

---

## 🔧 실습 가이드

### 실습 1: SVD 로 이미지 압축
```python
import numpy as np
from PIL import Image

# 이미지 로드
img = np.array(Image.open('image.jpg').convert('L'))

# SVD 수행
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# 압축률 비교
for k in [10, 20, 50, 100]:
    img_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    compression = k / min(img.shape)
    print(f"k={k}, 압축률={compression:.2%}")
```

### 실습 2: PCA 와 SVD
```python
from sklearn.decomposition import PCA
import numpy as np

# 데이터 준비
X = np.random.randn(100, 50)

# PCA 수행
pca = PCA(n_components=10)
X_transformed = pca.fit_transform(X)

# SVD 로 직접 구현
U, S, Vt = np.linalg.svd(X, full_matrices=False)
X_svd = U[:, :10] @ np.diag(S[:10])
```

---

## 📊 주요 행렬 분해 비교

| 분해 | 적용 조건 | 계산량 | 주요 응용 |
|------|--------|------|------|
| LU | 일반 정방행렬 | O(n³) | 연립방정식 |
| QR | 일반 행렬 | O(n²m) | 최소제곱 |
| SVD | 일반 행렬 | O(min(nm², n²m)) | 차원축소, 압축 |
| 대각화 | n 개 독립 고유벡터 | O(n³) | 선형시스템 |
| Cholesky | 대칭 양정치 | O(n³/3) | 최적화, 베이지안 |

---

## 📝 추가 학습 자료

- **핵심:** 4 장 전체 Jupyter 노트북 (chapter4.ipynb)
- **심화:** SVD 기반 PCA, 이미지 처리 실험

---

## 🎯 다음 챕터

[챕터 5: 선형대수 심화](../chapter05/README.md)

*최종 수정일: 2026 년 3 월*
