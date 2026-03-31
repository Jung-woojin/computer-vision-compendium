# 챕터 5: 선형대수 심화 및 컴퓨터비전 응용

## 📚 개요

이 챕터에서는 선형대수의 심화 개념과 컴퓨터비전 분야에서의 실제 응용을 다룹니다.

---

## 📖 주요 내용

### 1. 가우시안 필터와 그라디언트

#### 1.1 2D 가우시안 함수
```python
G(x, y) = (1/(2πσ²)) · exp(-(x²+y²)/(2σ²))
```

#### 1.2 그라디언트
```python
∂G/∂x = -(x/σ²) · G(x,y)
∂G/∂y = -(y/σ²) · G(x,y)
```

**Sobel 필터:**
```
Gx = [-1, 0, 1]     Gy = [-1, -2, -1]
     [-2, 0, 2]             [ 0,   0,   0]
     [-1, 0, 1]             [ 1,   2,   1]
```

---

### 2. 라플라시안 (Laplacian)

**2 차 미분:**
```python
∇²G = ∂²G/∂x² + ∂²G/∂y²
```

**LoG (Laplacian of Gaussian):**
```python
LoG = -((x²+y²-2σ²)/(πσ⁸)) · exp(-(x²+y²)/(2σ²))
```

**응용:**
- 엣지 검출
- blob detection
- Marr-Hildreth 엣지 검출

---

### 3. 가버 필터 (Gabor Filter)

**정의:**
```python
G(x, y; λ, θ, σ, γ) = exp(-π²(x'²+γ²y'²)/σ²) · cos(2πx'/λ)
```
여기서:
- **x'** = x·cosθ + y·sinθ
- **y'** = -x·sinθ + y·cosθ
- **λ**: 파장
- **θ**: 방향
- **σ**: 표준편차
- **γ**: 횡방향 비율

**응용:**
- 텍스처 분석
- 엣지 검출 (방향성)
- 얼굴 인식

---

### 4. Receptive Field 분석

**Effective Receptive Field:**
- CNN 의 각 레이어가 감지하는 영역
- 깊어질수록 증가
- Dilated convolution 으로 제어 가능

**수식:**
```
ERF_k = ERF_{k-1} + (kernel_size - 1) · ∏_{i=1}^{k-1} stride_i
```

---

## 🔧 실습 가이드

### 실習 1: 가버 필터 구현
```python
import numpy as np
import cv2

def gabor_kernel(size, sigma, theta, lambd, gamma):
    x, y = np.meshgrid(np.arange(-size//2, size//2),
                       np.arange(-size//2, size//2))
    x_prime = x * np.cos(theta) + y * np.sin(theta)
    y_prime = -x * np.sin(theta) + y * np.cos(theta)
    
    kernel = np.exp(-np.pi**2 * (x_prime**2 + gamma**2 * y_prime**2) / sigma**2)
    kernel *= np.cos(2 * np.pi * x_prime / lambd)
    return kernel / kernel.sum()

# 예제 생성
kernel = gabor_kernel(size=31, sigma=5, theta=np.pi/4, lambd=10, gamma=0.5)
```

### 실습 2: Receptive Field 계산
```python
def calculate_erf(layers):
    """
    layers: [(kernel_size, stride), ...]
    """
    erf = 1
    dilation = 1
    
    for k, s in layers:
        erf += (k - 1) * dilation
        dilation *= s
    
    return erf

# 예시: ResNet 구조
layers = [(7, 1), (3, 2), (3, 1), (3, 1), (3, 1)]
erf = calculate_erf(layers)
print(f"Effective Receptive Field: {erf}")
```

---

## 📊 필터 비교

| 필터 | 용도 | 방향성 | 노이즈 강인성 |
|------|-----|--------|------|
| Sobel | 엣지 검출 | O(90°) | 높음 |
| Prewitt | 엣지 검출 | O(90°) | 보통 |
| Laplacian | 엣지/로브 | 360° | 낮음 |
| Gabor | 텍스처/방향 | 360° (설정에 따라) | 중간 |

---

## 📝 추가 학습 자료

- **핵심:** 5 장 전체 Jupyter 노트북 (chatper5.ipynb - 오타: 'chapter5')
- **시각화:** 가버 필터 공식 이미지 포함 (Fig_Gabor_*.svg)
- **데이터:** data/ 디렉토리 확인

---

## 🎯 다음 챕터

[챕터 6: 심화 주제](../chapter06/README.md)

*최종 수정일: 2026 년 3 월*
