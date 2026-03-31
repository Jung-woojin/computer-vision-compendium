# CNN 알고리즘 🧠

합성곱 신경망 (CNN) 의 핵심 알고리즘과 구현 디테일

## 📚 목차

- [CNN 기본 구조](#cnn-기본-구조)
- [주요 알고리즘](#-주요-알고리즘)
- [백프로파게이션](#-백프로파게이션)
- [연산 비교](#-연산-비교)
- [실전 구현](#-실전-구현)

## CNN 기본 구조

```
Input → [Conv] → [Activation] → [Pooling] → ... → [FC] → Output
            ↓                    ↓              ↓
        필터 연산          비선형성        차원 축소    최종 분류
```

## 🧮 주요 알고리즘

### 1. 합성곱 연산 (Convolution)

**기본 원리**:
- 필터 (커널) 와 입력의 지역적 상관관계 계산
- 공간적 특징 추출
- 가중치 공유 (parameter sharing)

**연산 디테일**:
```python
# Pseudo code
output[h, w, c] = sum(
    input[h:h+k, w:w+k, :] * kernel[0:k, 0:k, :]
) + bias[c]
```

**변형**:
- **Standard Conv**: 기본 컨벌루션
- **Depthwise Conv**: 채널별 독립 연산
- **Separable Conv**: Depthwise + Pointwise
- **Dilated/Atrous Conv**: 수용 영역 확대
- **Grouped Conv**: 그룹별 연산

### 2. 풀링 연산 (Pooling)

**유형**:
- **Max Pooling**: 지역 최대값
- **Average Pooling**: 지역 평균값
- **Global Pooling**: 전체 영역 집계

**역할**:
- 차원 축소 (계산량 감소)
- 위치 불변성 증가
- 특징 강화 (max pooling)

### 3. 정규화 (Normalization)

**유형 비교**:

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Batch Norm** | 배치 내 정규화 | 학습 안정화 | 배치 크기에 의존 |
| **Layer Norm** | 층 내 정규화 | RNN 에 좋음 | 안정성 낮음 |
| **Instance Norm** | 이미지별 정규화 | 스타일 전이 | 일반화 낮음 |
| **Group Norm** | 그룹별 정규화 | 작은 배치에도 좋음 | BN 보다 느림 |

### 4. 활성화 함수 (Activation)

**주요 활성화 함수 비교**:

| 함수 | 수식 | 장단점 |
|------|------|--------|
| **ReLU** | `max(0, x)` | 단순, 빠름, dead neurons |
| **Leaky ReLU** | `max(αx, x)` | gradient flow 개선 |
| **ELU** | `x if x>0 else α(e^x-1)` | 부드러운 전환 |
| **Swish** | `x * sigmoid(x)` | 자동 최적화 가능 |
| **GELU** | `x * Φ(x)` | Transformer 표준 |

## 🔄 백프로파게이션

### 기본 원리

**연쇄법칙 (Chain Rule)**을 활용한 역전파:

```
Loss → Output → Hidden → Input → Weights Update
```

### 컨벌루션 레이어 역전파

**가중치 그래디언트**:
```
∂L/∂W = Convolution(∂L/∂Output, Input)
```

**입력 그래디언트**:
```
∂L/∂Input = ConvTranspose(∂L/∂Output, W)
```

### 풀링 레이어 역전파

**Max Pooling**:
- forward 시 최대값 인덱스 저장
- backward 시 해당 위치로만 gradient 전달

**Average Pooling**:
- gradient 를 균등 분배

## 📊 연산 비교

### 계산 효율성 (FLOPs)

| 연산 | FLOPs/패스 | 메모리 | 속도 |
|------|-----------|--------|------|
| **Standard Conv** | 2×H×W×K²×C_in×C_out | 높음 | 표준 |
| **Depthwise** | H×W×K²×C_in | 낮음 | 빠름 |
| **Pointwise** | H×W×C_in×C_out | 중간 | 빠름 |
| **Separable** | H×W×(K²×C_in + C_in×C_out) | 매우 낮음 | 매우 빠름 |

### 공간 효율성 (Parameter Count)

| 레이어 | 파라미터 수 |
|--------|-------------|
| Conv(3×3, 64→128) | 3×3×64×128 + 128 = 73,856 |
| Depthwise(3×3, 128) | 3×3×128 = 1,152 |
| Pointwise(1×1, 128→128) | 128×128 + 128 = 16,512 |

## 🛠 실전 구현

### PyTorch 구현 예제

```python
import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### 최적화 팁

1. **Gradient Clipping**: Exploding gradient 방지
2. **Learning Rate Scheduling**: Warmup, Cosine annealing
3. **Data Augmentation**: Random crop, flip, color jitter
4. **Label Smoothing**: Overfitting 감소
5. **Mixup/Cutout**: Regularization

## 📈 학습曲线 분석

### 정상 학습 패턴
- Loss: 지수 감소 후 수렴
- Accuracy: 증가 후 안정화
- Validation: Training 과 유사한 패턴

###常见问题
- **Loss 가 증가**: Learning rate 너무 높음
- **Underfitting**: 모델 용량 부족, 학습 시간 부족
- **Overfitting**: Regularization 추가 필요

---
*마지막 업데이트: 2026-03-30*
