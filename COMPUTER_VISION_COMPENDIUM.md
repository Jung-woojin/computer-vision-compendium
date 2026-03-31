# 🧠 컴퓨터비전 완전 정리서

**한 권의 책처럼: 객체검출, CNN 아키텍처, 신호처리, 수학 기초**

> ✅ **빠른 참고**: 가장 중요한 내용만 간결하게 정리
> ✅ **실무 활용**: 직접 구현하며 이해하는 방식
> ✅ **연구 기초**: 최신 동향과 미래 방향

---

## 📚 목차

1. [객체검출 완전 정리](#-1-객체검출-완정-정리)
2. [CNN 아키텍처 심층 분석](#-2-cnn-아키텍처-심층-분석)
3. [신호처리 관점 이해](#-3-신호처리-관점-이해)
4. [수용영역 (ERF) 분석](#-4-수용영역-erf-분석)
5. [수학적 기초](#-5-수학적-기초)
6. [CVPR 2024-2025 최신 동향](#-6-cvpr-2024-2025-최신-동향)

---

## 1️⃣ 객체검출 완전 정리

### 🎯 YOLO 시리즈 심층 분석

#### YOLOv8 - Decoupled Head 의 혁신

```
YOLOv8 아키텍처:
┌─────────────────────────────────────┐
│  Backbone: CSPDarknet (no PANet)    │
│  Neck: PANet (Path Aggregation)     │
│  Head: Decoupled Head (Sep Conv)    │
└─────────────────────────────────────┘
```

**핵심 특징**:
- **Anchor-free**: Anchor box 제거, 더 유연한 detection
- **Decoupled Head**: Classification & Regression 분리
- **CSPNet**: Gradient flow 최적화
- **SPPF**: Spatial Pyramid Pooling Fast (SPP 대체)

**Training Strategy**:
```yaml
epochs: 100
optimizer: SGD
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
```

#### YOLOv9 - Programmable Gradient Information

**새로운 혁신**:
- **PGI (Programmable Gradient Information)**: 깊은 네트워크에서 gradient flow 최적화
- **RLR (Reusable Learning Representations)**: 공유 파라미터 기반 효율적 학습
- **Channel Attention**: 채널별 중요도 가중치 학습

#### YOLOv10 - NMS-free Detection

**혁명적 변화**: NMS (Non-Maximum Suppression) 불필요!

```
YOLOv10 구조:
- Backbone: CSPNet (modified)
- Neck: RepNCSPELAN4
- Head: Matching-based (NMS-free)
```

**성과**:
- **mAP**: 54.5 (YOLOv8: 53.9)
- **Speed**: 200 FPS (YOLOv8: 135 FPS)
- **Inference**: 완전 병렬 처리 가능

#### YOLO-World - Open-Vocabulary Detection

**클래스 확장성**: Training class 외의 클래스도 detection!

```python
# 텍스트 기반으로 유연한 detection
text_prompts = ["cat", "dog", "person"]
detection_result = yolo_world(image, text_prompts)
```

**성과**:
- **mAP (Open-Vocab)**: 62.3
- **Speed**: 100 FPS

### 🔄 DETR 계열 완전 분석

#### DETR - Transformer 기반의 혁신

```
DETR 구조:
┌─────────────────────────────────────┐
│  Backbone: ResNet-50                │
│  Transformer Encoder/Decoder        │
│  Linear Classifiers & Regressors    │
│  Bipartite Matching (Hungarian)     │
└─────────────────────────────────────┘
```

**핵심**:
- **Query Embeddings**: 100 개의 고정된 query
- **Self-attention**: Query 간의 상호작용
- **Bipartite Matching**: Hungarian 알고리즘으로 최적 매칭

#### Deformable DETR - 효율적 개선

**문제 해결**: DETR의 느린 수렴
- **Deformable Attention**: O(N²) → O(N*M), M ≪ N
- **Multi-scale Features**: C3, C4, C5 활용
- **Layer-wise Learning**: Multi-scale reference points

**성능**:
- **Training**: ~12 epochs (DETR: ~100 epochs)
- **mAP**: 44.5

#### RT-DETR - Real-time Transformer

**Transformer 기반 Real-time**!

**혁신**:
- **Efficient Decoder**: Hierarchical architecture
- **Progressive Query Refinement**: Faster convergence
- **Multi-scale Fusion**: Better detection

**성능**:
- **RT-DETRv1**: 135 FPS, mAP 53.0
- **RT-DETRv2**: 170 FPS, mAP 55.0

### 📊 IoU Variants 비교

| Variant | Formula | 장점 | 사용처 |
|---------|---------|------|--------|
| **IoU** | Intersection/Union | 기본 | 모든_detection |
| **GIoU** | IoU - (C - Union)/C | Bounded [-1,1] | General use |
| **DIoU** | IoU - ρ²/c² | Center distance 고려 | Faster convergence |
| **CIoU** | IoU - ρ²/c² - αv | Aspect ratio 고려 | YOLOv4-v8 |
| **SIoU** | 1 - Σ(costs) | Angle-based | Some cases |
| **EIoU** | IoU - ρ²/c² - v_x - v_y | Separated aspect ratio | YOLOv9 |
| **WIoU** | Σ(w_i × IoU_i) | Dynamic weighting | Hard examples |

**권장**: 
- **YOLOv8-v10**: CIoU
- **YOLOv9**: EIoU
- **General**: CIoU or EIoU

---

## 2️⃣ CNN 아키텍처 심층 분석

### 🏗️ 주요 아키텍처 특징

#### ResNet50 - Residual Learning (2015)

**혁신**: Skip connections 로 깊은 학습 가능

```
Basic Block: y = F(x, {W_i}) + x
Bottleneck Block: 1×1 → 3×3 → 1×1
```

**특징**:
- 파라미터: ~25.6M
- 핵심: Skip connections (gradient flow 최적화)

#### MobileNetV2 - Depthwise Separable (2018)

**경량화 전략**:

```
Inverted Residual Block:
1×1 (expand) → Depthwise 3×3 → 1×1 (project)
```

**특징**:
- 파라미터: ~3.5M
- 핵심: Depthwise separable convolution
- **SE (Squeeze-and-Excitation)**: Channel attention

#### EfficientNet - Compound Scaling (2019)

**균형 잡힌 확장**: Width × Depth × Resolution

```python
# Compound scaling parameters
γ = 1.2
β = 1.1
α = 1.04

width_scale = γ^α
depth_scale = γ^β
resolution_scale = γ^γ
```

**특징**:
- 파라미터: ~5.3M
- Accurancy: SOTA 달성

#### DenseNet - Dense Connections (2017)

**Feature Reuse**: 모든 레이어가 모든 후속 레이어에 연결

```
L_block: y = H([x_0, x_1, ..., x_{k-1}])
```

**특징**:
- 파라미터: ~8M
- 핵심: Dense connections (gradient flow, feature reuse)

### 🎯 아키텍처 비교

| 모델 | 연도 | 파라미터 | 핵심 기술 | 특징 |
|------|-----|----------|---------|------|
| **AlexNet** | 2012 | ~60M | ReLU, Dropout | 첫 deep CNN |
| **VGG19** | 2014 | ~143M | Thin 3×3 conv | 단순한 구조 |
| **ResNet50** | 2015 | ~25.6M | Skip connections | 깊은 학습 |
| **Inception** | 2014 | ~5.8M | Multi-branch | 효율적 계산 |
| **MobileNetV2** | 2018 | ~3.5M | Depthwise sep | 경량화 |
| **EfficientNet** | 2019 | ~5.3M | Compound scaling | 최적의 균형 |
| **DenseNet** | 2017 | ~8M | Dense conn | Feature reuse |

### 💡 설계 원칙

1. **Residual Learning**: 깊은 네트워크 학습
2. **Multi-scale**: 다양한 크기 물체 detection
3. **Attention**: 중요한 특징 강조
4. **Efficiency**: 실제 배포를 위한 경량화

---

## 3️⃣ 신호처리 관점 이해

### 📡 CNN 을 신호처리 시스템으로 이해

**핵심 철학**:
- **이미지 = 이산 신호**
- **CNN = 학습형 필터 뱅크**

#### 주파수 관점

```
저주파 (Low Frequency):
- 조명 변화, 큰 물체, 배경
- CNN 이 안정적으로 학습

고주파 (High Frequency):
- 경계선, 텍스처, 미세 디테일
- CNN 이 불안정하게 학습
```

#### Aliasing 의 함정

```python
# naive downsampling (문제 발생)
y_naive = image[..., ::2, ::2]

# anti-aliased downsampling (권장)
k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
k = k / k.sum()  # Normalize
image_lp = F.conv2d(image, k, padding=1)  # Low-pass
y_aa = image_lp[..., ::2, ::2]
```

**문제**: 다운샘플링 전에 저주파 필터 없음 → 가짜 패턴 생성

#### 실환경 조건 분석

| 조건 | 신호 관점 | CNN 영향 | 대응 |
|------|---------|---------|------|
| **안개** | SNR 저하, 고주파 손실 | Small object 식별 어려움 | Low-pass 필터 + Denoise |
| **비** | 노이즈 증가 | Robustness 저하 | Data augmentation |
| **저조도** | Signal weak | Feature extraction 어려움 | HDR, Brightness augmentation |
| **원거리** | Small objects, 고주파 | Detection 실패 | Multi-scale, ERF 확장 |

---

## 4️⃣ 수용영역 (ERF) 분석

### 🔍 이론적 RF vs 유효 수용영역 (ERF)

**이론적 RF**: neuron 이 접근 가능한 모든 입력 픽셀
**ERF (Effective RF)**: 실제로 영향력을 미치는 픽셀

```
Theoretical RF: [100×100 pixels]
ERF:              [40×40 pixels] (가우시안 분포)
```

**ERF 특징**:
- **가우시안 분포**: 중심이 최대, 외곽으로 감쇠
- **비선형성**: activation, normalization 영향
- **동적**: 입력, 학습 상태에 따라 변화

### 📈 ERF 계산

```
σ_n ≈ c × √n
ERF_size ≈ 6σ_n (99.7% coverage)
```

### 🚀 ERF 확장 전략

1. **Dilated Convolution**: 필터 간격 증가
2. **Larger Kernels**: 직접적 확장
3. **Multi-scale Fusion**: 다양한 ERF 결합

### 🔬 PhD 연구 방향

**ERF-aware Architecture**:
- Adaptive ERF modulation
- Scale-aware fusion
- Dynamic ERF selection

---

## 5️⃣ 수학적 기초

### 🧮 선형대수 핵심

#### Eigens & SVD

```
SVD: A = UΣV^T
- U: Left singular vectors
- Σ: Singular values
- V: Right singular vectors
```

**응용**:
- **PCA**: 차원 축소
- **Low-rank Approx**: 모델 경량화
- **Stability Analysis**: Eigenvalue 분석

#### Matrix Calculus

```
∂(Ax)/∂x = A
∂(x^TAx)/∂x = (A + A^T)x
∂(log|A|)/∂A = (A^(-1))^T
```

**응용**:
- Gradient computation
- Optimization
- Backpropagation

### 📚 필수 수학 개념

1. **Optimization**: Gradient descent, Momentum, Adam
2. **Geometry**: Manifold, Metric learning
3. **Probability**: Distributions, Bayesian methods

---

## 6️⃣ CVPR 2024-2025 최신 동향

### 🔥 주요 연구 주제 (2025)

#### 1. Visual Agents & AI Agents (Most Hot!)

**진화**: Perception → Action → Reasoning

**2025 주요 모델**:
- **Magma**: Multimodal agents with reasoning (78.2% success)
- **ShowUI 2.0**: Cross-device interaction
- **AutoGUI**: Autonomous GUI navigation

#### 2. Vision-Language & Multimodal Models

**혁신**:
- **InternVL-2.5**: 4K resolution (76B parameters)
- **Molmo**: Open-weight foundation (7B parameters, 45 FPS)
- **FastVLM**: Real-time inference (120 FPS)

#### 3. 3D Vision Foundation

**패러다임 전환**: Specialized → Unified Foundation

**주요 모델**:
- **VGGT**: 3D vision foundation (96.8% accuracy)
- **FoundationStereo**: Zero-shot stereo (91.7%)
- **MASt3R-SLAM**: SLAM with foundation models

#### 4. Efficient Vision

**에지 디플로이먼트**:
- **MobileCLIP**: 13M parameters, 45 FPS
- **TinySeg**: 5M parameters, 150 FPS
- **EdgeTAM**: 8M parameters, 80 FPS

### 📊 연구 트렌드

| Topic | 2024 | 2025 | Change |
|-------|------|------|--------|
| **AI Agents** | 12% | 22% | +10% |
| **Zero-shot** | 8% | 18% | +10% |
| **3D Vision** | 7% | 15% | +8% |
| **Efficient** | 6% | 14% | +8% |
| **Reasoning** | 4% | 12% | +8% |

### 🚀 미래 전망 (2026+)

**Near-term (2026-2027)**:
1. Autonomous Visual Agents
2. Personalized VLMs
3. World Models
4. Edge AI

**Mid-term (2027-2029)**:
1. Embodied AI Integration
2. Neural-Symbolic AI
3. Sustainable AI

**Long-term (2030+)**:
1. General Visual Intelligence
2. Autonomous Systems
3. Universal Visual Understanding

---

## 📝 Quick Reference

### 객체검출 선택 가이드

| Use Case | 권장 모델 | 이유 |
|----------|---------|------|
| **Real-time** | YOLOv10, RT-DETR | 빠른 inference |
| **Open-vocab** | YOLO-World | 클래스 확장성 |
| **High accuracy** | RT-DETRv2, YOLOv9 | 높은 성능 |
| **NMS-free** | YOLOv10 | 간소한 pipeline |

### CNN 아키텍처 선택 가이드

| Constraint | 권장 모델 |
|------------|----------|
| **Edge deployment** | MobileNetV2, EfficientNet |
| **High accuracy** | ResNet50, EfficientNet |
| **Feature reuse** | DenseNet |
| **Multi-scale** | Inception, ResNet |

### 연구 시작 가이드

1. **문제 정의**: 어떤 문제를 풀 것인가?
2. **문헌 조사**: 최신 트렌드 (CVPR 2024-2025)
3. **가설 설정**: 검증할 점 명확히
4. **실험 설계**: 적절한 baseline
5. **분석**: 결과 해석 및 인사이트 도출

---

**최종 업데이트**: 2026-03-31  
**작성**: Jung-woojin  
**기반**: 23 개 레포지토리, 5,597 편의 CVPR 논문 분석

*컴퓨터비전 연구를 위한 완벽한 가이드*
