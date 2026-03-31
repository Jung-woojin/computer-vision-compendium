# Convolution as Linear Operator

## 1. 목적
CNN의 convolution을 "슬라이딩 필터" 수준이 아니라 선형연산자 관점으로 이해해,  
아키텍처 설계와 효율화 선택을 수학적으로 설명할 수 있게 한다.

## 2. 핵심 질문
- 왜 convolution은 translation equivariance를 갖는가?
- 1x1 conv, depthwise conv, group conv는 선형대수적으로 무엇이 다른가?
- convolution을 factorization하면 언제 이득이고 언제 손해인가?

## 3. 핵심 개념

### 3.1 Convolution = 공유 가중치 선형연산
- 2D convolution은 입력을 큰 벡터로 펼치면 거대한 행렬곱으로 표현 가능하다.
- 이 행렬은 보통 Toeplitz(또는 block-Toeplitz) 구조를 가진다.
- 구조적 제약 덕분에 파라미터 수와 계산량이 dense linear layer보다 작다.

### 3.2 채널 혼합과 공간 혼합 분해
- `1x1 conv`: 채널 차원 선형변환
- `kxk conv`: 공간 이웃 정보 혼합
- `depthwise + pointwise`: 공간/채널 연산을 분리한 factorized operator

### 3.3 Frequency 관점
- convolution은 주파수 영역에서 점곱에 대응한다.
- 고주파/저주파 성분을 어떤 필터가 강조/억제하는지 스펙트럼으로 해석 가능.

## 4. 연구자 관점
- convolution 커널 설계는 inductive bias 설계다.
- 작은 커널을 깊게 쌓는 방식과 큰 receptive field를 직접 쓰는 방식은 서로 다른 연산자 근사.
- layer별 operator spectrum을 보면 정보 손실 지점을 진단할 수 있다.

## 5. 엔지니어 관점
- 연산 효율은 FLOPs만이 아니라 memory access 패턴에 좌우된다.
- depthwise는 FLOPs가 낮아도 실제 하드웨어에서 항상 빠르지 않다.
- kernel size, stride, dilation 변경은 receptive field와 aliasing을 동시에 바꾼다.

## 6. CV 예시
- Edge detector: Sobel 필터는 방향 미분 연산자의 선형 근사.
- Super-resolution: 고주파 복원 실패는 특정 주파수 축이 연산자 null space에 가깝기 때문일 수 있다.
- Segmentation: dilated conv는 해상도 유지와 문맥 수용영역 확보를 동시에 노린다.

## 7. 딥러닝 예시
- ResNet bottleneck: `1x1 -> 3x3 -> 1x1`는 채널 축소/공간 처리/채널 복원으로 분해된 선형 블록.
- MobileNet 계열: depthwise separable conv로 연산자 rank를 제한해 효율 확보.
- ConvNeXt: 현대적 학습 설정에서 conv operator를 transformer와 경쟁 가능한 수준으로 재설계.

## 8. 논문에서 보이는 형태
- "depthwise separable convolution"
- "group convolution"
- "factorized convolution"
- "dilated convolution"
- "dynamic convolution / conditional kernel"

## 9. 구현 체크리스트
1. shape 확인: `B,C,H,W` 기준으로 channel/order 일관성 유지
2. padding 규칙 명시: same/valid 혼용 금지
3. stride와 downsampling 위치를 아키텍처 차원에서 통일
4. depthwise/group 사용 시 실제 latency를 장비에서 측정
5. kernel fusion 가능성(배포 엔진) 확인
6. convolution 뒤 normalization/activation 순서 고정

## 10. 미니 실험 아이디어
```python
import torch
import torch.nn as nn
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(8, 128, 112, 112, device=device)

ops = {
    "conv3x3": nn.Conv2d(128, 128, 3, padding=1).to(device),
    "dw+pw": nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1, groups=128),
        nn.Conv2d(128, 128, 1)
    ).to(device),
}

for name, op in ops.items():
    for _ in range(10):
        _ = op(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        _ = op(x)
    if device == "cuda":
        torch.cuda.synchronize()
    print(name, "sec:", time.time() - t0)
```

## 11. 연결 문서
- [허브 문서](./linear-algebra-for-cv.md)
- [저랭크 효율화](./low-rank-efficient-models.md)
- [최적화 곡률](./optimization-curvature.md)
