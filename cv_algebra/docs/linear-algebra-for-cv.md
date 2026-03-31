# 1. 문서 제목
컴퓨터비전 연구를 위한 선형대수 연결 지도: CNN, Attention, Embedding, PCA, Metric, Optimization까지

# 2. 문제의식: 왜 비전 연구자는 선형대수를 다시 공부해야 하는가
비전 모델은 "큰 네트워크"처럼 보이지만, 내부적으로는 거의 전부가 행렬/벡터 연산이다.  
성능이 안 나올 때도 실제 원인은 다음처럼 선형대수 언어로 설명되는 경우가 많다.

- 표현이 붕괴한다: feature covariance 스펙트럼이 한두 축으로 몰림
- 학습이 불안정하다: Hessian 곡률이 큰 축에서 업데이트가 폭주
- 검색 성능이 낮다: embedding 거리 설계(L2/cosine/Mahalanobis)가 task와 불일치
- 모델은 큰데 효율이 낮다: 저랭크 구조를 활용하지 못함
- attention이 잘 안 된다: projection 공간 해석 없이 하이퍼파라미터만 조절

즉, 선형대수는 "수학 기초"가 아니라, 연구 가설을 세우고 실패 원인을 분해하는 도구다.

# 3. 핵심 개념 10개
1. Convolution as Linear Operator
2. Feature Embedding Space
3. Similarity and Inner Product
4. Projection and Attention Intuition
5. Covariance in Representation Analysis
6. Eigenvalues / SVD / PCA
7. Metric Learning Distance Design
8. Hessian and Optimization Curvature
9. Low-rank Approximations in Efficient Vision Models
10. Matrix Factorization Intuition in Representation Learning

# 4. 개념별

## 4.1 Convolution as Linear Operator
정의:
- convolution은 입력 텐서에 대해 지역 선형 연산자를 적용하는 과정이다.
- 2D conv는 엄밀히 쓰면 Toeplitz(혹은 doubly block-Toeplitz) 구조의 큰 행렬 곱으로 해석 가능하다.

직관:
- "슬라이딩 필터"는 결국 같은 선형변환을 위치마다 공유하는 연산이다.
- weight sharing은 파라미터 효율성 + translation equivariance를 만든다.

비전 예시:
- edge/texture 검출은 특정 공간 주파수 성분에 대한 선형 필터링으로 이해 가능하다.

딥러닝 예시:
- 1x1 convolution은 채널 공간의 basis 변환(선형 mixing)이다.
- depthwise + pointwise는 공간/채널 선형연산 분리(factorized operator)다.

논문에서 보이는 형태:
- "linear projection", "token/channel mixing", "depthwise separable conv"
- 효율 모델에서 conv factorization 또는 FFT-domain operator로 표현

## 4.2 Feature Embedding Space
정의:
- 네트워크 출력 feature를 벡터 공간의 점으로 보고, 클래스/의미 구조를 기하학으로 해석하는 관점.

직관:
- 좋은 임베딩은 같은 의미는 가까이, 다른 의미는 멀리 배치한다.
- 단순 차원 수가 아니라 유효 차원(effective dimension)과 분리도(separability)가 중요하다.

비전 예시:
- 얼굴 인식 임베딩에서 사람별 군집이 각도 기준으로 분리되는지 관찰.

딥러닝 예시:
- contrastive pretraining 후 선형 분류기 성능(linear probe)로 embedding 품질 점검.

논문에서 보이는 형태:
- "embedding geometry", "latent space", "linear separability", "representation collapse"

## 4.3 Similarity and Inner Product
정의:
- 유사도는 내적 또는 거리로 정의되며, 내적은 각도와 크기 정보를 동시에 반영한다.

직관:
- cosine similarity는 방향 중심 비교, L2는 절대 위치/스케일까지 반영.
- normalize 여부가 유사도 의미를 바꾼다.

비전 예시:
- image retrieval에서 cosine top-k와 L2 top-k 결과가 다르게 나올 수 있다.

딥러닝 예시:
- CLIP류 모델의 image-text alignment는 정규화된 내적(logit scale 포함)을 사용한다.

논문에서 보이는 형태:
- "normalized dot product", "cosine similarity", "temperature-scaled logits"

## 4.4 Projection and Attention Intuition
정의:
- projection은 고차원 벡터를 특정 부분공간으로 사상하는 연산.
- attention은 Q/K/V projection 이후, 유사도 기반 가중합으로 정보를 재구성하는 연산.

직관:
- attention은 "어떤 기준축(query)에서 볼 때, 어떤 정보(key/value)를 얼마나 반영할지"를 정하는 동적 projection.
- QK^T는 본질적으로 Gram-like similarity 행렬.

비전 예시:
- ViT에서 object token이 주변 patch에서 관련 정보만 선택적으로 집계.

딥러닝 예시:
- cross-attention은 서로 다른 modality 공간 사이의 projection alignment 문제로 해석 가능.

논문에서 보이는 형태:
- `Q=XW_Q, K=XW_K, V=XW_V`, `softmax(QK^T/sqrt(d))V`
- "attention map", "token affinity", "projection head"

## 4.5 Covariance in Representation Analysis
정의:
- covariance는 feature 차원 간 공분산 구조를 나타내는 행렬.

직관:
- 대각 성분은 분산, 비대각 성분은 차원 간 중복/상관.
- 한두 개 고유값만 크면 정보가 특정 축에 몰렸을 가능성.

비전 예시:
- self-supervised 표현에서 covariance spectrum을 보면 collapse 조기 탐지 가능.

딥러닝 예시:
- batch feature whitening/decorrelation은 중복 표현을 줄이고 학습 안정성을 높일 수 있다.

논문에서 보이는 형태:
- "covariance regularization", "decorrelation objective", "whitening transform"

## 4.6 Eigenvalues / SVD / PCA
정의:
- eigen decomposition: 선형변환의 고정축과 스케일(고유벡터/고유값) 분해.
- SVD: 임의 행렬을 회전-스케일-회전으로 분해 (`A=UΣV^T`).
- PCA: 데이터 공분산의 주요 고유축을 이용한 차원 축소.

직관:
- 큰 고유값/특이값 축이 "정보가 많이 흐르는 방향".
- low-rank 근사는 작은 특이값 축을 버려 압축.

비전 예시:
- feature 시각화에서 PCA 2D/3D 투영으로 클래스 분리 패턴 점검.

딥러닝 예시:
- weight matrix SVD로 rank-k 근사 후 속도/메모리 개선.

논문에서 보이는 형태:
- "principal components", "spectral analysis", "truncated SVD", "rank constraint"

## 4.7 Metric Learning Distance Design
정의:
- 학습 목표를 "거리 함수 설계"로 두는 접근(contrastive, triplet, proxy 등).

직관:
- 어떤 거리를 쓰느냐가 모델이 보존할 invariance를 결정한다.
- 거리 함수 = 모델의 판단 규칙.

비전 예시:
- person re-ID에서 camera/domain shift 하에서도 같은 사람 임베딩 거리 최소화.

딥러닝 예시:
- ArcFace는 정규화된 각도 공간에서 margin을 둬 클래스 경계를 분명하게 만든다.

논문에서 보이는 형태:
- `d(x_i, x_j)`, margin-based loss, proxy-anchor, circle loss

## 4.8 Hessian and Optimization Curvature
정의:
- Hessian은 손실 함수의 2차 미분(곡률) 정보.

직관:
- 곡률이 큰 방향에서는 작은 step도 손실을 크게 바꾼다.
- 평평한 방향은 업데이트가 느리거나 모호할 수 있다.

비전 예시:
- 대규모 비전 사전학습에서 warmup 부재 시 초반 발산은 곡률 불일치로 해석 가능.

딥러닝 예시:
- SAM류 방법은 sharp한 영역을 피하도록 업데이트해 일반화 성능 향상을 노린다.

논문에서 보이는 형태:
- "curvature", "sharpness", "Hessian trace/top eigenvalue", "second-order approximation"

## 4.9 Low-rank Approximations in Efficient Vision Models
정의:
- 행렬/텐서를 낮은 rank 구조로 근사해 연산량과 파라미터를 줄이는 방법.

직관:
- 중요한 정보 축만 남기고 잔차는 버리거나 별도 보정.
- 정확도-효율 trade-off를 선형대수 관점으로 조절.

비전 예시:
- 모바일 비전 모델에서 conv/attention weight를 저랭크로 분해해 latency 절감.

딥러닝 예시:
- LoRA/adapter 계열은 업데이트를 low-rank 경로로 제한해 효율적인 미세조정 수행.

논문에서 보이는 형태:
- "rank-k decomposition", "low-rank adaptation", "factorized projection"

## 4.10 Matrix Factorization Intuition in Representation Learning
정의:
- 복잡한 표현/관계를 더 작은 요인 행렬들의 곱으로 분해하는 관점.

직관:
- representation을 잠재 요인들의 선형 조합으로 보면 해석과 제어가 쉬워진다.
- 분해는 압축뿐 아니라 "무엇이 정보 요인인가"를 드러낸다.

비전 예시:
- 배경/조명/객체 성분을 분리하는 저차원 요인 모델로 데이터 구조 파악.

딥러닝 예시:
- dictionary-like basis 학습, NMF류 접근, factorized token/channel decomposition.

논문에서 보이는 형태:
- "factorized latent space", "dictionary learning", "disentangled factors", "CP/Tucker"

# 5. 구현 체크포인트
- 텐서 shape를 수식 단위로 추적한다 (`B, N, D`, `C, H, W` 명시).
- normalize 위치를 고정하고 실험한다 (projection 전/후, loss 전/후).
- similarity 정의를 코드에서 일관되게 유지한다 (cosine인지 dot인지 혼용 금지).
- covariance/eigen-spectrum 로그를 주기적으로 저장한다.
- attention에서 `1/sqrt(d)` 스케일링 누락 여부 확인.
- 저랭크 근사 전후의 정확도/latency/메모리 3축을 같이 측정한다.
- mixed precision 환경에서 SVD/eigen 연산 안정성(FP32 cast) 확인.
- metric learning에서는 batch 구성(positive/negative sampling)을 먼저 검증한다.
- optimizer 이슈를 LR만으로 보지 말고 curvature proxy(HVP, sharpness)도 확인한다.
- 실험 리포트에 "수식 가정 vs 구현 가정" 차이를 명시한다.

# 6. 자주 나오는 오해
- "선형대수는 모델 설계와 무관하다": 실제로 거의 모든 설계 선택이 선형연산 구조와 연결된다.
- "차원만 키우면 표현력이 오른다": 유효차원과 노이즈 축을 구분해야 한다.
- "cosine이 항상 L2보다 낫다": task와 데이터 분포에 따라 달라진다.
- "attention은 완전히 비선형이다": 핵심 계산은 projection + 유사도 + 가중합의 선형대수 구조다.
- "PCA는 옛날 기법이다": feature 분석/압축/전처리에서 여전히 강력한 baseline.
- "저랭크 = 무조건 성능 저하": 적절한 rank와 fine-tuning 전략이면 효율 이득이 크다.
- "optimizer를 바꾸면 해결된다": 곡률/스케일 문제를 먼저 진단해야 한다.

# 7. 관련 후속 키워드
- spectral bias
- neural tangent kernel (NTK)
- random matrix theory in deep learning
- whitening / ZCA
- Fisher information matrix
- natural gradient / K-FAC
- subspace training
- manifold hypothesis
- isotropy in embedding space
- robust optimization and sharpness

# 8. GitHub 링크 구조 제안
권장 디렉터리:

```text
cv_algebra/
  README.md
  docs/
    linear-algebra-for-cv.md
    convolution-as-operator.md
    embedding-and-metric-geometry.md
    eigens-svd-pca.md
    optimization-curvature.md
    low-rank-efficient-models.md
  experiments/
    svd-rank-sweep.py
    covariance-collapse-check.py
    attention-projection-toy.py
  references/
    paper-notes-template.md
```

링크 운영 원칙:
- 허브 문서(`linear-algebra-for-cv.md`)에서 각 세부 문서로 단방향 링크
- 각 세부 문서에는 "실험 코드"와 "대표 논문" 섹션을 고정으로 둠
- `experiments/`는 재현 가능한 최소 코드만 유지하고 결과는 별도 로그로 관리
