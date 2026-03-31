# Metric and Representation Geometry for CV

## 1. 목적
임베딩 품질을 "감"이 아니라 선형대수 지표(거리, 각도, 공분산, 유효 차원)로 설계하고 평가하기 위한 문서.

## 2. 기본 용어
- Norm: 벡터 크기 (`||x||`)
- Distance: 두 벡터 간 거리 (`||x-y||`, Mahalanobis 등)
- Similarity: 유사도 (`x^T y`, cosine)
- Covariance: 표현 분포의 분산 구조
- Whitening: 공분산을 단위행렬에 가깝게 정규화

## 3. 연구자 관점
- metric choice는 task의 불변량(invariance)을 암묵적으로 정의한다.
- representation이 좋은지 보려면 "분리도 + 다양성"을 동시에 봐야 한다.
- collapse는 단순 성능 하락이 아니라 rank/eigen-spectrum 이상신호다.

## 4. 엔지니어 관점
- normalize 타이밍(pre-projection/post-projection)에 따라 결과가 크게 달라진다.
- cosine 계열 손실은 스케일 민감도가 낮지만 margin 튜닝이 중요하다.
- retrieval 서비스에서는 feature post-processing(PCA/whitening)으로 latency와 recall을 동시 개선할 수 있다.

## 5. 핵심 개념별 실전 해석

### 5.1 L2 vs Cosine vs Mahalanobis
- L2: 절대 스케일 포함
- Cosine: 방향 중심, 크기 영향 제거
- Mahalanobis: 축별 분산/상관을 반영한 거리

CV 예시:
- face verification: cosine margin 손실이 robust
- re-ID: camera/domain shift가 크면 Mahalanobis 근사가 유리할 때가 있음

논문 등장:
- ArcFace/CosFace/Triplet/ProxyAnchor
- class-conditional covariance metric

### 5.2 Orthogonality and Decorrelation
- feature 간 상관이 높으면 redundancy 증가
- decorrelation 제약은 representation diversity를 확보

CV 예시:
- self-supervised pretraining에서 projection head covariance regularization

논문 등장:
- redundancy reduction, whitening-based SSL objectives

### 5.3 PCA / Whitening
- PCA: 분산 큰 축 유지
- Whitening: 축간 상관 제거 + 스케일 정렬

CV 예시:
- 이미지 검색에서 임베딩 압축 및 ANN 인덱스 효율화

논문 등장:
- feature post-processing으로 retrieval score 향상

## 6. 진단 지표
- class-wise centroid distance
- intra/inter class variance ratio
- covariance eigenvalue decay
- effective rank
- alignment-uniformity 지표

## 7. 실험 아이디어

```python
import torch

def covariance(x):
    x = x - x.mean(0, keepdim=True)
    return x.T @ x / (x.shape[0] - 1)

feat = torch.randn(2048, 256)
C = covariance(feat)
eig = torch.linalg.eigvalsh(C)
print("top eigenvalues:", eig[-5:])
print("condition number:", float(eig.max() / (eig.min().clamp_min(1e-12))))

# cosine vs L2 nearest neighbor overlap
q = torch.randn(100, 256)
db = torch.randn(2000, 256)
q_n = torch.nn.functional.normalize(q, dim=-1)
db_n = torch.nn.functional.normalize(db, dim=-1)

cos_idx = (q_n @ db_n.T).topk(10, dim=-1).indices
l2_idx = (-(q[:, None, :] - db[None, :, :]).pow(2).sum(-1)).topk(10, dim=-1).indices
overlap = (cos_idx == l2_idx).float().mean()
print("top-10 overlap:", float(overlap))
```

## 8. 논문 읽기 체크리스트
1. 손실이 강제하는 geometry가 task와 맞는가?
2. normalize/margin/temperature 설정의 역할이 분리 설명되는가?
3. 분리도 향상이 다양성 손실(collapse)로 이어지지 않았는가?
4. downstream 평가가 retrieval/verification/segmentation 중 무엇에 최적화되어 있는가?

## 9. 운영 관점 팁
- 온라인 시스템에서는 metric change가 인덱스 재구성 비용을 수반한다.
- embedding dimension은 모델 성능뿐 아니라 저장소/검색 latency와 함께 설계한다.
- 학습/배포 metric 불일치(학습 cosine, 배포 L2)는 성능 저하의 흔한 원인.

## 10. 연결 문서
- [fundamentals](./fundamentals.md)
- [eigens-and-svd](./eigens-and-svd.md)
- [optimization-geometry](./optimization-geometry.md)
