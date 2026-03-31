# Embedding and Metric Geometry

## 1. 목적
feature embedding 공간을 거리/각도/분산 구조로 해석하고,  
metric learning 손실 설계를 실험 가능한 형태로 정리한다.

## 2. 핵심 질문
- cosine과 L2는 어떤 데이터에서 다르게 동작하는가?
- embedding collapse를 어떻게 조기에 감지할 수 있는가?
- retrieval/re-ID/face에서 metric 설계를 어떻게 분리해야 하는가?

## 3. 기본 좌표계
- 벡터 크기: `||x||`
- 방향 유사도: `cos(x,y)=x^Ty/(||x||||y||)`
- 거리: `||x-y||_2`, Mahalanobis `sqrt((x-y)^T M (x-y))`
- 분산 구조: covariance `C`

## 4. 연구자 관점
- metric은 모델이 보존할 invariance를 정의한다.
- 좋은 임베딩은 분리도(inter-class)와 다양성(intra-feature diversity)을 동시에 만족한다.
- representation collapse는 정확도 저하 이전에 스펙트럼 이상으로 먼저 나타난다.

## 5. 엔지니어 관점
- normalize 시점 하나만 바꿔도 실험 결과가 크게 달라진다.
- 배치 샘플링이 metric learning 성능의 절반 이상을 결정하는 경우가 많다.
- 학습 metric과 서비스 metric이 다르면 온라인 품질이 급락할 수 있다.

## 6. CV 예시
- Face verification: cosine margin 기반 경계가 identity 분리를 명확히 함.
- Re-ID: hard negative mining이 과하면 불안정해지고 일반화가 떨어질 수 있음.
- Image retrieval: PCA/whitening 후 ANN 검색 정확도와 latency 균형 개선 가능.

## 7. 딥러닝 예시
- Contrastive loss: positive는 가깝게, negative는 멀게.
- Triplet loss: `(anchor, positive, negative)` 상대 거리 제약.
- ArcFace/CosFace: normalized hypersphere 위 각도 margin 학습.

## 8. 논문에서 보이는 형태
- "alignment vs uniformity"
- "temperature scaling"
- "proxy-based metric learning"
- "hard negative mining"
- "feature normalization"

## 9. 구현 체크리스트
1. similarity 정의를 코드 전역에서 통일
2. normalize 연산의 위치와 epsilon 고정
3. batch 내 positive/negative 구성 비율 로깅
4. class imbalance가 metric에 미치는 영향 점검
5. covariance/effective-rank를 주기적으로 계산
6. train/val에서 동일한 retrieval metric 사용

## 10. 미니 실험 아이디어
```python
import torch
import torch.nn.functional as F

q = torch.randn(128, 256)
db = torch.randn(4096, 256)

# cosine retrieval
q_n = F.normalize(q, dim=-1)
db_n = F.normalize(db, dim=-1)
cos_topk = (q_n @ db_n.T).topk(10, dim=-1).indices

# l2 retrieval
l2_scores = -((q[:, None, :] - db[None, :, :]) ** 2).sum(-1)
l2_topk = l2_scores.topk(10, dim=-1).indices

overlap = (cos_topk == l2_topk).float().mean()
print("top-10 overlap:", float(overlap))
```

## 11. 자주 나오는 오해
- "cosine이 무조건 좋다": 절대 스케일 정보가 중요한 task에서는 L2가 더 맞을 수 있다.
- "embedding 차원 크게 하면 해결": 차원 증가가 곧 정보 증가를 보장하지 않는다.
- "hard mining은 많을수록 좋다": 너무 공격적이면 noise/outlier에 과적합 가능.

## 12. 연결 문서
- [허브 문서](./linear-algebra-for-cv.md)
- [고유값/SVD/PCA](./eigens-svd-pca.md)
- [최적화 곡률](./optimization-curvature.md)
