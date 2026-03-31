# Eigens, SVD, PCA in Vision

## 1. 목적
고유값/특이값/PCA를 "수학 정의"가 아니라  
representation 진단, 모델 압축, 논문 재현에 직접 연결하는 가이드.

## 2. 핵심 질문
- 스펙트럼을 보면 어떤 실패를 미리 감지할 수 있는가?
- SVD rank-k 근사는 어느 레이어에 먼저 적용해야 하는가?
- PCA는 시각화용인가, 실제 성능 개선용인가?

## 3. 개념 지도
- Eigen decomposition: 주로 대칭 행렬(공분산, Laplacian, Hessian 근사) 해석에 유리
- SVD: 임의 행렬 분해의 표준 도구
- PCA: 공분산 eigenvector 기반 차원 축소

## 4. 연구자 관점
- 스펙트럼은 데이터/모델의 유효 자유도를 보여준다.
- tail singular values를 보면 노이즈 축과 유용 축의 경계를 가늠할 수 있다.
- 학습 중 스펙트럼 변화는 representation phase transition 신호가 된다.

## 5. 엔지니어 관점
- 전체 SVD는 고비용이므로 top-k 추적 또는 샘플링 기반 근사를 활용한다.
- 압축은 accuracy/FLOPs/latency/메모리 네 축으로 평가해야 한다.
- PCA/whitening은 온라인 인덱싱 성능(속도+정확도)에 실질적 영향이 있다.

## 6. CV 예시
- Self-supervised feature 분석: covariance eigen-spectrum으로 collapse 진단.
- ViT/Conv 백본 압축: projection weight를 rank-k로 근사.
- 3D geometry: SVD로 rank 제약을 만족하는 해 구성.

## 7. 딥러닝 예시
- Low-rank finetuning(LoRA류): 업데이트를 저랭크 경로로 제한.
- Truncated SVD pruning: 큰 행렬을 작은 인자 행렬 곱으로 교체.
- PCA-based feature compression: 검색용 임베딩 저장 비용 절감.

## 8. 논문에서 보이는 형태
- "spectral analysis"
- "effective rank"
- "nuclear norm regularization"
- "principal components of feature covariance"
- "rank-constrained optimization"

## 9. 구현 체크리스트
1. 스펙트럼 분석 대상 레이어를 고정(예: QKV, MLP 첫 projection)
2. rank sweep 실험에서 동일 학습 스텝/데이터로 비교
3. SVD 연산은 필요 시 FP32로 캐스팅해 안정성 확보
4. 압축 전후 latency는 실제 배포 장비에서 측정
5. PCA 적용 시 fit 데이터와 eval 데이터 누수 방지

## 10. 미니 실험 아이디어
```python
import torch

W = torch.randn(1024, 512)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)

for k in [16, 32, 64, 128, 256]:
    Wk = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    rel_err = torch.norm(W - Wk) / (torch.norm(W) + 1e-12)
    print(f"k={k}, rel_err={float(rel_err):.6f}")

# effective rank
p = S / (S.sum() + 1e-12)
erank = torch.exp(-(p * torch.log(p + 1e-12)).sum())
print("effective rank:", float(erank))
```

## 11. 자주 나오는 오해
- "PCA는 시각화 전용": retrieval/압축/노이즈 제거에 실전적이다.
- "rank 낮추면 무조건 망가진다": 레이어별 민감도가 다르고, 재학습으로 회복 가능.
- "spectral norm만 보면 충분": tail 구조와 effective rank도 함께 봐야 한다.

## 12. 연결 문서
- [허브 문서](./linear-algebra-for-cv.md)
- [저랭크 효율화](./low-rank-efficient-models.md)
- [임베딩/메트릭](./embedding-and-metric-geometry.md)
