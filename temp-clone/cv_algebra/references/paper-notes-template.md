# Paper Note Template (CV Algebra)

이 템플릿은 컴퓨터비전 연구 논문을 **선형대수 관점**에서 분석할 때 사용합니다.

---

## 1) 논문 정보

| 항목 | 내용 |
|------|------|
| **제목** | |
| **저자** | |
| **학회/연도** | |
| **링크** | |
| **코드** | |

---

## 2) 핵심 주장 (한 문장)

> 논문의 주요 기여를 한 문장으로 요약합니다.

---

## 3) 선형대수 포인트

### 3.1 핵심 연산/구조

- [ ] **행렬 분해**: SVD, Eigen, Cholesky, QR, Low-rank
- [ ] **선형 연산자**: Projection, Convolution, Attention
- [ ] **거리/유사도**: Cosine, L2, Mahalanobis, Inner product
- [ ] **곡률/조건수**: Hessian, Condition number, Curvature

### 3.2 제약 조건

| 제약 | 적용 부분 | 의미 |
|------|----------|------|
| Low-rank | | |
| Orthogonality | | |
| Sparsity | | |
| Non-negativity | | |

### 3.3 수식 핵심

```
# 핵심 수식을 여기에 기록
# 예: A = U @ diag(S) @ V.T, rank-k 근사
```

---

## 4) 구현 체크

### 4.1 Shape 추적

| 연산 | 입력 shape | 출력 shape | 주의점 |
|------|-----------|-----------|--------|
| | | | |

### 4.2 수치 안정성

- [ ] Normalize 위치 고정
- [ ] Epsilon 추가 (avoid division by zero)
- [ ] FP32 casting (SVD/eigen 안정성)
- [ ] Mixed precision 호환성

### 4.3 실험 재현 로그

```bash
# 실행 명령어
python experiment.py --config config.yaml

# 기대 결과
- Accuracy: XX.X%
- Latency: XX ms
- FLOPs: XX GFLOPs
```

---

## 5) 내 연구와의 연결

### 5.1 바로 적용 가능한 아이디어

1. 
2. 
3. 

### 5.2 추가 검증 실험

| 실험 | 변수 | 기대 효과 |
|------|------|----------|
| | | |
| | | |

### 5.3 위험 요소

- [ ] Hyperparameter 감 의존
- [ ] Hardware 의존적
- [ ] 데이터 특성에 민감
- [ ] 재현 어려움 예상

---

## 6) 오해와 한계

| 오해 | 진실 |
|------|------|
| | |
| | |

---

## 7) 연결 문서

- [Linear Algebra Hub](../docs/linear-algebra-for-cv.md)
- [Convolution as Operator](../docs/convolution-as-operator.md)
- [Embedding & Metric Geometry](../docs/embedding-and-metric-geometry.md)
- [Eigens, SVD, PCA](../docs/eigens-svd-pca.md)
- [Optimization Curvature](../docs/optimization-curvature.md)
- [Low-rank Efficient Models](../docs/low-rank-efficient-models.md)

---

**작성일**: YYYY-MM-DD  
**수정일**: YYYY-MM-DD  
**작성자**:
