# CV Algebra Knowledge Base

컴퓨터비전(CV) 연구 관점에서 **선형대수를 다시 정리**하는 지식베이스입니다.  
비전 연구자는 선형대수가 "수학 기초"가 아니라, **연구 가설을 세우고 실패 원인을 분해하는 도구**임을 배웁니다.

---

## 🎯 이 레포지토리가 다루는 것

| 주제 | 핵심 질문 | 관련 문서 |
|------|----------|----------|
| **Convolution as Operator** | Convolution 을 행렬연산자로 해석하면 아키텍처 설계와 효율화가 왜 달라지는가? | [docs/convolution-as-operator.md](./docs/convolution-as-operator.md) |
| **Feature Embedding Space** | 임베딩 공간이 "기하학적 구조"로 해석되면 similarity/metric 학습이 왜 달라지는가? | [docs/embedding-and-metric-geometry.md](./docs/embedding-and-metric-geometry.md) |
| **Eigens, SVD, PCA** | 스펙트럼(고유값/특이값)을 보면 representation collapse, 학습 안정성, 압축 효율을 어떻게 감지/개선하는가? | [docs/eigens-svd-pca.md](./docs/eigens-svd-pca.md) |
| **Optimization Curvature** | 학습 불안정을 hyperparameter tuning 이 아닌 Hessian 곡률 관점으로 진단하면 무엇이 달라지는가? | [docs/optimization-curvature.md](./docs/optimization-curvature.md) |
| **Low-rank Efficient Models** | 행렬 저랭크 근사를 활용하면 정확도-효율 trade-off 를 어떻게 체계적으로 조절하는가? | [docs/low-rank-efficient-models.md](./docs/low-rank-efficient-models.md) |

---

## 📚 시작하기

1. **허브 문서**부터 읽기:  
   [📄 선형대수학을 컴퓨터비전 연구에 연결하기](./docs/linear-algebra-for-cv.md)  
   - 왜 CV 연구자가 선형대수를 다시 공부해야 하는가?
   - 10 개 핵심 개념과 구현 체크포인트
   - 자주 나오는 오해와 후속 키워드

2. **세부 문서**로 진입:  
   각 주제를 연구자/엔지니어 관점에서 정리했습니다.  
   미니 실험 코드 포함.

3. **실험 코드** 실행:  
   `experiments/` 폴더에 재현 가능한 예제 코드가 있습니다.

---

## 🔥 주요 통찰

### 비전 모델의 실패 원인 = 선형대수 언어로 설명 가능

| 증상 | 선형대수적 해석 |
|------|----------------|
| Feature covariance 스펙트럼이 한두 축으로 몰림 | **표현 붕괴** (representation collapse) |
| 학습이 불안정하거나 발산 | **곡률 불균형** (Hessian condition number 문제) |
| 검색/분류 성능이 낮음 | **임베딩 기하학** (거리/각도 설계 불일치) |
| 모델은 큰데 효율이 낮음 | **저랭크 구조 미활용** |
| Attention 이 잘 안 됨 | **projection 공간 해석 부재** |

---

## 🧪 실험과 구현

각 문서에는 **미니 실험 아이디어**가 포함되어 있습니다:

- `experiments/svd-rank-sweep.py` — SVD rank sweep 으로 압축 효율 측정
- `experiments/covariance-collapse-check.py` — representation collapse 진단
- `experiments/attention-projection-toy.py` — attention projection 직관 실험

실전 체크포인트는 코드 수준에서 **일관성**과 **재현성**을 확보하는 데 초점을 맞췄습니다.

---

## 🎯 연구자 관점 vs 엔지니어 관점

각 주제는 **두 층위**로 정리했습니다:

- **연구자 관점**:  
  수학적 직관, 실패 원인 분해, 실험 설계
- **엔지니어 관점**:  
  실전 체크리스트, 구현 팁, 성능/레이턴시 측정

---

## 📋 자주 묻는 질문

**Q. 왜 "선형대수 다시 공부"인가?**  
A. 비전 모델은 "큰 네트워크"처럼 보이지만, 내부 연산의 대부분이 행렬/벡터 연산입니다. 성능이 안 나올 때 실제 원인은 선형대수 언어로 설명되는 경우가 많습니다.

**Q. 이 레포지토리를 어떻게 활용해야 하나요?**  
A. 연구 중 실패 원인을 분석할 때, 또는 아키텍처 설계 시 수식/구현 선택의 근거로 사용하세요. 각 문서의 "자주 나오는 오해" 섹션을 참고하면 실수를 줄일 수 있습니다.

**Q. 실험 코드는 어떤 환경에서 실행하나요?**  
A. PyTorch 기반이며, 최소 `torch`, `numpy`로 실행 가능합니다. CUDA 환경에서는 GPU 가속을 사용합니다.

---

## 📝 커밋 정책

- **문서**: 주제별 가이드, 연구자/엔지니어 관점 정리
- **실험**: 재현 가능한 최소 코드, 결과 로그는 별도 관리
- **링크**: 각 주제는 허브 문서에서 단방향 링크

---

## 🤝 기여

이 레포지토리는 **개인의 연구 기록**을 공유하는 형태입니다.  
의견, 추가 주제, 발견한 오해는 Issue 로 남겨주세요.

---

## 📜 License

개인 연구 기록으로, 모든 문서는 **창작共用 라이선스** 하에 공개합니다.  
상세는 각 문서의 머리말을 참고하세요.

---

## 🔗 연결 문서

- [허브 문서](./docs/linear-algebra-for-cv.md)
- [Convolution as Operator](./docs/convolution-as-operator.md)
- [Embedding and Metric Geometry](./docs/embedding-and-metric-geometry.md)
- [Eigens, SVD, PCA](./docs/eigens-svd-pca.md)
- [Optimization Curvature](./docs/optimization-curvature.md)
- [Low-rank Efficient Models](./docs/low-rank-efficient-models.md)

---

**CV Algebra** — 선형대수는 비전 연구의 "수학 기초"가 아니라, **가설과 실패 원인을 분해하는 도구**입니다.
