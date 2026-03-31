# CVPR 2024-2025 심화 트렌드 리포트 (Deep Dive, Korean)

## 0. 이 문서의 목적
이 문서는 "핫키워드 요약"이 아니라 다음을 목표로 한다.
- 연구 축 이동의 원인 가설 정리
- 실환경 비전(robustness, edge, small object)과의 연결
- ICCV/ECCV 확장 시 검증할 질문 정의

---

## 1. 핵심 결론 (요약 7줄)
1. 2024->2025는 단순 성능 경쟁보다 "시스템화(Agent, Interaction, Tool Use)"가 강해졌다.  
2. 3D/Temporal은 독립 트랙이 아니라 foundation model 내부 capability로 흡수되는 방향이다.  
3. Open-vocabulary는 성숙 단계로 들어가고, zero-shot compositional generalization으로 초점이 이동했다.  
4. Efficient/Edge는 보조 테마가 아니라 실제 배포 제약을 반영한 1급 테마가 되었다.  
5. 복원/강건성(denoise, deblur, weather robustness)은 perception과 분리된 문제에서 "결합 설계"로 이동 중이다.  
6. 벤치마크 SOTA보다 "범용성 + 안정성 + 비용"의 삼중 최적화가 중요해졌다.  
7. ICCV/ECCV 확장 시 같은 결론이 재현되는지 교차 검증이 필수다.

---

## 2. 왜 기존 요약이 얕아 보이는가
요약 레포가 얕아 보이는 전형적 이유는 다음과 같다.
- 분류 기준이 문서화되지 않음
- 근거 데이터와 인사이트가 연결되지 않음
- 연구 축 간 상호작용(예: 3D x Agent, Efficient x Robustness)이 빠짐
- "핫하다"와 "장기 투자 가치"를 구분하지 않음

이 레포 개편의 핵심은 위 4가지를 보완하는 것이다.

---

## 3. 테마별 심화 해석

## 3.1 멀티모달/VLM -> Agent-Ready Vision
### 관측
- 2024: 멀티모달 정렬과 foundation 스케일링 중심
- 2025: 인터랙션, 툴 호출, GUI grounding, actionability 강화

### 해석
- 단일 이미지 이해 성능은 포화 구간 진입
- 차별화는 "멀티스텝 의사결정 + 외부도구 연결"에서 발생

### 연구 공백
- agent 평가가 task success 위주라 시각적 추론 오류 유형 분석이 약함
- 장기 horizon에서 perceptual memory 관리 이슈가 큼

---

## 3.2 3D/Temporal -> 범용 기하 인식으로 수렴
### 관측
- 3D가 별도 파이프라인에서 foundation capability로 흡수
- video/temporal consistency 요구가 generation/perception 전반에 확산

### 해석
- 실제 응용(자율주행, 로보틱스, XR)은 2D-only로는 한계
- "한 장면 정답"보다 "시간축 일관성"이 품질 지표로 부상

### 연구 공백
- zero-shot 3D claim의 데이터 편향 검증 부족
- long-tail geometry/occlusion 시나리오에서 안정성 리포트가 부족

---

## 3.3 Detection/Segmentation -> Open에서 Compositional로
### 관측
- open-vocabulary는 기반 기술로 고착
- compositional prompt / contextual caching / anomaly 쪽으로 확장

### 해석
- 객체 이름 매칭 문제에서 "상황 이해" 문제로 이동
- closed-set 개선은 incremental, open-world 일반화가 차별화 포인트

### 연구 공백
- 합성 프롬프트 조합 폭발에 대한 효율적 search 전략 부족
- segmentation에서 explanation/uncertainty 결합이 아직 약함

---

## 3.4 Efficient/Edge -> 본류화
### 관측
- 경량화가 "학술적 옵션"에서 "배포 필수 요건"으로 이동
- on-device latency/energy-aware 설계 논의 증가

### 해석
- 산업 적용 관점에서 cloud-only 접근의 비용/지연 한계가 명확
- 모델 설계 자체가 하드웨어 제약을 전제로 하는 방향으로 이동

### 연구 공백
- FLOPs 중심 보고가 여전히 많고 실제 장치 latency/energy 리포트가 부족
- robustness와 효율의 동시 최적화 연구가 아직 초기

---

## 3.5 Robustness/Restoration 결합
### 관측
- adverse weather, low-light, motion blur에서 perception 성능 붕괴가 반복 보고
- denoise/deblur를 단독 task가 아니라 perception 전처리로 결합하는 접근 증가

### 해석
- 실환경에서는 데이터 분포 편차가 크고 SNR이 낮다
- "인식 모델만 개선"으로는 한계, 신호 복원과 인식을 같이 설계해야 함

### 연구 공백
- restoration-perception joint training의 안정성/overfit 문제
- 열화 추정 오차가 downstream에 미치는 영향 정량화 부족

---

## 4. 단일 테마가 아닌 "교차 테마"가 중요하다

## 4.1 3D x Agent
- 시각정보를 행동으로 연결하는 핵심 조합
- 로봇/GUI 에이전트에서 scene grounding 실패가 병목

## 4.2 Efficient x Robust
- edge 환경일수록 열화가 심함
- 경량화가 robustness를 깎는 패턴을 줄이는 구조 설계가 필요

## 4.3 Generation x Perception
- 생성 모델은 augmentation/시뮬레이터로, 인식 모델은 평가기로 상호보완 가능
- synthetic-real gap 정량화가 핵심

---

## 5. 실환경 과제와 직접 연결

## 5.1 adverse weather perception
- rain/fog/snow/night에서 feature drift가 크게 발생
- 주파수/대비/SNR 축에서 원인 분해 필요

## 5.2 small object detection
- 다운샘플링/aliasing에 특히 취약
- high-frequency 보존 경로와 anti-aliasing 설계 중요

## 5.3 deblurring/denoising + detection
- `restore -> detect` cascade와 joint training의 trade-off 비교 필요
- 측정은 mAP뿐 아니라 consistency metric, failure mode taxonomy가 필요

---

## 6. ICCV/ECCV 확장 설계 (실행 계획)

## 6.1 데이터 확장
- CVPR뿐 아니라 ICCV/ECCV 동일 스키마 수집
- venue/year별 규모 차이 보정 지표 추가

## 6.2 taxonomy 확장
- 기존 10개 테마 + 실환경 축(robust_perception) 세분화
- 3D/agent/edge 교차 태그를 허용하는 다중 라벨 구조 유지

## 6.3 분석 지표
- raw count
- normalized share (테마 비중)
- growth ratio
- cross-venue consistency score

## 6.4 보고 체계
- 자동 리포트: 스크립트 기반 주기 생성
- 심화 리포트: 연구 질문 중심 해석 문서

---

## 7. 논문 아이디어 발굴용 질문
1. Agent 모델의 시각 추론 실패는 어떤 유형으로 분해할 수 있는가?  
2. 3D capability 주장을 데이터 분포 편향 없이 검증하려면 어떤 평가셋이 필요한가?  
3. Edge 제약에서 robustness 손실을 최소화하는 구조(학습/추론) 설계는 가능한가?  
4. Small object 성능 저하를 aliasing 관점에서 직접 제어하는 모듈은 무엇인가?  
5. 복원-인식 결합 모델에서 일반화 붕괴를 막는 학습 전략은 무엇인가?

---

## 8. 다음 액션 (레포 단위)
1. `analysis/fetch_openaccess_papers.py`로 CVPR/ICCV/ECCV 메타 수집
2. `taxonomy/cv_taxonomy_v1.yaml` 키워드 보강
3. 자동 리포트 생성 후 수동 해석 문서 갱신
4. 실환경 태스크(악천후/소물체) 전용 서브 리포트 분리

---

## 9. 메모
이 문서는 "최종 결론"이 아니라, 데이터 파이프라인 기반으로 반복 업데이트되는 작업 문서다.  
핵심은 한 번의 요약이 아니라, 같은 방법론으로 분기별 추세를 재평가하는 것이다.
