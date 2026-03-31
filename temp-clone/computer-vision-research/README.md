# 컴퓨터 비전 연구 🔍

컴퓨터 비전 연구 프로젝트: 최신 모델, 실험, 및 인사이트 기록

## 🎯 연구 주제

### 1. 객체 검출 (Object Detection)
- **YOLO 시리즈**: v8, v9, v10, World
- **DETR 계열**: DETR, Deformable-DETR, DINO
- **오픈 볼케뮬레이션**: Open Vocabulary Detection

**현재 진행 중**:
- YOLOv9 inference 최적화
- DN-DETR 데노이징 튜닝
- RT-DETR 실시간 성능 분석

### 2. 이미지 분할 (Image Segmentation)
- **semantic segmentation**: U-Net, DeepLab
- **instance segmentation**: Mask R-CNN, YOLACT
- **panoptic segmentation**: Panoptic FPN

### 3. 비전 - 언어 모델 (Vision-Language Models)
- **CLIP**: Contrastive Language-Image Pre-training
- **BLIP**: Bootstrapping Language-Image Pre-training
- **Flamingo**: Few-shot vision-language learning

### 4. 자기-supervised 학습 (Self-Supervised Learning)
- **Contrastive learning**: SimCLR, MoCo
- **Masked modeling**: MAE, BEiT
- **Predictive coding**: DINO, BYOL

## 🧪 실험 노트

### 실험 관리 체계

각 실험은 다음을 기록합니다:
1. **가설**: 어떤 점을 검증하려는가?
2. **세팅**: 하이퍼파라미터, 데이터, 환경
3. **결과**: 정량/정성 결과
4. **인사이트**: 배운 점, 다음 단계

### 실험 템플릿

```markdown
## [실험명]

**가설**: ...
**세팅**:
- 모델: ...
- 데이터: ...
- 하이퍼파라미터: ...

**결과**:
- metrics: ...
- observations: ...

**인사이트**:
- 배운 점: ...
- 다음 단계: ...
```

## 📊 현재 연구 흐름

### 1. 효율적 아키텍처
- 경량화 모델 개발
- 추론 속도 최적화
- 메모리 효율성 개선

### 2. 대규모 전이학습
- Foundation models
- Multi-modal pretraining
- Zero-shot/Few-shot 학습

### 3. 평가 지표 개선
- Robustness evaluation
- Fairness metrics
- Computational efficiency

## 🚀 실험 시작 가이드

1. **주제 선택**: 연구 관심사 정의
2. **문헌 조사**: 관련 논문 읽기
3. **가설 설정**: 검증할 문제 명확화
4. **세팅 준비**: 코드, 데이터, 환경
5. **실행 및 기록**: 결과 저장, 관찰 기록
6. **분석 및 문서화**: 인사이트 도출

---
*마지막 업데이트: 2026-03-30*
