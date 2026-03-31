# 방법론 문서 (Methodology)

## 1) 분석 목적
CVPR/ICCV/ECCV 논문 동향을 "느낌"이 아닌 재현 가능한 방식으로 요약한다.

## 2) 분석 단위
- 기본 단위: 논문 1편
- 1차 메타데이터: 제목, venue, year, source_url
- 2차 메타데이터: taxonomy 기반 theme tags

## 3) 데이터 수집
- 기본 소스:
  - CVPR/ICCV: OpenAccess(The CVF)
  - ECCV 2024: ECVA virtual papers 페이지
- 수집 스크립트: `analysis/fetch_openaccess_papers.py`
- abstract 보강: `analysis/enrich_abstracts.py`
- 기본 산출물:
  - `data/raw/papers_openaccess.csv`
  - `data/raw/papers_with_abstracts.csv`

## 4) 분류 기준 (taxonomy)
- 분류 파일: `taxonomy/cv_taxonomy_v1.yaml`
- 방식:
  - 기본: 제목 기반 키워드 매칭
  - 확장: 제목 + abstract 가중 점수 매칭 (`classify_with_abstract.py`)
- 다중 태깅 허용
- 예:
  - `3d_vision`: `3d`, `neural rendering`, `gaussian splatting`, `slam`
  - `multimodal_vlm`: `vision-language`, `multimodal`, `vlm`, `clip`
  - `agents_reasoning`: `agent`, `planner`, `reasoning`, `tool use`

## 5) 집계 방식
- 연도별 테마 논문 수
- venue-연도별 테마 분포
- growth ratio (year-over-year)
- 다중 태깅이므로 총합과 테마별 합이 일치하지 않을 수 있음

## 6) 해석 시 주의점
1. 제목 키워드 기반 분류는 한계가 있음 (false positive/negative)
2. acceptance 규모가 venue/year마다 다르므로 raw count 해석 주의
3. 같은 테마라도 표현 키워드가 바뀌는 경우 taxonomy 갱신 필요
4. "인기"와 "중요도"는 다름 (산업 파급, 재현성, 하드웨어 비용 등 별도 판단 필요)

## 7) 개선 계획
- abstract 임베딩 기반 분류(규칙 + 임베딩 하이브리드) 추가
- 코드/모델 공개 여부, 벤치마크 영향도 지표 추가
- ICCV/ECCV 포함 교차 venue normalization
- 트렌드와 실환경 태스크(robustness, edge, small object) 연결 분석 강화
