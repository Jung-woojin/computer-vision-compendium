# Analysis Scripts

## 1) 메타데이터 수집
```powershell
python .\analysis\fetch_openaccess_papers.py --venues CVPR2024 CVPR2025 ICCV2025 ECCV2024
```

## 2) abstract 수집 보강
```powershell
python .\analysis\enrich_abstracts.py --input data/raw/papers_openaccess.csv --output data/raw/papers_with_abstracts.csv --workers 12
```

## 3) taxonomy 분류 (title + abstract)
```powershell
python .\analysis\classify_with_abstract.py --input data/raw/papers_with_abstracts.csv
```

## 4) 자동 리포트 생성
```powershell
python .\analysis\build_trend_report.py --tagged data/processed/papers_tagged_abstract.csv --summary data/reports/trend_summary_by_theme_abstract.csv
python .\analysis\build_cross_venue_report.py --summary data/reports/trend_summary_by_theme_abstract.csv
```
