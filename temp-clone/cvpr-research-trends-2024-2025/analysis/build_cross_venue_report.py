import argparse
import os
from datetime import datetime

import pandas as pd


def theme_share_table(summary: pd.DataFrame) -> pd.DataFrame:
    totals = summary.groupby(["venue", "year"], as_index=False)["paper_count"].sum().rename(
        columns={"paper_count": "venue_year_total"}
    )
    merged = summary.merge(totals, on=["venue", "year"], how="left")
    merged["share_pct"] = (merged["paper_count"] / (merged["venue_year_total"] + 1e-9)) * 100.0
    return merged


def top_n_by_group(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return (
        df.sort_values(["venue", "year", "share_pct"], ascending=[True, True, False])
        .groupby(["venue", "year"], as_index=False)
        .head(n)
    )


def build_md(summary: pd.DataFrame, source_file: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    years = sorted(summary["year"].dropna().astype(int).unique().tolist())
    venues = sorted(summary["venue"].dropna().astype(str).unique().tolist())

    share = theme_share_table(summary)
    top = top_n_by_group(share, n=8)

    lines = []
    lines.append("# CVPR/ICCV/ECCV 교차 트렌드 리포트 (자동 생성)")
    lines.append("")
    lines.append(f"- 생성 시각: {now}")
    lines.append(f"- 입력 요약 파일: `{source_file}`")
    lines.append(f"- 분석 venue: {', '.join(venues)}")
    lines.append(f"- 분석 연도: {', '.join(map(str, years))}")
    lines.append("")

    lines.append("## 1) Venue-Year별 Top 테마 (비중 기준)")
    for venue in venues:
        for year in years:
            sub = top[(top["venue"] == venue) & (top["year"] == year)]
            if sub.empty:
                continue
            lines.append("")
            lines.append(f"### {venue} {year}")
            lines.append("| theme | papers | share(%) |")
            lines.append("|---|---:|---:|")
            for _, r in sub.iterrows():
                lines.append(f"| {r['theme']} | {int(r['paper_count'])} | {float(r['share_pct']):.2f} |")

    lines.append("")
    lines.append("## 2) 교차 venue 관찰 포인트")
    lines.append("- 동일 연도라도 venue별 테마 비중 차이가 존재할 수 있다.")
    lines.append("- raw count보다 share(%) 비교가 venue 규모 차이를 보정하는 데 유리하다.")
    lines.append("- ICCV/ECCV 확장 후에도 CVPR과 같은 성장 테마가 반복되는지 확인하는 것이 핵심이다.")
    lines.append("")

    lines.append("## 3) 연구용 메모")
    lines.append("- abstract 기반 태깅이므로 title-only보다 주제 분류 신뢰도가 높다.")
    lines.append("- 다중 태깅 허용 구조라 합계가 총 논문수보다 클 수 있다.")
    lines.append("- 다음 단계는 abstract 임베딩 기반 분류기로 taxonomy 오탐/누락을 줄이는 것이다.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-venue report from theme summary.")
    parser.add_argument("--summary", default="data/reports/trend_summary_by_theme_abstract.csv")
    parser.add_argument("--output", default="reports/2024-2025/cross-venue-cvpr-iccv-eccv-ko.md")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    md = build_md(summary, args.summary)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[done] saved: {args.output}")


if __name__ == "__main__":
    main()
