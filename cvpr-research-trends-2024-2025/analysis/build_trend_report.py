import argparse
import os
from datetime import datetime
from typing import List

import pandas as pd


def top_themes_by_year(summary: pd.DataFrame, year: int, n: int = 8) -> pd.DataFrame:
    s = summary[summary["year"] == year].copy()
    if s.empty:
        return s
    s = s.groupby(["theme"], as_index=False)["paper_count"].sum()
    s = s.sort_values("paper_count", ascending=False).head(n)
    return s


def growth_table(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        summary.groupby(["year", "theme"], as_index=False)["paper_count"]
        .sum()
        .pivot(index="theme", columns="year", values="paper_count")
        .fillna(0)
    )
    years: List[int] = sorted([c for c in pivot.columns if isinstance(c, (int, float))])
    if len(years) >= 2:
        y0, y1 = years[-2], years[-1]
        pivot["growth_abs"] = pivot[y1] - pivot[y0]
        pivot["growth_pct"] = (pivot["growth_abs"] / (pivot[y0] + 1e-9)) * 100.0
    return pivot.reset_index()


def build_markdown(tagged: pd.DataFrame, summary: pd.DataFrame) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    years = sorted(tagged["year"].dropna().astype(int).unique().tolist())
    venues = sorted(tagged["venue"].dropna().astype(str).unique().tolist())

    lines = []
    lines.append("# CV 트렌드 자동 생성 리포트")
    lines.append("")
    lines.append(f"- 생성 시각: {now}")
    lines.append(f"- 분석 venue: {', '.join(venues)}")
    lines.append(f"- 분석 year: {', '.join(map(str, years))}")
    lines.append(f"- 총 논문 수: {len(tagged)}")
    lines.append("")

    lines.append("## 1) 연도별 상위 테마")
    for y in years:
        top = top_themes_by_year(summary, y, n=8)
        lines.append("")
        lines.append(f"### {y}")
        if top.empty:
            lines.append("- 데이터 없음")
            continue
        lines.append("| theme | papers |")
        lines.append("|---|---:|")
        for _, r in top.iterrows():
            lines.append(f"| {r['theme']} | {int(r['paper_count'])} |")

    lines.append("")
    lines.append("## 2) 테마 성장률 (최근 2개 연도 기준)")
    g = growth_table(summary)
    if g.empty:
        lines.append("- 데이터 없음")
    else:
        g = g.sort_values("growth_abs", ascending=False).head(12)
        cols = [c for c in g.columns if c != "theme"]
        year_cols = [c for c in cols if isinstance(c, (int, float))]
        lines.append("| theme | " + " | ".join(map(str, year_cols)) + " | growth_abs | growth_pct |")
        lines.append("|---|" + "|".join(["---:"] * (len(year_cols) + 2)) + "|")
        for _, r in g.iterrows():
            values = [str(int(r[c])) for c in year_cols]
            lines.append(
                f"| {r['theme']} | " + " | ".join(values) + f" | {int(r.get('growth_abs', 0))} | {float(r.get('growth_pct', 0.0)):.1f} |"
            )

    lines.append("")
    lines.append("## 3) 해석 메모")
    lines.append("- keyword 기반 태깅이므로 theme count는 근사치다.")
    lines.append("- venue/year 규모 차이를 보정한 분석은 별도 normalization 테이블이 필요하다.")
    lines.append("- ICCV/ECCV를 같은 파이프라인으로 추가해 cross-venue 추세를 비교하는 것이 다음 단계다.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown trend report from tagged CSV.")
    parser.add_argument("--tagged", default="data/processed/papers_tagged.csv")
    parser.add_argument("--summary", default="data/reports/trend_summary_by_theme.csv")
    parser.add_argument("--output", default="reports/auto-trend-report-ko.md")
    args = parser.parse_args()

    tagged = pd.read_csv(args.tagged)
    summary = pd.read_csv(args.summary)
    md = build_markdown(tagged, summary)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[done] saved: {args.output}")


if __name__ == "__main__":
    main()
