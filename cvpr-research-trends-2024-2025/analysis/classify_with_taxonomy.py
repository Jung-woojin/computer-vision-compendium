import argparse
import os
from typing import Dict, List

import pandas as pd
import yaml


def load_taxonomy(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    themes = obj.get("themes", {})
    mapping: Dict[str, List[str]] = {}
    for theme, conf in themes.items():
        kws = conf.get("keywords", [])
        mapping[theme] = [str(k).strip().lower() for k in kws if str(k).strip()]
    return mapping


def match_themes(title: str, taxonomy: Dict[str, List[str]]) -> List[str]:
    t = title.lower()
    tags = []
    for theme, kws in taxonomy.items():
        if any(k in t for k in kws):
            tags.append(theme)
    if not tags:
        tags = ["other_uncategorized"]
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag papers with taxonomy keywords.")
    parser.add_argument("--input", default="data/raw/papers_openaccess.csv")
    parser.add_argument("--taxonomy", default="taxonomy/cv_taxonomy_v1.yaml")
    parser.add_argument("--output", default="data/processed/papers_tagged.csv")
    parser.add_argument("--summary-output", default="data/reports/trend_summary_by_theme.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    taxonomy = load_taxonomy(args.taxonomy)

    df["themes"] = df["title"].apply(lambda x: ";".join(match_themes(str(x), taxonomy)))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    exploded = df.assign(theme=df["themes"].str.split(";")).explode("theme")
    summary = (
        exploded.groupby(["venue", "year", "theme"], as_index=False)
        .size()
        .rename(columns={"size": "paper_count"})
        .sort_values(["year", "paper_count"], ascending=[True, False])
    )

    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    summary.to_csv(args.summary_output, index=False)

    print(f"[done] tagged: {args.output} ({len(df)} rows)")
    print(f"[done] summary: {args.summary_output} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
