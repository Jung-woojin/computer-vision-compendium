import argparse
import json
import os
from typing import Dict, List, Tuple

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


def count_occurrences(text: str, kw: str) -> int:
    return text.count(kw)


def score_themes(title: str, abstract: str, taxonomy: Dict[str, List[str]], wt_title: float, wt_abs: float) -> Dict[str, float]:
    t = (title or "").lower()
    a = (abstract or "").lower()
    scores: Dict[str, float] = {}
    for theme, kws in taxonomy.items():
        s = 0.0
        for kw in kws:
            if kw in t:
                s += wt_title * max(1, count_occurrences(t, kw))
            if kw in a:
                s += wt_abs * max(1, count_occurrences(a, kw))
        if s > 0:
            scores[theme] = s
    return scores


def pick_labels(scores: Dict[str, float], top_k: int, min_score: float) -> List[str]:
    if not scores:
        return ["other_uncategorized"]
    ranked: List[Tuple[str, float]] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, v in ranked if v >= min_score][: max(1, top_k)]
    if not labels:
        labels = ["other_uncategorized"]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Taxonomy classification with title + abstract scoring.")
    parser.add_argument("--input", default="data/raw/papers_with_abstracts.csv")
    parser.add_argument("--taxonomy", default="taxonomy/cv_taxonomy_v1.yaml")
    parser.add_argument("--output", default="data/processed/papers_tagged_abstract.csv")
    parser.add_argument("--summary-output", default="data/reports/trend_summary_by_theme_abstract.csv")
    parser.add_argument("--weight-title", type=float, default=2.0)
    parser.add_argument("--weight-abstract", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    taxonomy = load_taxonomy(args.taxonomy)
    if "abstract" not in df.columns:
        df["abstract"] = ""

    theme_scores = []
    themes = []
    for _, row in df.iterrows():
        scores = score_themes(
            title=str(row.get("title", "")),
            abstract=str(row.get("abstract", "")),
            taxonomy=taxonomy,
            wt_title=args.weight_title,
            wt_abs=args.weight_abstract,
        )
        labels = pick_labels(scores, top_k=args.top_k, min_score=args.min_score)
        theme_scores.append(json.dumps(scores, ensure_ascii=False))
        themes.append(";".join(labels))

    df["theme_scores"] = theme_scores
    df["themes"] = themes

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

    print(f"[done] tagged with abstract: {args.output} ({len(df)} rows)")
    print(f"[done] summary: {args.summary_output} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
