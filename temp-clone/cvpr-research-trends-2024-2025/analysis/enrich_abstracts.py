import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


def extract_abstract(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    selectors = [
        "div#abstract",
        "div.abstract",
        "p#abstract",
        "meta[name='description']",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if not el:
            continue
        if el.name == "meta":
            text = (el.get("content") or "").strip()
        else:
            text = el.get_text(" ", strip=True)
        if text:
            return text
    return ""


def fetch_single(url: str, timeout: int, retries: int, sleep_sec: float) -> Tuple[str, str]:
    headers = {"User-Agent": "cv-trend-bot/1.0 (research-use)"}
    last_err = None
    for _ in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            abs_text = extract_abstract(r.text)
            return url, abs_text
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    return url, f"__ERROR__::{last_err}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich paper metadata with abstract text.")
    parser.add_argument("--input", default="data/raw/papers_openaccess.csv")
    parser.add_argument("--output", default="data/raw/papers_with_abstracts.csv")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-papers", type=int, default=0, help="0 means all")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "abstract" not in df.columns:
        df["abstract"] = ""
    if "abstract_status" not in df.columns:
        df["abstract_status"] = "missing"

    mask_missing = df["abstract"].fillna("").astype(str).str.len() == 0
    target_idx = df[mask_missing].index.tolist()

    if args.max_papers and args.max_papers > 0 and len(target_idx) > args.max_papers:
        rnd = random.Random(args.shuffle_seed)
        rnd.shuffle(target_idx)
        target_idx = target_idx[: args.max_papers]

    print(f"[info] total rows: {len(df)}")
    print(f"[info] abstract missing rows: {mask_missing.sum()}")
    print(f"[info] target rows for fetch: {len(target_idx)}")

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i in target_idx:
            url = str(df.at[i, "paper_url"])
            futures[ex.submit(fetch_single, url, args.timeout, args.retries, args.sleep_sec)] = i

        done = 0
        for fu in as_completed(futures):
            i = futures[fu]
            _, text = fu.result()
            if text.startswith("__ERROR__::"):
                df.at[i, "abstract_status"] = "error"
                df.at[i, "abstract"] = ""
            elif text:
                df.at[i, "abstract_status"] = "ok"
                df.at[i, "abstract"] = text
            else:
                df.at[i, "abstract_status"] = "empty"
                df.at[i, "abstract"] = ""

            done += 1
            if done % 200 == 0:
                print(f"[progress] {done}/{len(target_idx)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    status_counts: Dict[str, int] = df["abstract_status"].value_counts(dropna=False).to_dict()
    print(f"[done] saved: {args.output}")
    print(f"[done] abstract_status counts: {status_counts}")


if __name__ == "__main__":
    main()
