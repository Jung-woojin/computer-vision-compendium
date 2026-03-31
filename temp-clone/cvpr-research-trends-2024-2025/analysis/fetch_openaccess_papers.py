import argparse
import os
import re
from typing import Dict, List
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_URLS: Dict[str, str] = {
    "CVPR2024": "https://openaccess.thecvf.com/CVPR2024?day=all",
    "CVPR2025": "https://openaccess.thecvf.com/CVPR2025?day=all",
    "ICCV2025": "https://openaccess.thecvf.com/ICCV2025?day=all",
    # ECCV 2024 papers are served on ECVA virtual site rather than CVF OpenAccess.
    "ECCV2024": "https://eccv.ecva.net/virtual/2024/papers.html",
}


def infer_year(venue: str) -> int:
    m = re.search(r"(20\d{2})", venue)
    if m:
        return int(m.group(1))
    return -1


def parse_titles_from_openaccess(html: str, base_url: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    seen = set()

    # Main pattern used by many OpenAccess pages
    for dt in soup.find_all("dt", class_="ptitle"):
        a = dt.find("a")
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        href = a.get("href", "")
        if not title:
            continue
        paper_url = urljoin(base_url, href)
        key = (title.lower(), paper_url)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"title": title, "paper_url": paper_url})

    # Fallback pattern
    if not rows:
        for a in soup.find_all("a", href=True):
            title = a.get_text(" ", strip=True)
            href = a["href"]
            if "/content/" not in href:
                continue
            if len(title) < 8:
                continue
            paper_url = urljoin(base_url, href)
            key = (title.lower(), paper_url)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"title": title, "paper_url": paper_url})

    return rows


def parse_titles_from_eccv_virtual(html: str, base_url: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(" ", strip=True)
        if not title:
            continue
        if len(title) < 12:
            continue
        # ECCV virtual paper detail links are typically /virtual/2024/poster/<id> or /oral/<id>
        if "/virtual/2024/" not in href:
            continue
        if "/papers.html" in href:
            continue
        if title.lower() in {"poster", "oral", "abstract", "paper"}:
            continue

        paper_url = urljoin(base_url, href)
        key = (title.lower(), paper_url)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"title": title, "paper_url": paper_url})

    return rows


def fetch_venue(venue: str, url: str, timeout: int) -> pd.DataFrame:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    if "eccv.ecva.net/virtual/2024/papers.html" in url:
        papers = parse_titles_from_eccv_virtual(resp.text, url)
    else:
        papers = parse_titles_from_openaccess(resp.text, url)
    year = infer_year(venue)
    df = pd.DataFrame(papers)
    if df.empty:
        return pd.DataFrame(columns=["venue", "year", "title", "paper_url", "source_url"])
    df["venue"] = venue
    df["year"] = year
    df["source_url"] = url
    return df[["venue", "year", "title", "paper_url", "source_url"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch paper titles from OpenAccess pages.")
    parser.add_argument(
        "--venues",
        nargs="+",
        default=["CVPR2024", "CVPR2025"],
        help="Venue keys in default url map",
    )
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--output", default="data/raw/papers_openaccess.csv")
    args = parser.parse_args()

    frames: List[pd.DataFrame] = []
    failed = []
    for venue in args.venues:
        url = DEFAULT_URLS.get(venue)
        if not url:
            failed.append((venue, "missing URL in DEFAULT_URLS"))
            continue
        try:
            df = fetch_venue(venue, url, args.timeout)
            frames.append(df)
            print(f"[ok] {venue}: {len(df)} rows")
        except Exception as e:
            failed.append((venue, str(e)))
            print(f"[fail] {venue}: {e}")

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["venue", "title"]).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["venue", "year", "title", "paper_url", "source_url"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[done] saved: {args.output} ({len(out)} rows)")

    if failed:
        print("[warnings]")
        for venue, err in failed:
            print(f"  - {venue}: {err}")


if __name__ == "__main__":
    main()
