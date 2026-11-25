#!/usr/bin/env python3
"""
Fetch local, square PNG portraits for every PIT WALL driver.

Sources (build-time only; the page never calls either one):
  * OpenF1  /v1/drivers -> headshot_url, matched by driver name/code
  * MediaWiki prop=pageimages, using the Wikipedia url in drivers.csv

Every portrait is normalised to 160x160 RGBA at assets/drivers/<driverRef>.png.
Already-downloaded files are skipped, so the script is safe to re-run.

Run:  python fetch_drivers.py
Use --force to replace existing files.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "drivers"
INDEX = ROOT / "data" / "index.json"
DRIVERS_CSV = ROOT / "drivers.csv"
OPENF1_URL = "https://api.openf1.org/v1/drivers"
WIKI_URL = "https://en.wikipedia.org/w/api.php"
SIZE = 160
HEADERS = {"User-Agent": "PITWALL portrait fetcher/1.0 (offline static dashboard)"}


def norm(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def collect_refs() -> tuple[dict[str, dict], list[str]]:
    drivers = pd.read_csv(DRIVERS_CSV, na_values=["\\N"], low_memory=False)
    meta = {}
    for row in drivers.itertuples(index=False):
        meta[str(row.driverRef)] = {
            "driverId": int(row.driverId),
            "forename": str(row.forename),
            "surname": str(row.surname),
            "code": norm(row.code) if isinstance(row.code, str) else "",
            "url": str(row.url) if isinstance(row.url, str) and row.url != "nan" else "",
        }
    wanted = set()
    if INDEX.exists():
        index = json.loads(INDEX.read_text(encoding="utf-8"))
        for season in index.get("seasons", []):
            season_file = ROOT / "data" / f"s{season['year']}.json"
            if not season_file.exists():
                continue
            data = json.loads(season_file.read_text(encoding="utf-8"))
            for driver_id in data.get("drivers", {}):
                for ref, info in meta.items():
                    if info["driverId"] == int(driver_id):
                        wanted.add(ref)
                        break
        for champion in index.get("champions", []):
            ref = (champion.get("driver") or {}).get("driverRef")
            if ref in meta:
                wanted.add(ref)
    return meta, sorted(wanted)


def openf1_headshots(session: requests.Session) -> dict[str, str]:
    response = session.get(OPENF1_URL, timeout=60)
    response.raise_for_status()
    latest: dict[str, tuple[int, str]] = {}
    for row in response.json():
        url = row.get("headshot_url")
        if not url:
            continue
        key = norm(f"{row.get('first_name', '')}{row.get('last_name', '')}")
        if not key:
            key = norm(row.get("full_name", ""))
        rank = int(row.get("meeting_key") or 0) * 100000 + int(row.get("session_key") or 0)
        if key not in latest or rank >= latest[key][0]:
            latest[key] = (rank, url)
    by_code = {}
    for row in response.json():
        url = row.get("headshot_url")
        code = norm(row.get("name_acronym"))
        if url and code:
            by_code[code] = url
    return {**{key: url for key, (_, url) in latest.items()}, **by_code}


def wiki_thumbnails(session: requests.Session, refs: list[str], meta: dict[str, dict]) -> dict[str, str]:
    out = {}
    for start in range(0, len(refs), 40):
        batch = refs[start : start + 40]
        titles = []
        by_title = {}
        for ref in batch:
            url = meta[ref]["url"]
            if "/wiki/" not in url:
                continue
            title = unquote(url.split("/wiki/", 1)[1].split("#", 1)[0].replace("_", " "))
            titles.append(title)
            by_title[title] = ref
        if not titles:
            continue
        try:
            response = session.get(
                WIKI_URL,
                params={
                    "action": "query", "prop": "pageimages", "format": "json",
                    "pithumbsize": 640, "redirects": 1, "titles": "|".join(titles),
                },
                timeout=60,
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            redirects = {r["from"]: r["to"] for r in response.json().get("query", {}).get("redirects", [])}
            by_title.update({to: by_title[frm] for frm, to in redirects.items() if frm in by_title})
            for page in pages.values():
                title = page.get("title")
                ref = by_title.get(title)
                thumb = (page.get("thumbnail") or {}).get("source")
                if ref and thumb:
                    out[ref] = thumb
        except Exception as exc:
            print(f"  MediaWiki batch {start // 40 + 1} failed: {exc}")
        time.sleep(0.2)
    return out


def commons_fallbacks(session: requests.Session, refs: list[str], meta: dict[str, dict]) -> dict[str, str]:
    """Last resort for drivers whose Wikipedia article has no page image."""
    out = {}
    for ref in refs:
        info = meta[ref]
        query = f"{info['forename']} {info['surname']} portrait"
        try:
            response = session.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query", "generator": "search", "gsrsearch": query,
                    "gsrnamespace": 6, "gsrlimit": 8, "prop": "imageinfo",
                    "iiprop": "url", "iiurlwidth": 640, "format": "json",
                },
                timeout=60,
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            surname = norm(info["surname"])
            for page in sorted(pages.values(), key=lambda p: p.get("index", 99)):
                title = page.get("title", "")
                if surname not in norm(title) or not re.search(r"\.(jpe?g|png)$", title, re.I):
                    continue
                url = ((page.get("imageinfo") or [{}])[0]).get("thumburl")
                if url:
                    out[ref] = url
                    break
        except Exception as exc:
            print(f"  Commons search for {ref} failed: {exc}")
        time.sleep(0.15)
    return out


def normalise_png(content: bytes) -> Image.Image | None:
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except Exception:
        return None
    if image.width < 32 or image.height < 32:
        return None
    # RGBA conversion keeps the source RGB channels intact; no greyscale step.
    image = image.convert("RGBA")
    image.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2))
    return canvas


def save(session: requests.Session, ref: str, url: str) -> bool:
    for attempt in range(2):
        try:
            response = session.get(url, timeout=45, headers={**HEADERS, "Referer": "https://www.formula1.com/"})
            response.raise_for_status()
            image = normalise_png(response.content)
            if image is None:
                continue
            image.save(OUT / f"{ref}.png", format="PNG", optimize=True)
            return True
        except Exception as exc:
            if attempt:
                print(f"  {ref}: download failed ({exc})")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace portraits that already exist")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta, refs = collect_refs()
    print(f"target drivers: {len(refs)}")
    todo = [ref for ref in refs if args.force or not (OUT / f"{ref}.png").exists()]
    print(f"already present: {len(refs) - len(todo)}  to fetch: {len(todo)}")
    if not todo:
        return
    with requests.Session() as session:
        session.headers.update(HEADERS)
        print("fetching OpenF1 driver index ...")
        try:
            openf1 = openf1_headshots(session)
            print(f"  {len(openf1)} named drivers with headshots")
        except Exception as exc:
            print(f"  OpenF1 index failed: {exc}")
            openf1 = {}
        wiki_refs = []
        sources = {}
        for ref in todo:
            info = meta[ref]
            key = norm(f"{info['forename']}{info['surname']}")
            url = openf1.get(key) or openf1.get(info["code"])
            if url:
                sources[ref] = url
            else:
                wiki_refs.append(ref)
        print(f"OpenF1 matches: {len(sources)}  MediaWiki fallback candidates: {len(wiki_refs)}")
        if wiki_refs:
            wiki = wiki_thumbnails(session, wiki_refs, meta)
            sources.update(wiki)
            print(f"  MediaWiki thumbnails found: {len(wiki)}")
        still_missing = [ref for ref in todo if ref not in sources]
        if still_missing:
            commons = commons_fallbacks(session, still_missing, meta)
            sources.update(commons)
            print(f"  Wikimedia Commons search matches: {len(commons)}")
        saved = {"openf1": 0, "mediawiki": 0}
        failed = []
        for ref in todo:
            url = sources.get(ref)
            if not url:
                failed.append(ref)
                continue
            ok = save(session, ref, url)
            if ok:
                saved["mediawiki" if "wikipedia.org" in url else "openf1"] += 1
            else:
                failed.append(ref)
        print(f"saved {saved['openf1']} OpenF1 + {saved['mediawiki']} MediaWiki portraits")
        print(f"missing {len(failed)} portraits" + (": " + ", ".join(failed[:25]) if failed else ""))
    files = list(OUT.glob("*.png"))
    print(f"assets/drivers: {len(files)} PNG files")


if __name__ == "__main__":
    main()
