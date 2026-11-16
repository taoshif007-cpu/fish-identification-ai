#!/usr/bin/env python3
"""Download openly-licensed photos per species for (re)training (stdlib only).

Sources:
  * iNaturalist research-grade observations with photos
  * GBIF occurrences that carry a still image

Only photos with a licence are downloaded. Anything with no licence, or a
licence that says "all rights reserved" / copyrighted / non-derivative in a
way the metadata cannot express, is skipped. Every downloaded photo is logged
in dataset_open/photo_metadata.csv with its licence and photographer/observer
so attribution travels with the dataset.

Usage:
  python3 data/download_photos.py                      # 100 per species, all species
  python3 data/download_photos.py --per-species 50
  python3 data/download_photos.py --per-species 20 --species rohu,hilsa

Output layout:  dataset_open/<class_id>/<class_id>_<source>_0001.jpg ...
The dataset_open/ folder is git-ignored: photos are training data, not code.
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent
MASTER = DATA / "species_master.csv"
SEED = DATA / "seed_species.csv"
OUT_DIR = BASE / "dataset_open"
METADATA = OUT_DIR / "photo_metadata.csv"

GBIF = "https://api.gbif.org/v1"
INAT = "https://api.inaturalist.org/v1"
UA = {"User-Agent": "bd-fish-ai-open-data/1.0 (educational project)"}
META_FIELDS = ["class_id", "scientific_name", "source", "licence",
               "photographer", "source_url", "local_file"]


def fetch_json(url):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except Exception as err:  # noqa: BLE001
        print(f"  ! fetch failed: {url} ({err})", file=sys.stderr)
        return None


def licence_ok(licence):
    """True only for an explicit, reusable licence string/code."""
    if not licence:
        return False
    low = str(licence).lower()
    if "all rights reserved" in low or "copyright" in low:
        return False
    # Accept Creative Commons / public-domain marks, in URL or code form.
    return ("creativecommons.org" in low or "publicdomain" in low
            or low.startswith("cc") or "cc0" in low)


def inat_candidates(sci_name, want):
    """Yield dicts for iNaturalist research-grade observation photos."""
    q = urllib.parse.urlencode({
        "taxon_name": sci_name, "quality_grade": "research",
        "photos": "true", "per_page": min(200, max(want * 2, 30)),
        "order_by": "votes",
    })
    data = fetch_json(f"{INAT}/observations?{q}")
    for obs in (data or {}).get("results", []) if isinstance(data, dict) else []:
        user = obs.get("user") or {}
        photographer = user.get("name") or user.get("login") or ""
        for photo in obs.get("photos", []) or []:
            licence = photo.get("license_code") or ""
            if not licence:  # iNat photos without a licence are all-rights-reserved
                continue
            url = photo.get("url") or ""
            for size in ("square", "small", "medium"):
                if f"/{size}." in url:
                    url = url.replace(f"/{size}.", "/original.")
                    break
            if url:
                yield {"url": url, "licence": licence, "photographer": photographer,
                       "source": "iNaturalist",
                       "source_url": f"https://www.inaturalist.org/observations/{obs.get('id', '')}"}


def gbif_candidates(species_key, want):
    """Yield dicts for GBIF image occurrences."""
    if not species_key:
        return
    q = urllib.parse.urlencode({"speciesKey": species_key, "mediaType": "StillImage",
                                "limit": min(300, max(want * 2, 30))})
    data = fetch_json(f"{GBIF}/occurrence/search?{q}")
    for occ in (data or {}).get("results", []) if isinstance(data, dict) else []:
        photographer = occ.get("rightsHolder") or occ.get("recordedBy") or ""
        for media in occ.get("media", []) or []:
            if media.get("type") != "StillImage":
                continue
            licence = media.get("license") or ""
            if not licence_ok(licence):
                continue
            url = media.get("identifier") or ""
            if url:
                yield {"url": url, "licence": licence, "photographer": photographer,
                       "source": "GBIF",
                       "source_url": f"https://www.gbif.org/occurrence/{occ.get('key', '')}"}


def ext_for(url):
    path = urllib.parse.urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def download(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = resp.read()
    if len(data) < 2000:  # error page / thumbnail stub, not a usable photo
        raise ValueError("downloaded file too small")
    dest.write_bytes(data)


def load_species(subset):
    path = MASTER if MASTER.exists() else SEED
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if subset:
        wanted = {s.strip() for s in subset.split(",") if s.strip()}
        rows = [r for r in rows if r["class_id"] in wanted]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-species", type=int, default=100,
                    help="max photos to download per species (default 100)")
    ap.add_argument("--species", default="",
                    help="comma-separated class_id subset, e.g. rohu,hilsa")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_meta = []
    if METADATA.exists():
        with open(METADATA, encoding="utf-8", newline="") as f:
            existing_meta = list(csv.DictReader(f))
    seen_urls = {row["source_url"] + row["local_file"] for row in existing_meta}
    meta_rows = list(existing_meta)

    for sp in load_species(args.species):
        cid, sci = sp["class_id"], sp["scientific_name"]
        folder = OUT_DIR / cid
        folder.mkdir(parents=True, exist_ok=True)
        already = len([p for p in folder.iterdir() if p.is_file()])
        need = max(0, args.per_species - already)
        print(f"{cid} ({sci}): have {already}, downloading up to {need} more")
        if need == 0:
            continue
        got, idx = 0, already
        candidates = list(inat_candidates(sci, need)) + list(gbif_candidates(sp.get("gbif_species_key", ""), need))
        tried_urls = set()
        for cand in candidates:
            if got >= need:
                break
            if cand["url"] in tried_urls:
                continue
            tried_urls.add(cand["url"])
            idx += 1
            dest = folder / f"{cid}_{cand['source'].lower()}_{idx:04d}{ext_for(cand['url'])}"
            try:
                download(cand["url"], dest)
            except Exception as err:  # noqa: BLE001 - one bad photo must not stop the run
                print(f"  ! skip {cand['url']}: {err}", file=sys.stderr)
                continue
            rel = str(dest.relative_to(BASE))
            key = cand["source_url"] + rel
            if key not in seen_urls:
                meta_rows.append({"class_id": cid, "scientific_name": sci,
                                  "source": cand["source"], "licence": cand["licence"],
                                  "photographer": cand["photographer"],
                                  "source_url": cand["source_url"], "local_file": rel})
                seen_urls.add(key)
            got += 1
        print(f"  downloaded {got}")

    with open(METADATA, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"Metadata: {METADATA} ({len(meta_rows)} rows)")


if __name__ == "__main__":
    main()
