#!/usr/bin/env python3
"""Build gate-training folders: dataset_gate/{aquatic, other_living, human, non_living}.

The gate is STAGE 1 of identification (see app.py): aquatic animal vs human
vs other living thing vs non-living / unknown. Stage 2 is the species
classifier. This script only *collects* training photos -- the gate does not
exist until a model is trained on these folders (train.py / Colab flow).

What this script actually downloads (openly licensed, API-accessible):
  * other_living -- iNaturalist research-grade photos of birds, insects,
    plants and fungi (taxon ids below). Only photos with a licence code.
  * aquatic      -- a small per-species sample reusing the same iNaturalist
    photo pipeline as download_photos.py, across the seed species.

What it does NOT download, deliberately:
  * human      -- there is no clean, licence-safe API source for human photos
    reachable from a plain script, and face datasets must never be used.
    Build this class from a documented dataset instead (see the SOURCES.md
    this script writes into dataset_gate/: COCO 'person' category, Open
    Images 'Person' class -- check each dataset's licence/terms first).
  * non_living -- same situation: use COCO / Open Images object classes
    (boat, fishing net, bucket, market objects...) per SOURCES.md.

For human / non_living this script creates the folder, the metadata rows
header and SOURCES.md only. Target: 200+ images per gate class before
training (mirroring the repo's unknown_other guidance in README.md).

Usage:
  python3 data/download_negatives.py                    # 200 per downloadable class
  python3 data/download_negatives.py --per-class 20 --classes other_living,aquatic

Output is git-ignored (dataset_gate/).
"""
import argparse
import csv
import sys
import urllib.parse
from pathlib import Path

# Reuse the tested photo pipeline from the sibling script (same folder).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_photos import download, ext_for, fetch_json, inat_candidates  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent
OUT_DIR = BASE / "dataset_gate"
METADATA = OUT_DIR / "photo_metadata.csv"
META_FIELDS = ["gate_class", "scientific_name", "source", "licence",
               "photographer", "source_url", "local_file"]
INAT = "https://api.inaturalist.org/v1"

# iNaturalist iconic-taxon / clade ids used for "other living" negatives.
OTHER_LIVING_TAXA = [
    (3, "Aves (birds)"),
    (47158, "Insecta (insects)"),
    (47126, "Plantae (plants)"),
    (47170, "Fungi (fungi)"),
]

SOURCES_MD = """# Gate dataset — sources for the classes this script does not download

Target: **200+ images per gate class** before training the gate model.

## human/  (NOT downloaded by download_negatives.py)
No clean, licence-safe API source for human photos is available to a plain
script, and **face datasets must never be used**. Documented alternatives —
download them yourself from the official sites and check the licence/terms
before use:
- COCO dataset, `person` category — https://cocodataset.org (images are
  Flickr photos under their individual licences; COCO annotations CC BY 4.0)
- Open Images, `Person` class — https://storage.googleapis.com/openimages/web/index.html
  (images CC BY 2.0, annotations CC BY 4.0)
Prefer full-body / market / fishing-context photos over portraits: the gate
only needs "this is a person", never who the person is. Do not keep faces
where a wider crop works.

## non_living/  (NOT downloaded by download_negatives.py)
Same documented datasets, object classes instead:
- COCO categories such as boat, car, chair, bottle, cup, handbag, suitcase
- Open Images classes such as Boat, Fishing net (as available), Bucket,
  Table, Vehicle
Market, boat and household objects are the most useful negatives for a
fish-market / pond-side photo app.

## aquatic/ and other_living/
Downloaded by download_negatives.py from iNaturalist research-grade
observations (licence code required on every photo; attribution is in
photo_metadata.csv).
"""


def inat_taxon_candidates(taxon_id, want):
    """Licensed iNaturalist photos for a whole taxon (for other_living)."""
    q = urllib.parse.urlencode({
        "taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
        "per_page": min(200, max(want * 2, 30)), "order_by": "votes",
    })
    data = fetch_json(f"{INAT}/observations?{q}")
    for obs in (data or {}).get("results", []) if isinstance(data, dict) else []:
        user = obs.get("user") or {}
        photographer = user.get("name") or user.get("login") or ""
        taxon = obs.get("taxon") or {}
        for photo in obs.get("photos", []) or []:
            licence = photo.get("license_code") or ""
            if not licence:
                continue
            url = photo.get("url") or ""
            for size in ("square", "small", "medium"):
                if f"/{size}." in url:
                    url = url.replace(f"/{size}.", "/original.")
                    break
            if url:
                yield {"url": url, "licence": licence, "photographer": photographer,
                       "source": "iNaturalist",
                       "scientific_name": taxon.get("name", ""),
                       "source_url": f"https://www.inaturalist.org/observations/{obs.get('id', '')}"}


def fill_class(gate_class, candidates, per_class, meta_rows):
    folder = OUT_DIR / gate_class
    folder.mkdir(parents=True, exist_ok=True)
    already = len([p for p in folder.iterdir() if p.is_file()])
    need = max(0, per_class - already)
    print(f"{gate_class}: have {already}, downloading up to {need} more")
    got, idx, tried = 0, already, set()
    for cand in candidates:
        if got >= need:
            break
        if cand["url"] in tried:
            continue
        tried.add(cand["url"])
        idx += 1
        dest = folder / f"{gate_class}_{idx:04d}{ext_for(cand['url'])}"
        try:
            download(cand["url"], dest)
        except Exception as err:  # noqa: BLE001
            print(f"  ! skip {cand['url']}: {err}", file=sys.stderr)
            continue
        meta_rows.append({"gate_class": gate_class,
                          "scientific_name": cand.get("scientific_name", ""),
                          "source": cand["source"], "licence": cand["licence"],
                          "photographer": cand["photographer"],
                          "source_url": cand["source_url"],
                          "local_file": str(dest.relative_to(BASE))})
        got += 1
    print(f"  downloaded {got}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=200,
                    help="target photos per gate class (default 200)")
    ap.add_argument("--classes", default="aquatic,other_living,human,non_living",
                    help="comma-separated subset of gate classes")
    args = ap.parse_args()
    wanted = {c.strip() for c in args.classes.split(",") if c.strip()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "SOURCES.md").write_text(SOURCES_MD, encoding="utf-8")
    meta_rows = []
    if METADATA.exists():
        with open(METADATA, encoding="utf-8", newline="") as f:
            meta_rows = list(csv.DictReader(f))

    if "other_living" in wanted:
        per_taxon = max(1, args.per_class // len(OTHER_LIVING_TAXA))
        cands = []
        for taxon_id, label in OTHER_LIVING_TAXA:
            print(f"Fetching other_living candidates: {label}")
            cands.extend(inat_taxon_candidates(taxon_id, per_taxon))
        fill_class("other_living", cands, args.per_class, meta_rows)

    if "aquatic" in wanted:
        with open(DATA / "seed_species.csv", encoding="utf-8-sig", newline="") as f:
            seeds = list(csv.DictReader(f))
        per_species = max(1, args.per_class // max(1, len(seeds)))
        cands = []
        for seed in seeds:
            for cand in inat_candidates(seed["scientific_name"], per_species):
                cand = dict(cand, scientific_name=seed["scientific_name"])
                cands.append(cand)
        fill_class("aquatic", cands, args.per_class, meta_rows)

    for gate_class in ("human", "non_living"):
        if gate_class in wanted:
            (OUT_DIR / gate_class).mkdir(parents=True, exist_ok=True)
            print(f"{gate_class}: folder created, NOT downloaded — see dataset_gate/SOURCES.md "
                  "for the documented datasets to build this class from.")

    with open(METADATA, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"Metadata: {METADATA} ({len(meta_rows)} rows)")


if __name__ == "__main__":
    main()
