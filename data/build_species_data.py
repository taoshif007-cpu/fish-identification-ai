#!/usr/bin/env python3
"""Build data/species_master.csv from open data (stdlib only, no new deps).

For each species in data/seed_species.csv this fetches:
  * GBIF species match          -> kingdom/phylum/class/order/family/genus, speciesKey
  * GBIF occurrence search      -> occurrence counts: Bangladesh, world, and
                                   world occurrences that carry a still image
  * iNaturalist taxa            -> taxon id, class (from ancestors, GBIF's class
                                   field is often empty), preferred common name,
                                   default (sample) photo
  * FishBase                    -> a summary-page URL built from Genus-species
                                   (linked, not scraped)
Existing species_info.csv text (habitat, distribution, identification
characteristics, similar species) is merged in as a manual override wherever
a row with the same class_id exists.

The seed list carries a `group` per species (Fish / Aquatic mammal / Mollusk /
Crustacean / ...). GBIF and iNaturalist cover all taxa, so non-fish species go
through exactly the same pipeline; only the profile link differs — FishBase
for fish, SeaLifeBase for everything else (fishbase_url stays blank then).

API responses are cached as JSON files under data/cache/ so re-runs are fast
and gentle on the APIs. If an API fails for a species, that species still gets
a row (with blanks) and a note in info_status -- the run never aborts.

Run:  python3 data/build_species_data.py
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent
CACHE = DATA / "cache"
SEED = DATA / "seed_species.csv"
LEGACY_INFO = BASE / "species_info.csv"
OUT = DATA / "species_master.csv"

GBIF = "https://api.gbif.org/v1"
INAT = "https://api.inaturalist.org/v1"
UA = {"User-Agent": "bd-fish-ai-open-data/1.0 (educational project)"}

MASTER_FIELDS = [
    "class_id", "common_name", "local_name", "scientific_name", "group",
    "kingdom", "phylum", "class", "order", "family", "genus",
    "habitat", "distribution", "identification_characteristics",
    "similar_species", "bd_occurrences", "world_occurrences",
    "gbif_photo_records", "sample_photo_url", "fishbase_url", "sealifebase_url",
    "gbif_species_key", "inaturalist_taxon_id", "sources", "info_status",
]


def fetch_json(url, cache_name):
    """GET JSON with a simple file cache. Returns None on any failure."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / cache_name
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt cache: fall through to refetch
            pass
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (https APIs only)
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as err:  # noqa: BLE001 - per-species failures must not abort the run
        print(f"  ! fetch failed: {url} ({err})", file=sys.stderr)
        return None
    try:
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.2)  # be polite to the APIs
    return data


def safe_name(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def gbif_count(species_key, extra):
    q = urllib.parse.urlencode({"speciesKey": species_key, "limit": 0, **extra})
    data = fetch_json(f"{GBIF}/occurrence/search?{q}",
                      f"gbif_occ_{species_key}_{safe_name(str(sorted(extra.items())))}.json")
    if isinstance(data, dict) and isinstance(data.get("count"), int):
        return data["count"]
    return ""


def gbif_sample_photo(species_key):
    q = urllib.parse.urlencode({"speciesKey": species_key, "mediaType": "StillImage", "limit": 5})
    data = fetch_json(f"{GBIF}/occurrence/search?{q}", f"gbif_photo_{species_key}.json")
    for result in (data or {}).get("results", []) if isinstance(data, dict) else []:
        for media in result.get("media", []) or []:
            if media.get("type") == "StillImage" and media.get("identifier"):
                return media["identifier"]
    return ""


def inat_taxon(scientific_name):
    """Return (taxon_id, class_name, sample_photo_url) from iNaturalist."""
    q = urllib.parse.urlencode({"q": scientific_name, "per_page": 10})
    data = fetch_json(f"{INAT}/taxa?{q}", f"inat_search_{safe_name(scientific_name)}.json")
    taxon = None
    for cand in (data or {}).get("results", []) if isinstance(data, dict) else []:
        if (cand.get("name") or "").lower() == scientific_name.lower():
            taxon = cand
            break
    if taxon is None and isinstance(data, dict) and data.get("results"):
        taxon = data["results"][0]
    if not taxon:
        return "", "", ""
    taxon_id = taxon.get("id", "")
    # The search result does not always include ancestors, so fetch the taxon.
    detail = fetch_json(f"{INAT}/taxa/{taxon_id}", f"inat_taxon_{taxon_id}.json")
    detail_taxon = None
    if isinstance(detail, dict) and detail.get("results"):
        detail_taxon = detail["results"][0]
    class_name = ""
    photo_url = ""
    for src in (detail_taxon, taxon):
        if not src:
            continue
        for anc in src.get("ancestors", []) or []:
            if anc.get("rank") == "class" and anc.get("name"):
                class_name = anc["name"]
        if src.get("rank") == "class" and src.get("name"):
            class_name = class_name or src["name"]
        dp = src.get("default_photo") or {}
        if dp.get("medium_url"):
            photo_url = photo_url or dp["medium_url"]
    return taxon_id, class_name, photo_url


def build_row(seed, override):
    sci = seed["scientific_name"].strip()
    parts = sci.split()
    genus = parts[0] if parts else ""
    notes = []

    # ---- GBIF match -------------------------------------------------------
    q = urllib.parse.urlencode({"name": sci})
    match = fetch_json(f"{GBIF}/species/match?{q}", f"gbif_match_{safe_name(sci)}.json") or {}
    species_key = match.get("speciesKey") or match.get("usageKey") or ""
    if not species_key:
        notes.append("GBIF species match failed")

    bd = world = photos = ""
    sample_photo = ""
    if species_key:
        world = gbif_count(species_key, {})
        bd = gbif_count(species_key, {"country": "BD"})
        photos = gbif_count(species_key, {"mediaType": "StillImage"})
        sample_photo = gbif_sample_photo(species_key)

    # ---- iNaturalist ------------------------------------------------------
    taxon_id, inat_class, inat_photo = inat_taxon(sci)
    if not taxon_id:
        notes.append("iNaturalist taxon not found")
    if not sample_photo:
        sample_photo = inat_photo

    # ---- FishBase for fish / SeaLifeBase for everything else (links only) --
    group = (seed.get("group") or "Fish").strip() or "Fish"
    fishbase_url = ""
    sealifebase_url = ""
    if len(parts) >= 2:
        if group == "Fish":
            fishbase_url = f"https://www.fishbase.se/summary/{parts[0]}-{parts[1]}.html"
        else:
            sealifebase_url = f"https://www.sealifebase.se/summary/{parts[0]}-{parts[1]}.html"

    # ---- status ------------------------------------------------------------
    mapping_note = (seed.get("mapping_note") or "").strip()
    tentative = mapping_note.upper().startswith("NEEDS VERIFICATION")
    status_bits = []
    if tentative:
        status_bits.append("NEEDS VERIFICATION: tentative scientific-name mapping "
                           "(see data/seed_species.csv mapping_note) - verify before publishing")
    if override:
        status_bits.append("Open data (GBIF/iNaturalist, fetched) + manual text from "
                           "species_info.csv (still DRAFT - verify against FishBase / DoF Bangladesh)")
    else:
        status_bits.append("Open data only (GBIF/iNaturalist, fetched); no manual text in species_info.csv yet")
    status_bits.extend(notes)

    sources = "GBIF (taxonomy + occurrence counts); iNaturalist (class, sample photo); "
    sources += "FishBase (summary link)" if fishbase_url else "SeaLifeBase (summary link)"
    if override:
        sources += "; species_info.csv (manual habitat/identification override)"

    return {
        "class_id": seed["class_id"].strip(),
        "common_name": seed.get("common_name", "").strip(),
        "local_name": seed.get("local_name", "").strip(),
        "scientific_name": sci,
        "group": group,
        "kingdom": match.get("kingdom", "") or "Animalia",
        "phylum": match.get("phylum", "") or "",
        "class": match.get("class") or inat_class or "",
        "order": match.get("order", "") or "",
        "family": match.get("family", "") or (override or {}).get("family", ""),
        "genus": match.get("genus", "") or genus,
        "habitat": (override or {}).get("habitat", ""),
        "distribution": (override or {}).get("distribution", ""),
        "identification_characteristics": (override or {}).get("identification_characteristics", ""),
        "similar_species": (override or {}).get("similar_species", ""),
        "bd_occurrences": bd,
        "world_occurrences": world,
        "gbif_photo_records": photos,
        "sample_photo_url": sample_photo,
        "fishbase_url": fishbase_url,
        "sealifebase_url": sealifebase_url,
        "gbif_species_key": species_key,
        "inaturalist_taxon_id": taxon_id,
        "sources": sources,
        "info_status": "; ".join(status_bits),
    }


def main():
    with open(SEED, encoding="utf-8-sig", newline="") as f:
        seeds = list(csv.DictReader(f))
    overrides = {}
    if LEGACY_INFO.exists():
        with open(LEGACY_INFO, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                overrides[row["class_id"]] = row

    rows = []
    for seed in seeds:
        print(f"Fetching open data for {seed['class_id']} ({seed['scientific_name']})...")
        try:
            rows.append(build_row(seed, overrides.get(seed["class_id"])))
        except Exception as err:  # noqa: BLE001 - keep the row, note the failure
            print(f"  ! error for {seed['class_id']}: {err}", file=sys.stderr)
            rows.append({field: "" for field in MASTER_FIELDS} | {
                "class_id": seed["class_id"], "common_name": seed.get("common_name", ""),
                "local_name": seed.get("local_name", ""), "scientific_name": seed["scientific_name"],
                "group": seed.get("group", "Fish"),
                "info_status": f"Open-data fetch failed: {err}",
            })

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
