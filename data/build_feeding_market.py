#!/usr/bin/env python3
"""Add FEEDING HABITAT + BANGLADESH MARKET VALUE to data/species_master.csv.

Two independent halves:

1. Feeding habitat / diet (FishBase for fish, SeaLifeBase for the rest --
   the summary URL already stored in each CSV row):
     * the summary page's Ecology line fills `habitat` when empty;
     * the structured "Food items" table (TrophicEco/FoodItemsList.php,
       linked from every summary page) is turned into a short
       `feeding_habitat` phrase: feeding zone + diet items;
     * where a species has no food-items table, the summary page's own
       Biology feeding sentences are used verbatim (cleaned of refs);
     * where neither exists, feeding_habitat stays EMPTY -- nothing is
       invented. `diet_source` records the summary URL used.

2. Market value (Chaldal, chaldal.com -- Bangladeshi online grocer whose
   pages embed the full product catalogue as JSON):
     * species are matched to live listings by exact product name + pack
       size (see PRICE_MATCHES, verified against the live catalogue) and
       the listed price is converted to BDT per kg;
     * species without a listing get an honest `market_status`:
       protected species (dolphins, porpoise, Bryde's whale, turtles,
       water monitor) are "Not sold — protected species", known food
       species are "Local markets — price varies by season/size", the
       rest "Not a commercial food species". No price is ever invented.

Usage:
    python3 data/build_feeding_market.py               # everything
    python3 data/build_feeding_market.py --prices-only # refresh prices only
        (re-fetches Chaldal, updates market_price_bdt_kg / market_source /
         market_asof and prints a diff; FishBase work is untouched)

Prices are indicative retail snapshots, not official rates. Downloads are
cached in data/cache/ (FishBase pages); Chaldal pages are always fetched
live (cache only as an offline fallback).
"""
import argparse
import csv
import datetime
import html as htmlmod
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "species_master.csv"
CACHE = BASE / "cache"
CACHE.mkdir(exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"}
NEW_COLS = ["feeding_habitat", "diet_source",
            "market_price_bdt_kg", "market_status", "market_source", "market_asof"]


def asof_date():
    """Run date in the user's timezone (America/Los_Angeles)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    except Exception:  # noqa: BLE001
        return datetime.date.today().isoformat()


# --------------------------------------------------------------------- fetch
def fetch(url, cache_name, fresh=False, timeout=40):
    """GET url (FishBase/SeaLifeBase: .se host first, then .in/.us mirrors),
    with an on-disk cache. Returns (text, url_actually_used)."""
    path = CACHE / cache_name
    cached = path.read_text(encoding="utf-8", errors="ignore") \
        if path.exists() and path.stat().st_size > 2000 else None
    if cached and not fresh:
        return cached, url
    hosts = [url]
    for suffix in (".se", ".in", ".us"):
        alt = re.sub(r"(fishbase|sealifebase)\.(se|in|us)",
                     lambda m: f"{m.group(1)}{suffix}", url)
        if alt != url:
            hosts.append(alt)
    last_err = "no attempt made"
    for u in dict.fromkeys(hosts):
        try:
            req = urllib.request.Request(u, headers=UA)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read().decode("utf-8", errors="ignore")
            except Exception as read_err:  # noqa: BLE001
                # SeaLifeBase often stalls mid-page; the partial body
                # usually still carries the Ecology/Biology sections.
                partial = getattr(read_err, "partial", None)
                if partial and len(partial) > 8000:
                    data = partial.decode("utf-8", errors="ignore")
                else:
                    raise
            if (len(data) > 2000 and "Just a moment" not in data[:600]
                    and "not in the public version" not in data):
                path.write_text(data, encoding="utf-8")
                return data, u
            last_err = f"blocked/empty page from {u}"
        except Exception as err:  # noqa: BLE001
            last_err = f"{u}: {err}"
        time.sleep(0.4)
    if cached:
        return cached, url
    raise RuntimeError(last_err)


def page_text(raw):
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw,
               flags=re.S | re.I)
    t = htmlmod.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


# ------------------------------------------------------- FishBase / SeaLifeBase
def parse_environment(text):
    """Ecology line -> (habitat string, parts). Drops refs, depth range and
    the climate/coordinate tail."""
    m = re.search(r"\bEcology\s+(.{0,280})", text)
    if not m:
        return "", []
    seg = m.group(1).split(" Distribution")[0]
    seg = re.sub(r"\(Ref[^)]*\)", "", seg)
    keep = []
    for part in (p.strip(" .") for p in seg.split(";")):
        low = part.lower()
        if not part or "depth range" in low or "°" in part:
            break
        keep.append(part)
        if len(keep) >= 5:
            break
    return "; ".join(keep), keep


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = ""

    def handle_data(self, data):
        if self._cell is not None:
            self._cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None


FOOD_I = {"detritus", "plants", "zooplankton", "benthos", "nekton", "other"}


def parse_food_items(raw):
    """Ordered unique (Food I, Food II) pairs from a FoodItemsList page."""
    parser = _TableParser()
    parser.feed(raw)
    items, seen = [], set()
    for row in parser.rows:
        if len(row) >= 2 and row[0].strip().lower() in FOOD_I:
            pair = (row[0].strip().lower(), row[1].strip().lower())
            if pair not in seen:
                seen.add(pair)
                items.append(pair)
    return items


def map_diet(items):
    out = []
    for food_i, food_ii in items:
        phrase = None
        if food_i == "detritus":
            phrase = "detritus"
        elif food_i == "plants":
            phrase = {"phytoplankton": "phytoplankton",
                      "other plants": "aquatic plants",
                      "benthic algae/weeds": "benthic algae and weeds"
                      }.get(food_ii, food_ii or "plants and algae")
        elif food_i == "zooplankton":
            phrase = "zooplankton"
        elif food_i == "benthos":
            phrase = {"mollusks": "benthic mollusks",
                      "crustaceans": "benthic crustaceans (crabs and shrimps)",
                      "insects": "aquatic insects",
                      "worms": "worms"
                      }.get(food_ii, ("benthic " + food_ii) if food_ii
                            else "benthic invertebrates")
        elif food_i == "nekton":
            phrase = {"fish": "fish", "finfish": "fish",
                      "cephalopods": "squid and cuttlefish"
                      }.get(food_ii, food_ii or "fish")
        elif food_i == "other":
            phrase = food_ii or None
        if phrase and phrase not in out:
            out.append(phrase)
    return out[:4]


def join_list(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


_ZONE_PHRASES = [
    (r"surface and mid-?water feeder", "Surface/column feeder"),
    (r"surface feeder", "Surface feeder"),
    (r"mid-?water feeder|column feeder", "Mid-water/column feeder"),
    (r"bottom feeder", "Bottom feeder"),
]


def zone_label(env_parts, text, diet):
    for pattern, label in _ZONE_PHRASES:
        if re.search(pattern, text, re.I):
            zone = label
            break
    else:
        low = " ".join(env_parts).lower()
        if "benthopelagic" in low:
            zone = "Bottom and water-column feeder"
        elif "demersal" in low or "benthic" in low:
            zone = "Bottom (benthic) feeder"
        elif "pelagic" in low:
            zone = "Open-water feeder"
        elif "reef" in low:
            zone = "Reef feeder"
        else:
            zone = ""
    if zone and diet and diet[0] == "fish":
        zone = zone.replace("feeder", "predator")
    return zone


_FEED_WORDS = re.compile(
    r"\b(feed|feeds|feeding|prey|preys|predator|scaveng\w*|diet|eats?|consum\w*)\b",
    re.I)


def parse_biology_feeding(text):
    """The summary page's own Biology feeding sentences (refs stripped)."""
    idx = text.find("Biology")
    if idx < 0:
        return ""
    seg = text[idx:idx + 1800]
    seg = re.sub(r"^Biology\s+Glossary\s*(\(e\.g\.[^)]*\))?\s*", "", seg)
    seg = re.sub(r"\((?:Also\s+)?Refs?\.?[^)]*\)", "", seg)
    picked, seen = [], set()
    for sent in re.split(r"(?<=[.!?])\s+", seg):
        sent = re.sub(r"\s+([.!?])", r"\1", sent.strip())
        key = sent.lower()
        if (10 <= len(sent) <= 220 and _FEED_WORDS.search(sent)
                and key not in seen and "(" not in sent
                and sent[:1].isupper()):
            seen.add(key)
            picked.append(sent)
        if len(picked) >= 2 or len(" ".join(picked)) > 240:
            break
    return re.sub(r"\s+", " ", " ".join(picked)).strip()


def build_feeding(row):
    """-> (habitat_fill, feeding_habitat, diet_source). Any field may be ''
    when the source page genuinely lacks it -- never invented."""
    cid = row["class_id"]
    url = (row.get("fishbase_url") or row.get("sealifebase_url") or "").strip()
    if not url:
        return "", "", ""
    raw, used = fetch(url, f"fb_{cid}_summary.html")
    text = page_text(raw)
    habitat, env_parts = parse_environment(text)
    diet, feeding = [], ""
    stock = re.search(r"vstockcode=(\d+)", raw)
    words = str(row.get("scientific_name", "")).split()
    if stock and len(words) >= 2:
        split = urlsplit(used)
        food_url = (f"{split.scheme}://{split.netloc}/TrophicEco/"
                    f"FoodItemsList.php?vstockcode={stock.group(1)}"
                    f"&genus={words[0]}&species={words[1].lower()}")
        try:
            food_raw, _ = fetch(food_url, f"fb_{cid}_food.html")
            diet = map_diet(parse_food_items(food_raw))
        except Exception:  # noqa: BLE001 -- no food table for this species
            diet = []
    biology = parse_biology_feeding(text)
    sparse_slb = biology and len(diet) <= 2 and "sealifebase" in used
    if diet and not sparse_slb:
        # Structured food-items table wins -- except for SeaLifeBase
        # species whose table is very sparse (<=2 records) while the
        # Biology prose says more (e.g. mud crab's table lists only
        # detritus + finfish, while its Biology reads "Feeds on shrimps,
        # crabs, fishes, and bivalves. Benthic scavenger but prefers
        # oysters.").
        zone = zone_label(env_parts, text, diet)
        feeding = (f"{zone} — {join_list(diet)}" if zone
                   else f"Feeds on {join_list(diet)}")
    else:
        feeding = biology
    if not feeding:
        # Last resort, only from the page's own numbers: FishBase prints a
        # trophic level for most fish -- enough for a coarse guild phrase,
        # never for invented diet items.
        m = re.search(r"Trophic level \(Ref[^)]*\)\s*([\d.]+)", text)
        if m:
            lvl = float(m.group(1))
            guild = ("herbivore / detritus feeder" if lvl < 2.5
                     else "omnivore" if lvl < 3.5 else "carnivore")
            zone = zone_label(env_parts, text, [])
            feeding = (f"{zone} — {guild} (trophic level {lvl:g})" if zone
                       else f"{guild.capitalize()} (trophic level {lvl:g})")
    return habitat, feeding, used


# ------------------------------------------------------------------- market
CHALDAL_SEARCHES = ["fish", "chingri", "hilsha", "koral", "pomfret",
                    "sea-fish", "crab", "squid", "mach", "prawn", "seafood"]
_PRODUCT_RE = re.compile(
    r'"NameWithoutSubText":"((?:[^"\\]|\\.)*)","SubText":"((?:[^"\\]|\\.)*)"'
    r".{0,300}?\"Price\":\{\"Lo\":(\d+(?:\.\d+)?)"
    r".{0,200}?\"DiscountedPrice\":\{\"Lo\":(\d+(?:\.\d+)?)", re.S)

# Verified against the live Chaldal catalogue: class_id ->
# (product NameWithoutSubText, SubText). Price is read live at run time;
# the per-kg conversion uses the pack weight parsed from SubText.
PRICE_MATCHES = {
    "rohu": ("Rui Fish After Cutting ±50 gm", "1 kg"),                    # 439/kg
    "katla": ("Catla Fish After Cutting ±70 gm", "1.5 kg"),               # 789/1.5kg
    "mrigal": ("Mrigel Fish After Cutting ± 50 gm", "1 kg"),              # 369/kg
    "grass_carp": ("Grass Carp Fish After Cutting ±50 gm", "1 kg"),       # 369/kg
    "silver_carp": ("Silver Carp Fish After Cutting ±50 gm", "1 kg"),     # 289/kg
    "common_carp": ("Carfu Fish After Cutting ±50 gm", "1 kg"),           # 479/kg
    "nile_tilapia": ("Telapiya Fish Processed ±50 gm", "1 kg"),           # 349/kg
    "freshwater_shark": ("Boal Fish After Cutting  ±50 gm", "1 kg"),      # 869/kg
    "long_whiskered_catfish": ("Ayer Fish After Cutting ±30 gm", "500 gm"),  # 479/500g
    "pangas": ("Pangas Fish After Cutting ±50 gm", "1 kg"),               # 289/kg
    "koral": ("Koral Fish After Cutting ± 50 gm", "1 kg"),                # 1059/kg
    "shoal": ("Shol Fish Processed ±50 gm", "1 kg"),                      # 849/kg
    "koi_climbing_perch": ("Koi Fish Processed ±50 gm", "1 kg"),          # 369/kg
    "pabda_catfish": ("Pabda Fish ±30 gm", "500 gm"),                     # 199/500g
    "shing": ("Shing Fish Processed ±30 gm", "500 gm"),                   # 299/500g
    "magur": ("Magur Fish Processed ±30 gm", "500 gm"),                   # 399/500g
    "gulsha": ("Gulsha Fish Medium ±30 gm", "500 gm"),                    # 379/500g
    "bele": ("Baila Fish Medium ±30 gm", "500 gm"),                       # 359/500g
    "silver_pomfret": ("Rupchanda Fish Medium ±30 gm", "500 gm"),         # 729/500g
    "tengra": ("Tengra Fish Deshi ±20 gm", "400 gm"),                     # 279/400g
    "punti": ("Puti Fish ±15 gm", "250 gm"),                              # 149/250g
    "mola_carplet": ("Mola Fish Deshi ±15 gm", "250 gm"),                 # 169/250g
    "chapila": ("PQS Chapila Fish Processed", "250 gm"),                  # 165/250g
    "kachki": ("Kaski Fish ±15 gm", "250 gm"),                            # 169/250g
    "bata": ("PQS Bata Fish Processed", "250 gm"),                        # 190/250g
    "giant_river_prawn": ("Golda Chingri (Shrimp) 18-22 pcs ±30 gm", "500 gm"),  # 679/500g
    "bagda_chingri": ("Bagda Chingri (Shrimp) 25-30 pcs ±30 gm", "500 gm"),      # 399/500g
    "harina_chingri": ("Horina Chingri (Shrimp) 90-100 pcs ±15 gm", "250 gm"),   # 219/250g
    "hilsa": ("Whole Hilsha Fish (Asto Ilish) ± 50 gm", "1 kg"),          # 2799/kg
}

PROTECTED_IDS = {
    "ganges_river_dolphin", "irrawaddy_dolphin", "finless_porpoise",
    "humpback_dolphin", "bottlenose_dolphin", "spinner_dolphin",
    "brydes_whale", "northern_river_terrapin", "indian_softshell_turtle",
    "indian_roofed_turtle", "water_monitor",
}
NONCOMMERCIAL_IDS = {"mudskipper", "windowpane_oyster"}


def _json_str(s):
    """Decode a JSON string body (Chaldal embeds \\u00b1 for ± etc.)."""
    try:
        import json
        return json.loads(f'"{s}"')
    except Exception:  # noqa: BLE001
        return s


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _pack_kg(subtext):
    m = re.search(r"([\d.]+)\s*(kg|gm|g)\b", subtext.lower())
    if not m:
        return None
    val = float(m.group(1))
    return val if m.group(2) == "kg" else val / 1000.0


def fetch_chaldal_index():
    """Live product index: {(norm name, norm subtext): (name, subtext, price)}."""
    index = {}

    def absorb(raw):
        for name, sub, price, disc in _PRODUCT_RE.findall(raw):
            p = float(disc) or float(price)
            name, sub = _json_str(name), _json_str(sub)
            index[(_norm(name), _norm(sub))] = (name, sub, p)

    pages = [f"https://chaldal.com/search/{slug}" for slug in CHALDAL_SEARCHES]
    ok = 0
    for i, url in enumerate(pages):
        try:
            raw, _ = fetch(url, f"chaldal_{i}.html", fresh=True, timeout=90)
            absorb(raw)
            ok += 1
        except Exception as err:  # noqa: BLE001
            print(f"  ! Chaldal fetch failed: {url} ({err})")
        time.sleep(0.3)
    if not ok:
        raise RuntimeError("all Chaldal pages failed")
    return index


def market_status_for(row):
    if row["class_id"] in PROTECTED_IDS:
        return "Not sold — protected species"
    if row["class_id"] in NONCOMMERCIAL_IDS:
        return "Not a commercial food species"
    return "Local markets — price varies by season/size"


def apply_market(rows, full):
    """Fetch live prices and update the market columns in place.
    Returns (priced_count, diffs, missing)."""
    index = fetch_chaldal_index()
    today = asof_date()
    matches = dict(PRICE_MATCHES)
    diffs, missing, priced = [], [], 0
    for row in rows:
        cid = row["class_id"]
        target = matches.get(cid)
        if target:
            hit = index.get((_norm(target[0]), _norm(target[1])))
            kg = _pack_kg(target[1])
            if hit and kg:
                new_price = str(round(hit[2] / kg))
                old_price = row.get("market_price_bdt_kg", "")
                if old_price != new_price:
                    diffs.append((cid, old_price or "(none)", new_price,
                                  f"{hit[0]} — {hit[1]} — ৳{hit[2]:g}"))
                row["market_price_bdt_kg"] = new_price
                row["market_source"] = "Chaldal"
                row["market_asof"] = today
                row["market_status"] = ""  # a real price supersedes a status
                priced += 1
            else:
                missing.append(cid)
        if not row.get("market_price_bdt_kg"):
            row["market_status"] = market_status_for(row)
            if full:
                row["market_asof"] = today
    return priced, diffs, missing


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prices-only", action="store_true",
                    help="only refresh market prices from Chaldal "
                         "(skip the FishBase feeding work)")
    args = ap.parse_args()

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in NEW_COLS:
        if col not in fieldnames:
            fieldnames.append(col)
    for row in rows:
        for col in NEW_COLS:
            row.setdefault(col, "")

    if not args.prices_only:
        print(f"Feeding habitat from FishBase/SeaLifeBase for {len(rows)} species...",
              flush=True)
        from concurrent.futures import ThreadPoolExecutor

        def _one(row):
            try:
                habitat, feeding, source = build_feeding(row)
                if habitat and not row.get("habitat"):
                    row["habitat"] = habitat
                if feeding:
                    row["feeding_habitat"] = feeding
                if source:
                    row["diet_source"] = source
                return None
            except Exception as err:  # noqa: BLE001
                return f"  ! {row['class_id']}: {err}"

        with ThreadPoolExecutor(max_workers=4) as pool:
            for err in pool.map(_one, rows):
                if err:
                    print(err, flush=True)
        # SeaLifeBase often drops connections under parallel load: retry
        # whatever is still missing, sequentially and slowly.
        missing = [r for r in rows if not r.get("diet_source")]
        if missing:
            print(f"  sequential retry for {len(missing)} species...", flush=True)
            for row in missing:
                err = _one(row)
                if err:
                    print(err, flush=True)
                time.sleep(1.0)
        ok = sum(1 for r in rows if r.get("diet_source"))
        print(f"  feeding pass done: {ok} species fetched ok", flush=True)

    print("Market prices from Chaldal (live)...")
    priced, diffs, missing = apply_market(rows, full=not args.prices_only)
    print(f"  priced species: {priced}")
    if missing:
        print(f"  ! listings not found this run (old values kept): {missing}")
    if diffs:
        print("  price changes:")
        for cid, old, new, raw in diffs:
            print(f"    {cid}: ৳{old} -> ৳{new} /kg   [{raw}]")
    else:
        print("  no price changes vs the stored values")

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_feed = sum(1 for r in rows if r.get("feeding_habitat"))
    n_hab = sum(1 for r in rows if r.get("habitat"))
    n_price = sum(1 for r in rows if r.get("market_price_bdt_kg"))
    print(f"DONE. feeding_habitat {n_feed}/{len(rows)}, habitat {n_hab}/{len(rows)}, "
          f"real prices {n_price}/{len(rows)}")


if __name__ == "__main__":
    sys.exit(main())
