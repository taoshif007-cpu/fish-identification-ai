"""AquaPorichoy - Streamlit web app.
Run locally:  streamlit run app.py
Needs: models/fish_model.keras, models/class_names.json, and a species table
(data/species_master.csv if built, else species_info.csv).

IDENTIFICATION IS TWO-STAGE:
  Stage 1 - GATE: is this photo an aquatic animal at all, or is it a human,
    another (non-aquatic) living thing, a non-living object, or not a photo
    of an animal at all (screenshot / document / text)?
  Stage 2 - SPECIES CLASSIFIER: which catalogue species is it?
The species model has ONLY fish classes (no unknown_other class), so on its
own it would force every image -- including a screenshot of text -- into a
fish species. The gate therefore has three layers (see decide() below);
the screenshot/document layer is absolute, while a confident species-model
match (high confidence with a clear margin) outranks the generic ImageNet
layer -- the trained specialist beats the generic filter:
  1. An ImageNet filter: a MobileNetV2 pretrained on ImageNet (weights are
     downloaded by Keras at runtime, ~14 MB, never stored in this repo)
     estimates how much probability mass sits on aquatic-animal classes.
     This is a heuristic filter, NOT a trained gate model.
  2. A screenshot/document detector (numpy/PIL statistics): flat, light
     backgrounds with dark text-like marks in many thin horizontal bands.
  3. The species model's own confidence: the optional unknown_other class
     plus the HIGH_CONF / LOW_CONF / MIN_MARGIN thresholds below.
Anything that fails the gate is reported as "not reliably an aquatic
species in the catalogue" -- never forced into a species. A properly
trained gate model (see README "Gate plan" and data/download_negatives.py)
remains future work and would replace layer 1.
"""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "fish_model.keras"
CLASSES_PATH = BASE / "models" / "class_names.json"
INFO_PATH = BASE / "species_info.csv"
MASTER_PATH = BASE / "data" / "species_master.csv"   # open-data table (preferred when present)
# The trained model spells this class "catla"; the data tables use "katla".
CLASS_ALIASES = {"catla": "katla"}
IMG_SIZE = 224
UNKNOWN_CLASS = "unknown_other"      # optional "not one of my species" class
HIGH_CONF, LOW_CONF, MIN_MARGIN = 0.80, 0.60, 0.15
# Gate (layer 1) thresholds: reject when the ImageNet filter puts less than
# GATE_AQUATIC_MIN total probability on aquatic classes AND is itself
# confident (>= GATE_TOP1_MIN) that the top class is something non-aquatic.
# Second, stricter tier: below GATE_AQUATIC_HARD_MIN aquatic probability
# with a non-aquatic top class, reject unless the species model is at
# least GATE_FISH_OVERRIDE confident (a very sure species match outweighs
# a wishy-washy gate; tuned on real fish vs human/bird/plant photos).
GATE_AQUATIC_MIN, GATE_TOP1_MIN = 0.03, 0.30
GATE_AQUATIC_HARD_MIN, GATE_FISH_OVERRIDE = 0.02, 0.90

st.set_page_config(page_title="AquaPorichoy — Aquatic Species Identifier", page_icon="🌊", layout="wide")

st.markdown("""
<style>
/* ---- animated flowing river-gradient hero banner ---- */
.hero {background: linear-gradient(120deg,#083344,#0e7490,#0284c7,#059669,#06b6d4);
       background-size: 300% 300%;
       animation: heroFlow 12s ease infinite;
       color: white; padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
       box-shadow: 0 8px 24px rgba(8,145,178,.25);}
@keyframes heroFlow {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
.hero h1 {color: white; margin: 0; font-size: clamp(1.4rem, 4vw, 2.2rem); font-weight: 800; text-shadow: 0 1px 3px rgba(0,0,0,.25);}
.hero p {margin: .3rem 0 0 0; opacity: .95;}
/* ---- command bar: the bordered container gets the gradient treatment ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {background: linear-gradient(120deg, rgba(6,182,212,.10), rgba(2,132,199,.10), rgba(5,150,105,.10));
       border: 1.5px solid rgba(8,145,178,.45) !important; border-radius: 18px;
       box-shadow: 0 6px 20px rgba(8,145,178,.15);}
/* ---- result / profile cards: soft shadow + fade-in-up entrance ---- */
.card {background: white; color: #0c4a6e; border: 1px solid #bae6fd; border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .8rem;
       box-shadow: 0 4px 18px rgba(8,145,178,.12); animation: fadeInUp .6s ease both;}
@keyframes fadeInUp {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);}}
.card p, .card div, .card span, .card b {color: #0c4a6e;}
.card h4 {margin: 0 0 .4rem 0; color: #0e7490 !important;}
.card .species {font-size: 1.6rem; font-weight: 700; color: #0c4a6e;}
.card .sci {font-style: italic; color: #0891b2;}
.badge {display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600;}
.ok {background:#cffafe; color:#155e75;} .mid {background:#fef3c7; color:#92400e;} .bad {background:#fee2e2; color:#991b1b;}
.small {font-size: .82rem;}
.card .small {color: #475569;}
/* ---- chat bubbles (Ask AquaPorichoy) ---- */
.chat-q {background: linear-gradient(120deg,#0e7490,#0284c7); color: white;
       border-radius: 16px 16px 4px 16px; padding: .6rem .9rem; margin: .35rem 0 .35rem auto;
       max-width: 85%; width: fit-content; box-shadow: 0 3px 10px rgba(8,145,178,.25);
       animation: fadeInUp .4s ease both;}
.chat-a {background: white; color: #0c4a6e; border: 1px solid #bae6fd;
       border-radius: 16px 16px 16px 4px; padding: .6rem .9rem; margin: .35rem auto .35rem 0;
       max-width: 92%; width: fit-content; box-shadow: 0 3px 10px rgba(8,145,178,.12);
       animation: fadeInUp .4s ease both;}
/* ---- primary Identify button: gradient + hover lift ---- */
.stButton button[kind="primary"], div.stButton > button {background: linear-gradient(120deg,#0e7490,#0284c7,#059669);
       border: none; color: white; font-weight: 700;
       transition: transform .15s ease, box-shadow .15s ease;}
.stButton button[kind="primary"]:hover, div.stButton > button:hover {transform: translateY(-2px);
       box-shadow: 0 8px 18px rgba(2,132,199,.35); color: white;}
.stButton button[kind="primary"]:active, div.stButton > button:active {transform: translateY(0);}
/* ---- progress bar: gradient fill ---- */
div.stProgress > div > div > div > div {background: linear-gradient(90deg,#06b6d4,#0284c7,#059669);}
/* ---- whole-interface light animation pack ---- */
/* soft animated water-tinted page background */
.stApp {background: linear-gradient(160deg,#f0fdfa,#f0f9ff,#ecfdf5,#f0f9ff);
       background-size: 300% 300%; animation: heroFlow 24s ease infinite;}
/* everything rises in gently on load, slight stagger via child order */
.block-container > div {animation: fadeInUp .55s ease both;}
.block-container > div:nth-child(2) {animation-delay: .06s;}
.block-container > div:nth-child(3) {animation-delay: .12s;}
.block-container > div:nth-child(4) {animation-delay: .18s;}
/* hero: a slow light sweep shining across the banner */
.hero {position: relative; overflow: hidden;}
.hero::after {content: ""; position: absolute; top: 0; left: -80%; width: 55%; height: 100%;
       background: linear-gradient(100deg, transparent, rgba(255,255,255,.28), transparent);
       transform: skewX(-18deg); animation: heroShine 5.5s ease-in-out infinite; pointer-events: none;}
@keyframes heroShine {0%,55% {left: -80%;} 85%,100% {left: 130%;}}
/* command bar breathes in and lifts a touch on hover */
div[data-testid="stVerticalBlockBorderWrapper"] {animation: fadeInUp .6s ease both;
       transition: transform .25s ease, box-shadow .25s ease;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {transform: translateY(-2px);
       box-shadow: 0 10px 26px rgba(8,145,178,.22);}
/* sidebar: water tint ONLY -- no width/transform/animation rules. Streamlit
   owns the sidebar's show/hide state machine (on phones it slides the
   section with transforms and shows [data-testid="collapsedControl"]);
   a custom animation with fill-mode "both" pinned transform/opacity on the
   section and fought that state machine, leaving the sidebar stuck as a
   narrow letter-stacked strip overlapping the content. */
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#f0fdfa,#f0f9ff);}
/* keep Streamlit's sidebar toggle always visible and tappable */
[data-testid="collapsedControl"] {visibility: visible !important; opacity: 1 !important;
       z-index: 1000001 !important; background: rgba(255,255,255,.88);
       border-radius: 0 10px 10px 0; box-shadow: 0 2px 10px rgba(8,145,178,.35);
       min-width: 2.4rem; min-height: 2.4rem;}
section[data-testid="stSidebar"] h2 {background: linear-gradient(90deg,#0e7490,#059669);
       -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;}
/* section headings get the river gradient */
.stApp h2, .stApp h3 {background: linear-gradient(90deg,#0e7490,#0284c7,#059669);
       -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;}
/* photos fade in and grow a little on hover */
.stImage img {animation: fadeInUp .5s ease both; border-radius: 12px;
       transition: transform .3s ease, box-shadow .3s ease;}
.stImage img:hover {transform: scale(1.015); box-shadow: 0 10px 24px rgba(8,145,178,.25);}
/* upload dropzone + note field: smooth focus glow */
div[data-testid="stFileUploaderDropzone"] {transition: box-shadow .25s ease, border-color .25s ease;}
div[data-testid="stFileUploaderDropzone"]:hover {box-shadow: 0 0 0 3px rgba(6,182,212,.25);}
div[data-testid="stTextInput"] input {transition: box-shadow .25s ease, border-color .25s ease;}
div[data-testid="stTextInput"] input:focus {box-shadow: 0 0 0 3px rgba(6,182,212,.30); border-color: #0891b2;}
/* alerts, tables and badges animate in too */
div[data-testid="stAlert"], div[data-testid="stDataFrame"] {animation: fadeInUp .5s ease both;}
.badge {animation: badgePop .45s ease both;}
@keyframes badgePop {from {opacity: 0; transform: scale(.85);} to {opacity: 1; transform: scale(1);}}
/* ---- mobile: phones are the PRIMARY device ---- */
.stImage img {max-width: 100%; height: auto;}
@media (max-width: 640px) {
  .block-container {padding-left: .8rem; padding-right: .8rem; padding-top: 1rem;}
  /* the soft water-gradient background, a touch more vivid on phones */
  .stApp {background: linear-gradient(160deg,#ccfbf1,#e0f2fe,#d1fae5,#e0f2fe);
         background-size: 300% 300%;}
  /* every column row stacks full-width (command bar becomes one column) */
  [data-testid="stHorizontalBlock"] {flex-wrap: wrap !important;}
  [data-testid="column"] {flex: 1 1 100% !important; min-width: 100% !important;}
  .hero {padding: 1.1rem 1rem; border-radius: 14px;}
  .hero h1 {font-size: 1.4rem;}
  .hero p {font-size: .95rem;}
  .card {padding: .8rem .9rem; border-radius: 18px; line-height: 1.5;}
  .card .species {font-size: 1.3rem;}
  /* upload zone as the big hero action: tall, dashed, inviting tap target */
  div[data-testid="stFileUploaderDropzone"] {min-height: 120px;
         border: 2px dashed rgba(8,145,178,.55) !important; border-radius: 18px;
         background: linear-gradient(120deg, rgba(6,182,212,.12), rgba(5,150,105,.12));}
  div[data-testid="stFileUploaderDropzone"] button {min-height: 48px; font-size: 16px;}
  /* Identify = the primary action: full-width, tall, gentle pulse glow */
  .stButton button[kind="primary"], div.stButton > button {min-height: 52px; width: 100%;
         font-size: 1.05rem; animation: pulseGlow 2.4s ease-in-out infinite;}
  @keyframes pulseGlow {0%,100% {box-shadow: 0 6px 16px rgba(2,132,199,.30);}
         50% {box-shadow: 0 6px 26px rgba(6,182,212,.55);}}
  /* 16px input text prevents iOS auto-zoom on focus; 48px touch targets */
  div[data-testid="stTextInput"] input {font-size: 16px; min-height: 48px;}
  div[data-testid="stAlert"] {border-radius: 14px;}
  .stImage img {border-radius: 14px;}
  div[data-testid="stDataFrame"] {border-radius: 14px;}
}
/* ---- accessibility: honour reduced motion ---- */
@media (prefers-reduced-motion: reduce) {* {animation: none !important; transition: none !important;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    from tensorflow import keras  # imported here so the page can show errors nicely
    model = keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


@st.cache_data
def load_species_info(data_stamp: str):
    """Prefer the open-data master table; fall back to the legacy CSV.

    data_stamp identifies the CSV file's current mtime+size so the cache
    refreshes when the data file changes (st.cache_data otherwise keys only
    on the function code and would keep serving the old table after a
    redeploy that only changed the CSV)."""
    path = MASTER_PATH if MASTER_PATH.exists() else INFO_PATH
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    info = {row["class_id"]: row for row in df.to_dict("records")}
    return info, ("data/species_master.csv (open data: GBIF / iNaturalist / FishBase links)"
                  if path == MASTER_PATH else "species_info.csv")


# ------------------------------------------------------------------ gate
# Stage 1 (see module docstring) is implemented by the functions below.
# They are pure (no Streamlit calls) except load_gate_model, so the whole
# decision can be unit-tested headlessly. decide() is the single place the
# three layers are combined.

# ImageNet (1000-class) labels that are clearly aquatic animals. Only
# labels that actually exist in the ImageNet class index are listed
# (verified against keras' imagenet_class_index.json). Deliberately
# excluded: semi-aquatic / ambiguous labels such as otter, beaver,
# platypus, hippopotamus and African_crocodile.
AQUATIC_IMAGENET_LABELS = frozenset({
    # fish
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "coho", "barracouta", "eel", "anemone_fish",
    "sturgeon", "gar", "lionfish", "puffer", "rock_beauty",
    # crustaceans
    "Dungeness_crab", "rock_crab", "fiddler_crab", "king_crab",
    "American_lobster", "spiny_lobster", "crayfish", "hermit_crab",
    "isopod",
    # mollusks
    "conch", "snail", "sea_slug", "chiton", "chambered_nautilus",
    # echinoderms + cnidarians / corals
    "starfish", "sea_urchin", "sea_cucumber", "jellyfish", "sea_anemone",
    "brain_coral",
    # marine mammals (ImageNet has no humpback/blue-whale classes; the
    # whale classes that exist are grey_whale and killer_whale)
    "dugong", "sea_lion", "grey_whale", "killer_whale",
})


@st.cache_resource(show_spinner="Loading photo filter...")
def load_gate_model():
    """Layer 1 model: MobileNetV2 pretrained on ImageNet.

    Keras downloads the ~14 MB weights at runtime (cached by Keras outside
    this repo); nothing is stored in the repo. Returns
    (model, aquatic_indices, labels), or None if the model or the class
    index cannot be loaded -- the app then relies on gate layers 2+3 only.
    """
    try:
        import json as _json

        from tensorflow.keras.applications import MobileNetV2, imagenet_utils
        from tensorflow.keras.utils import get_file

        model = MobileNetV2(weights="imagenet", include_top=True)
        class_index = getattr(imagenet_utils, "CLASS_INDEX", None)
        if not class_index:
            fpath = get_file(
                "imagenet_class_index.json",
                "https://storage.googleapis.com/download.tensorflow.org/"
                "data/imagenet_class_index.json",
                cache_subdir="models")
            with open(fpath, encoding="utf-8") as f:
                class_index = _json.load(f)
        labels = [class_index[str(i)][1] for i in range(len(class_index))]
        aquatic_idx = [i for i, lab in enumerate(labels)
                       if lab in AQUATIC_IMAGENET_LABELS]
        if not aquatic_idx:
            return None
        return model, aquatic_idx, labels
    except Exception:  # noqa: BLE001 -- gate is best-effort; layers 2+3 remain
        return None


def gate_scores(img, gate=None):
    """Layer 1 scores for a PIL image, or None if the gate is unavailable.

    Returns dict(aquatic_prob, top1_label, top1_prob, top1_aquatic) where
    aquatic_prob is the summed softmax probability over ALL aquatic
    ImageNet classes (the full 1000-vector, not just the top-5).
    """
    if gate is None:
        gate = load_gate_model()
    if gate is None:
        return None
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model, aquatic_idx, labels = gate
    x = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = preprocess_input(np.asarray(x, dtype="float32"))
    probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    top1 = int(np.argmax(probs))
    return {
        "aquatic_prob": float(sum(float(probs[i]) for i in aquatic_idx)),
        "top1_label": labels[top1],
        "top1_prob": float(probs[top1]),
        "top1_aquatic": labels[top1] in AQUATIC_IMAGENET_LABELS,
    }


# Layer 2 thresholds (tuned on rendered screenshots/documents/chat UIs vs
# real fish and other photos: the text images score flat 0.84-0.96 while
# every real photo tested stays <= 0.46, and no real photo fires all four
# conditions at once).
DOC_FLAT_MIN = 0.55    # >55% of pixels flat light (bright + low colour spread)
DOC_DARK_MIN = 0.01    # dark "ink" pixels between 1% ...
DOC_DARK_MAX = 0.45    # ... and 45% of the image
DOC_EDGE_MIN = 0.015   # enough sharp edges for text strokes
DOC_BAND_MIN = 0.20    # >20% of rows look like text lines


def doc_like_features(img):
    """Layer 2 statistics for a PIL image (numpy/PIL only)."""
    x = ImageOps.exif_transpose(img).convert("RGB")
    x.thumbnail((1024, 1024))
    arr = np.asarray(x).astype("float32")
    gray = arr.mean(axis=2)
    spread = arr.max(axis=2) - arr.min(axis=2)
    # "flat light" = bright AND near-grey/uniform colour: white paper, app
    # backgrounds and chat wallpapers all qualify; water/sky in real photos
    # is bright but colour-spread, so it mostly does not.
    flat_fraction = float(((arr.min(axis=2) > 215) & (spread < 30)).mean())
    dark = gray < 150
    dark_fraction = float(dark.mean())
    gx = np.abs(np.diff(gray, axis=1)) > 40
    gy = np.abs(np.diff(gray, axis=0)) > 40
    edge_density = float((gx.mean() + gy.mean()) / 2)
    row_dark = dark.mean(axis=1)
    text_band_fraction = float(((row_dark > 0.005) & (row_dark < 0.60)).mean())
    return {"flat_fraction": flat_fraction, "dark_fraction": dark_fraction,
            "edge_density": edge_density, "text_band_fraction": text_band_fraction}


def is_doc_like(img):
    """True if the image looks like a screenshot / document / page of text
    rather than a photograph. Conservative by design: ALL signals must fire."""
    f = doc_like_features(img)
    return (f["flat_fraction"] > DOC_FLAT_MIN
            and DOC_DARK_MIN <= f["dark_fraction"] <= DOC_DARK_MAX
            and f["edge_density"] > DOC_EDGE_MIN
            and f["text_band_fraction"] > DOC_BAND_MIN)


def is_blank_like(img):
    """True if the image is essentially blank/uniform (a solid colour, a
    blank wall, an empty frame): almost no texture and almost no edges.
    Such an image cannot contain an identifiable animal, yet the species
    model will still force a class for it -- reject it up front. Real
    photos, even blurry or plain-background ones, carry far more texture
    than these thresholds."""
    x = ImageOps.exif_transpose(img).convert("L")
    x.thumbnail((512, 512))
    arr = np.asarray(x).astype("float32")
    if float(arr.std()) >= 12.0:
        return False
    gx = np.abs(np.diff(arr, axis=1)) > 40
    gy = np.abs(np.diff(arr, axis=0)) > 40
    edge_density = float((gx.mean() + gy.mean()) / 2)
    return edge_density < 0.002


def decide(fish_top_id, fish_conf, fish_margin, doc_like, gate, blank_like=False):
    """Combine the three gate layers. Returns (accepted, reason).

    Priority order:
      1. layer 2 (screenshot/document detector) is ABSOLUTE -- a document
         photo once scored rohu 0.91 on the species model, so nothing
         overrides it;
      2. a CONFIDENT species-model match (confidence >= HIGH_CONF with a
         clear margin >= MIN_MARGIN over the runner-up) is ACCEPTED
         regardless of the ImageNet gate -- the trained specialist outranks
         the generic filter. (Bug fix: gate Tier A previously vetoed such
         photos because it never looked at the species confidence.)
      3. otherwise the original rules apply unchanged: reject if the
         ImageNet filter (a) puts < GATE_AQUATIC_MIN total probability on
         aquatic classes while being itself confident (>= GATE_TOP1_MIN)
         about a non-aquatic top-1 class, or (b) puts
         < GATE_AQUATIC_HARD_MIN on aquatic classes with a non-aquatic
         top-1 class and the species model is below GATE_FISH_OVERRIDE;
         or layer 3: the species model predicts unknown_other or is below
         LOW_CONF.
    """
    if doc_like:
        return False, "screenshot/document-like image"
    if blank_like:
        return False, "blank/uniform image"
    if fish_top_id == UNKNOWN_CLASS:
        return False, "species model not confident enough"
    if fish_conf >= HIGH_CONF and fish_margin >= MIN_MARGIN:
        return True, ""
    if (gate is not None
            and gate["aquatic_prob"] < GATE_AQUATIC_MIN
            and not gate["top1_aquatic"]
            and gate["top1_prob"] >= GATE_TOP1_MIN):
        return False, (f"ImageNet gate: top class '{gate['top1_label']}' "
                       f"({gate['top1_prob']:.0%}), aquatic probability "
                       f"{gate['aquatic_prob']:.1%}")
    if (gate is not None
            and gate["aquatic_prob"] < GATE_AQUATIC_HARD_MIN
            and not gate["top1_aquatic"]
            and fish_conf < GATE_FISH_OVERRIDE):
        return False, (f"ImageNet gate (strict): aquatic probability "
                       f"{gate['aquatic_prob']:.1%}, top class "
                       f"'{gate['top1_label']}'")
    if fish_conf < LOW_CONF:
        return False, "species model not confident enough"
    return True, ""


def rec_for(cid, info):
    """Species record for a class id, honouring CLASS_ALIASES; {} if none."""
    return info.get(cid) or info.get(CLASS_ALIASES.get(cid, ""), {}) or {}


def preprocess(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1600, 1600))
    resized = img.resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.asarray(resized, dtype="float32"), 0)   # raw 0-255; model rescales itself


def e(text):
    return html.escape(str(text))


def name_of(cid, info):
    return rec_for(cid, info).get("common_name") or cid.replace("_", " ").title()


def _present(rec, field):
    return bool(str(rec.get(field, "")).strip())


def _fmt_price(val):
    """Prices are stored as whole BDT; pandas may hand them back as floats
    (2799.0) -- render those without the trailing .0."""
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return str(val)


def render_species_profile(rec, info=None):
    """Full species profile from a master/legacy record. Every field is
    optional: partial rows render only what they have, never crash."""
    if not rec:
        return
    # --- classification (full hierarchy when the master table provides it)
    ranks = [("Kingdom", "kingdom"), ("Phylum", "phylum"), ("Class", "class"),
             ("Order", "order"), ("Family", "family"), ("Genus", "genus")]
    shown = [(label, rec[field]) for label, field in ranks if _present(rec, field)]
    if shown:
        body = " &nbsp;|&nbsp; ".join(f"{label}: <b>{e(val)}</b>" for label, val in shown)
        st.markdown(f'<div class="card"><h4>Scientific classification</h4>{body}</div>',
                    unsafe_allow_html=True)
    if _present(rec, "identification_characteristics"):
        st.markdown(f'<div class="card"><h4>Key identification characteristics</h4>'
                    f'{e(rec["identification_characteristics"])}</div>', unsafe_allow_html=True)
    if _present(rec, "habitat") or _present(rec, "distribution"):
        body = ""
        if _present(rec, "habitat"):
            body += f'<h4>Habitat</h4>{e(rec["habitat"])}'
        if _present(rec, "distribution"):
            body += f'<h4 style="margin-top:.7rem">Distribution</h4>{e(rec["distribution"])}'
        st.markdown(f'<div class="card">{body}</div>', unsafe_allow_html=True)
    # --- feeding habitat & diet (FishBase / SeaLifeBase, build_feeding_market.py)
    if _present(rec, "feeding_habitat"):
        st.markdown(f"🍽️ **Feeding habitat & diet:** {e(rec['feeding_habitat'])}")
    # --- Bangladesh market value (indicative retail; refreshed weekly)
    if _present(rec, "market_price_bdt_kg"):
        src = f", {e(rec['market_source'])}" if _present(rec, "market_source") else ""
        asof = f", as of {e(rec['market_asof'])}" if _present(rec, "market_asof") else ""
        st.markdown(f"🏷️ **Market value (Bangladesh):** ৳{e(_fmt_price(rec['market_price_bdt_kg']))} /kg "
                    f"— indicative retail{src}{asof}. Prices vary by size, season and market.")
    elif _present(rec, "market_status"):
        st.markdown(f"🏷️ **Market value (Bangladesh):** {e(rec['market_status'])}.")
    # --- biodiversity: GBIF occurrence counts
    if _present(rec, "bd_occurrences") or _present(rec, "world_occurrences"):
        bits = []
        if _present(rec, "bd_occurrences"):
            bits.append(f"<b>{e(rec['bd_occurrences'])}</b> recorded in Bangladesh")
        if _present(rec, "world_occurrences"):
            bits.append(f"<b>{e(rec['world_occurrences'])}</b> worldwide")
        if _present(rec, "gbif_photo_records"):
            bits.append(f"<b>{e(rec['gbif_photo_records'])}</b> worldwide with photos")
        st.markdown(f'<div class="card"><h4>Biodiversity (GBIF occurrence records)</h4>'
                    f'{" &nbsp;|&nbsp; ".join(bits)}</div>', unsafe_allow_html=True)
    if _present(rec, "sample_photo_url"):
        st.image(str(rec["sample_photo_url"]), caption="Sample photo (open data)", use_container_width=True)
    sims = [s.strip() for s in str(rec.get("similar_species", "")).split(";") if s.strip()]
    if sims:
        display = [name_of(s, info) if info else s.replace("_", " ").title() for s in sims]
        st.markdown("**Visually similar species:** " + ", ".join(display))
    links = []
    if _present(rec, "fishbase_url"):
        links.append(f'[FishBase]({rec["fishbase_url"]})')
    if _present(rec, "sealifebase_url"):
        links.append(f'[SeaLifeBase]({rec["sealifebase_url"]})')
    if _present(rec, "gbif_species_key"):
        links.append(f'[GBIF](https://www.gbif.org/species/{rec["gbif_species_key"]})')
    if _present(rec, "inaturalist_taxon_id"):
        links.append(f'[iNaturalist](https://www.inaturalist.org/taxa/{rec["inaturalist_taxon_id"]})')
    if links:
        st.markdown("**Sources:** " + " · ".join(links))
    if _present(rec, "info_status"):
        st.markdown(f'<p class="small">Information status: {e(rec["info_status"])}</p>',
                    unsafe_allow_html=True)


# ------------------------------------------------------------------ chat Q&A
# Built-in assistant: answers ONLY from the species' own data fields (no
# external LLM, no invented facts). answer_question is a pure function so
# it can be unit-tested headlessly.

# Words that must never count as a species mention on their own: intent
# vocabulary ("how COMMON is it?") and generic group words.
_MENTION_BLOCKLIST = {
    "common", "price", "cost", "market", "habitat", "where", "found",
    "live", "lives", "similar", "classification", "family", "order",
    "genus", "scientific", "local", "name", "bengali", "bangla",
    "protected", "edible", "safe", "record", "records", "biodiversity",
    "population", "rare", "diet", "food", "feed", "fish", "river",
}


def _species_mentioned(question_lower, info):
    """Class id of a species named in the question, or None.

    Matches common_name, local_name (including the Bengali-script part),
    scientific_name and class_id words as case-insensitive substrings;
    the longest match wins so "mud crab" beats "crab". A single class_id
    word (e.g. "tilapia") only counts when it is unique to one species
    and not an intent/generic word."""
    word_freq = {}
    for cid in info:
        for word in set(cid.split("_")):
            word_freq[word] = word_freq.get(word, 0) + 1
    best, best_len = None, 0
    for cid, rec in info.items():
        names = set()
        for field in ("common_name", "scientific_name"):
            val = str(rec.get(field, "")).strip()
            if val:
                names.add(val.lower())
        local = str(rec.get("local_name", "")).strip()
        if local:
            names.add(local.lower())
            for part in re.split(r"[()]", local):
                if part.strip():
                    names.add(part.strip().lower())
        names.add(cid.replace("_", " "))
        for word in cid.split("_"):
            if (len(word) >= 4 and word_freq.get(word, 0) == 1
                    and word not in _MENTION_BLOCKLIST):
                names.add(word)
        for cand in names:
            if cand and cand in question_lower and len(cand) > best_len:
                best, best_len = cid, len(cand)
    return best


def answer_question(question, info, context_id):
    """Answer a question about a species from its data record only.

    Returns (reply_text, species_id_used). If the question names a species,
    that species becomes the context; otherwise the current context is
    used. Empty fields get an honest "I don't have that in my data yet".
    """
    ql = str(question or "").lower()
    mentioned = _species_mentioned(ql, info)
    cid = mentioned or context_id
    if not cid or not rec_for(cid, info):
        return ("Please identify a photo first, or ask me about a species by "
                "name — for example, \"What is the price of Hilsa?\"",
                context_id)
    rec = rec_for(cid, info)
    name = name_of(cid, info)
    switched = bool(mentioned and mentioned != context_id)

    def has(*words):
        return any(w in ql for w in words)

    def price_text():
        if _present(rec, "market_price_bdt_kg"):
            txt = f"about ৳{_fmt_price(rec['market_price_bdt_kg'])} per kg"
            if _present(rec, "market_source"):
                txt += f" ({rec['market_source']}"
                txt += f", as of {rec['market_asof']}" if _present(rec, "market_asof") else ""
                txt += ")"
            return txt
        return ""

    reply = None
    if has("price", "cost", "market", "dam", "দাম", "koto dam"):
        if _present(rec, "market_price_bdt_kg"):
            reply = (f"The {name} currently sells for {price_text()} in Bangladesh — "
                     "indicative retail. Prices vary by size, season and market.")
        elif _present(rec, "market_status"):
            reply = f"For the {name}, market value: {str(rec['market_status']).rstrip('.')}."
        else:
            reply = f"I don't have market information for the {name} in my data yet."
    elif has("protected", "edible", "eat it", "safe", "can i eat", "can we eat"):
        status = str(rec.get("market_status", ""))
        if "protected" in status.lower():
            reply = (f"Yes — the {name} is a protected species in Bangladesh. "
                     "It is not sold legally and should not be eaten.")
        elif _present(rec, "market_price_bdt_kg") or "local markets" in status.lower():
            reply = f"Yes, the {name} is commonly sold as food in Bangladesh"
            reply += f", at {price_text()}." if _present(rec, "market_price_bdt_kg") \
                else " in local markets."
        else:
            reply = (f"I don't have information in my data about whether the {name} "
                     "is protected or eaten — only what's in its profile.")
    elif has("eat", "diet", "food", "feed", "khay", "খায়", "খাবার"):
        if _present(rec, "feeding_habitat"):
            reply = (f"For the {name}, its feeding habitat and diet: "
                     f"{str(rec['feeding_habitat']).rstrip('.')}.")
        else:
            reply = f"I don't have diet information for the {name} in my data yet."
    elif has("habitat", "live", "found", "where", "river", "sea"):
        bits = []
        if _present(rec, "habitat"):
            bits.append(f"Habitat: {str(rec['habitat']).rstrip('.')}")
        if _present(rec, "distribution"):
            bits.append(f"Distribution: {str(rec['distribution']).rstrip('.')}")
        reply = (". ".join(bits) + ".") if bits else \
            f"I don't have habitat information for the {name} in my data yet."
    elif has("similar", "look like", "confus"):
        sims = [s.strip() for s in str(rec.get("similar_species", "")).split(";") if s.strip()]
        if sims:
            display = [name_of(s, info) for s in sims]
            reply = f"The {name} can be confused with: {', '.join(display)}."
        else:
            reply = f"I don't have similar species listed for the {name} in my data yet."
    elif has("common", "rare", "record", "biodiversity", "population", "koto", "how many"):
        bits = []
        if _present(rec, "bd_occurrences"):
            bits.append(f"{rec['bd_occurrences']} occurrence records in Bangladesh")
        if _present(rec, "world_occurrences"):
            bits.append(f"{rec['world_occurrences']} worldwide")
        if _present(rec, "gbif_photo_records"):
            bits.append(f"{rec['gbif_photo_records']} with photos")
        reply = (f"GBIF has {' and '.join(bits)} for the {name}.") if bits else \
            f"I don't have occurrence records for the {name} in my data yet."
    elif has("classification", "family", "order", "scientific", "genus"):
        ranks = [("Kingdom", "kingdom"), ("Phylum", "phylum"), ("Class", "class"),
                 ("Order", "order"), ("Family", "family"), ("Genus", "genus")]
        chain = [f"{label}: {rec[field]}" for label, field in ranks if _present(rec, field)]
        sci = f" Its scientific name is {rec['scientific_name']}." \
            if _present(rec, "scientific_name") else ""
        reply = (f"The {name}'s classification — " + " · ".join(chain) + "." + sci) if chain \
            else f"I don't have classification data for the {name} in my data yet."
    elif has("local name", "bengali", "bangla"):
        if _present(rec, "local_name"):
            reply = f"The local (Bengali) name of the {name} is {rec['local_name']}."
        else:
            reply = f"I don't have a local name for the {name} in my data yet."
    if reply is None:
        bits = [f"The {name}"]
        if _present(rec, "scientific_name"):
            bits[0] += f" ({rec['scientific_name']})"
        if _present(rec, "group"):
            bits[0] += f" is a {str(rec['group']).lower()} species found in Bangladesh."
        else:
            bits[0] += " is a species found in Bangladesh."
        if _present(rec, "habitat"):
            bits.append(f"Habitat: {str(rec['habitat']).rstrip('.')}.")
        if _present(rec, "market_price_bdt_kg"):
            bits.append(f"It sells for {price_text()}.")
        elif _present(rec, "market_status"):
            bits.append(f"Market: {str(rec['market_status']).rstrip('.')}.")
        bits.append("You can ask me about its price, diet, habitat, similar species, "
                    "classification or how common it is.")
        reply = " ".join(bits)
    if switched:
        reply = f"About the {name} — " + reply
    return reply, cid


# ------------------------------------------------------------------ header
st.markdown("""<div class="hero"><h1>🌊 AquaPorichoy</h1>
<p>Bangladesh's aquatic species identifier — snap a photo, know your species.</p></div>""",
            unsafe_allow_html=True)

if not (MODEL_PATH.exists() and CLASSES_PATH.exists()
        and (MASTER_PATH.exists() or INFO_PATH.exists())):
    st.error("Model files not found. Train the model in Google Colab, then place "
             "`fish_model.keras` and `class_names.json` inside the `models/` folder "
             "(see README.md, Step 3).")
    st.stop()

try:
    model, class_names = load_model()
    _data_path = MASTER_PATH if MASTER_PATH.exists() else INFO_PATH
    _stat = _data_path.stat()
    info, info_source = load_species_info(f"{_stat.st_mtime_ns}-{_stat.st_size}")
except Exception as err:  # noqa: BLE001
    st.error(f"Could not load the model. Check that the TensorFlow version in requirements.txt "
             f"matches the one used for training.\n\nDetails: {err}")
    st.stop()

known = [name_of(c, info) for c in class_names if c != UNKNOWN_CLASS]

# Optional dark theme ("water at night"), injected only when the sidebar
# toggle is on; with the toggle off the page markup is exactly the light
# theme. It restyles AquaPorichoy's own background/cards/bar -- Streamlit's
# widget chrome still follows the viewer's Streamlit theme. The global
# prefers-reduced-motion rule above uses !important, so it keeps winning
# over the animated dark background too.
DARK_CSS = """
<style>
.stApp {background: linear-gradient(160deg,#020617,#082f3a,#052e2b,#082f3a);
       background-size: 300% 300%; animation: heroFlow 24s ease infinite;}
.stApp p, .stApp label, .stMarkdown {color: #e0f2f1;}
.card {background: rgba(8,47,73,.92); border-color: #155e75; color: #e6f4f1;
       box-shadow: 0 4px 18px rgba(0,0,0,.45);}
.card p, .card div, .card span, .card b {color: #e6f4f1;}
.card h4 {color: #7dd3fc !important;}
.card .species {color: #f0fdfa;}
.card .sci {color: #67e8f9;}
.card .small {color: #94a3b8;}
.ok {background:#134e4a; color:#99f6e4;} .mid {background:#713f12; color:#fde68a;}
.bad {background:#7f1d1d; color:#fecaca;}
div[data-testid="stVerticalBlockBorderWrapper"] {
       background: linear-gradient(120deg, rgba(6,182,212,.16), rgba(2,132,199,.16), rgba(5,150,105,.16));
       border-color: rgba(103,232,249,.4) !important;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#082f3a,#020617);}
[data-testid="collapsedControl"] {background: rgba(8,47,73,.92); color: #e0f2f1;}
.chat-a {background: rgba(8,47,73,.92); border-color: #155e75; color: #e6f4f1;}
.chat-q {background: linear-gradient(120deg,#155e75,#0369a1); color: #f0fdfa;}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {color: #e0f2f1;}
div[data-testid="stTextInput"] input {background: #0f2f42; color: #f0fdfa;}
div[data-testid="stFileUploaderDropzone"] {background: rgba(8,47,73,.6); color: #e0f2f1;}
div[data-testid="stAlert"] {background: rgba(8,47,73,.85); color: #e0f2f1;}
</style>
"""

with st.sidebar:
    st.caption("Appearance")
    dark_mode = st.toggle("🌙 Dark mode", key="dark_mode", value=False)
    st.caption("Dark mode restyles AquaPorichoy's cards and background.")
    if dark_mode:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    st.header("🌊 About AquaPorichoy")
    st.write("**AquaPorichoy** — *“aquatic identity”* — identifies aquatic species "
             "found in Bangladesh from a photo, and shows the species name with its "
             "classification, habitat and biodiversity.")
    st.write("**How to use:** attach a photo in the command bar, add an optional "
             "note if you like, and press **🔍 Identify**.")
    st.write(f"Species profiles are powered by open data (GBIF, iNaturalist, "
             f"FishBase / SeaLifeBase) covering **{len(info)} species** — fish, "
             "aquatic mammals, crustaceans, mollusks and aquatic reptiles.")
    st.write(f"Photo identification currently recognises **{len(known)} species**:")
    for n in known:
        st.write(f"• {n}")
    st.caption("More species are on the way — their profiles are already data-ready, "
               "and identification expands with the next model training. If a photo "
               "is not one of the recognised species, AquaPorichoy says it cannot "
               "identify it reliably instead of guessing.")
    st.caption(f"Biological information comes from {info_source} and is marked with its "
               "verification status. Predictions are AI estimates, not expert identification.")

# ------------------------------------------------------- command bar
# ONE chat-style command bar: attach a photo AND type an optional note in
# the same bar, then submit once. Built from standard widgets only (no
# st.chat_input) inside a bordered container, which the CSS above styles
# as the gradient command bar. On narrow screens the columns stack.
file = None
image = None
bad_file = False
note = ""
go = False
with st.container(border=True):
    bar_photo, bar_note, bar_go = st.columns([1.1, 2.3, 1])
    with bar_photo:
        st.markdown("**📷 Photo**")
        file = st.file_uploader("Upload a photo of the species",
                                type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
        st.caption("JPG, PNG or WEBP · max 10 MB")
    with bar_note:
        st.markdown("**✍️ Note (optional)**")
        note = st.text_input(
            "Add a note with your photo",
            placeholder="Add a note with your photo (optional) — e.g. where it was caught, size, local name…",
            label_visibility="collapsed")
    if file is not None:
        try:
            image = Image.open(file)
            image.load()
        except Exception:  # noqa: BLE001
            bad_file = True
            image = None
    with bar_go:
        st.markdown("**🔍 Identify**")
        go = st.button("🔍 Identify", type="primary", use_container_width=True,
                       disabled=image is None)
st.caption("The AI identifies from the **photo** — your note is kept as context "
           "and shown with the result; it does not change the prediction yet.")
if bad_file:
    st.error("This file could not be opened as an image. Please try another photo.")

# ------------------------------------------------------------------ result
st.subheader("Result")
if image is not None:
    st.image(image, caption="Your photo", width=480)
if not go:
    st.info("Upload a clear photo of the whole animal (side view works best), then press **Identify**.")
else:
    if note.strip():
        st.markdown(f"**📝 Your note:** {e(note.strip())}")
    if min(image.size) < 100:
        st.warning("The image is very small, so the result may be unreliable.")
    with st.spinner("Analysing..."):
        probs = model.predict(preprocess(image), verbose=0)[0]
        # Gate layers 1+2 (layer 3 is the species confidence itself). The
        # ImageNet gate is best-effort: if it cannot load, gate stays None
        # and layers 2+3 still apply -- see decide().
        doc_like = is_doc_like(image)
        gate = None
        try:
            gate = gate_scores(image)
        except Exception:  # noqa: BLE001
            gate = None
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    top_id, conf = class_names[top], float(probs[top])
    margin = conf - float(probs[second])
    accepted, _gate_reason = decide(top_id, conf, margin, doc_like, gate,
                                    blank_like=is_blank_like(image))
    is_unknown = not accepted

    if is_unknown:
        st.markdown('<div class="card"><span class="badge bad">Cannot identify reliably</span>'
                    '<p class="species">Unknown / not reliable</p>'
                    '<p>The AI is not confident this is an aquatic species it knows. '
                    'The image may be a screenshot, a document or photo of text, a human, '
                    'another animal, a plant, or an object — '
                    'or a fish/aquatic species outside its training data, or the photo may '
                    'be unclear. It will not guess a species in that case. '
                    'Try a sharper, whole-animal, side-view photo of an aquatic species.</p></div>',
                    unsafe_allow_html=True)
    else:
        rec = rec_for(top_id, info)
        if conf >= HIGH_CONF and margin >= MIN_MARGIN:
            badge = '<span class="badge ok">High confidence</span>'
        else:
            badge = '<span class="badge mid">Moderate confidence - please double-check</span>'
        st.markdown(f"""<div class="card">{badge}
<div class="species">{e(name_of(top_id, info))}</div>
<div class="sci">{e(rec.get('scientific_name', 'n/a'))}</div>
<p class="small">Local name: {e(rec.get('local_name', 'n/a'))}</p></div>""", unsafe_allow_html=True)
        if badge.count("mid"):
            st.warning("Confidence is not high or the second guess is close. "
                       "Compare with the similar species below.")

    if not is_unknown:
        # The chat assistant now talks about this species by default.
        st.session_state["chat_species"] = top_id
        # Only show confidence/top-3 for an accepted identification — for a
        # rejected photo they would read as "the app named a species anyway".
        st.write(f"**Prediction confidence:** {conf*100:.1f}%")
        st.progress(min(max(conf, 0.0), 1.0))

        st.markdown("**Top 3 predictions**")
        top3 = pd.DataFrame({
            "Species": [name_of(class_names[i], info) for i in order[:3]],
            "Confidence (%)": [round(float(probs[i]) * 100, 1) for i in order[:3]],
        })
        st.dataframe(top3, hide_index=True, use_container_width=True)

    if not is_unknown:
        rec = rec_for(top_id, info)
        if rec:
            render_species_profile(rec, info)
        else:
            st.warning("No biological information is stored for this species yet.")

# ------------------------------------------------------------------ chat UI
st.subheader("💬 Ask AquaPorichoy")
st.caption("Ask about a species — its price, diet, habitat, similar species, "
           "classification or how common it is. Answers come only from the "
           "species data above, never guessed.")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "chat_species" not in st.session_state:
    st.session_state["chat_species"] = None


def _ask_chat(text):
    text = str(text or "").strip()
    if not text:
        return
    reply, used = answer_question(text, info, st.session_state.get("chat_species"))
    st.session_state["chat_history"].append({"role": "user", "text": text})
    st.session_state["chat_history"].append({"role": "assistant", "text": reply})
    if used:
        st.session_state["chat_species"] = used


for _msg in st.session_state["chat_history"]:
    _cls = "chat-q" if _msg["role"] == "user" else "chat-a"
    st.markdown(f'<div class="{_cls}">{e(_msg["text"])}</div>', unsafe_allow_html=True)

_CHIPS = [("💰 Price", "What is the price?"),
          ("🍽️ What does it eat?", "What does it eat?"),
          ("🌊 Where is it found?", "Where is it found?"),
          ("🐟 Similar species", "Which species are similar?"),
          ("📊 How common is it?", "How common is it?"),
          ("🔬 Classification", "What is its classification?")]
_chip_cols = st.columns(len(_CHIPS))
for _col, (_label, _question) in zip(_chip_cols, _CHIPS):
    with _col:
        if st.button(_label, key=f"chip_{_label}", use_container_width=True):
            _ask_chat(_question)
            st.rerun()

with st.form("chat_form", clear_on_submit=True):
    _typed = st.text_input("Ask about a species",
                           placeholder="e.g. What is the price of Hilsa?",
                           label_visibility="collapsed")
    _sent = st.form_submit_button("Ask")
if _sent and _typed.strip():
    _ask_chat(_typed)
    st.rerun()
if st.session_state["chat_history"]:
    if st.button("Clear chat", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()

st.divider()
st.caption("AI predictions are estimates and not a substitute for expert taxonomic identification.")
