"""Bangladesh Fish Species Identification AI - Streamlit web app.
Run locally:  streamlit run app.py
Needs: models/fish_model.keras, models/class_names.json, species_info.csv
"""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "fish_model.keras"
CLASSES_PATH = BASE / "models" / "class_names.json"
INFO_PATH = BASE / "species_info.csv"
IMG_SIZE = 224
UNKNOWN_CLASS = "unknown_other"      # optional "not one of my species" class
HIGH_CONF, LOW_CONF, MIN_MARGIN = 0.80, 0.50, 0.15

st.set_page_config(page_title="Bangladesh Fish Species Identification AI", page_icon="🐟", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(120deg,#0e7490,#06b6d4 60%,#67e8f9); color: white;
       padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0; font-size: clamp(1.4rem, 4vw, 2.2rem);}
.hero p {margin: .3rem 0 0 0; opacity: .95;}
.card {background: white; color: #0c4a6e; border: 1px solid #bae6fd; border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .8rem;}
.card p, .card div, .card span, .card b {color: #0c4a6e;}
.card h4 {margin: 0 0 .4rem 0; color: #0e7490 !important;}
.card .species {font-size: 1.6rem; font-weight: 700; color: #0c4a6e;}
.card .sci {font-style: italic; color: #0891b2;}
.badge {display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600;}
.ok {background:#cffafe; color:#155e75;} .mid {background:#fef3c7; color:#92400e;} .bad {background:#fee2e2; color:#991b1b;}
.small {font-size: .82rem;}
.card .small {color: #475569;}
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
def load_species_info():
    df = pd.read_csv(INFO_PATH, encoding="utf-8-sig").fillna("")
    return {row["class_id"]: row for row in df.to_dict("records")}


def preprocess(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1600, 1600))
    resized = img.resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.asarray(resized, dtype="float32"), 0)   # raw 0-255; model rescales itself


def e(text):
    return html.escape(str(text))


def name_of(cid, info):
    return info.get(cid, {}).get("common_name") or cid.replace("_", " ").title()


# ------------------------------------------------------------------ header
st.markdown("""<div class="hero"><h1>🐟 Bangladesh Fish Species Identification AI</h1>
<p>Upload a photograph of a fish and the AI will suggest which species it is.</p></div>""",
            unsafe_allow_html=True)

if not (MODEL_PATH.exists() and CLASSES_PATH.exists() and INFO_PATH.exists()):
    st.error("Model files not found. Train the model in Google Colab, then place "
             "`fish_model.keras` and `class_names.json` inside the `models/` folder "
             "(see README.md, Step 3).")
    st.stop()

try:
    model, class_names = load_model()
    info = load_species_info()
except Exception as err:  # noqa: BLE001
    st.error(f"Could not load the model. Check that the TensorFlow version in requirements.txt "
             f"matches the one used for training.\n\nDetails: {err}")
    st.stop()

known = [name_of(c, info) for c in class_names if c != UNKNOWN_CLASS]
with st.sidebar:
    st.header("About")
    st.write(f"This model only knows **{len(known)} species**:")
    for n in known:
        st.write(f"• {n}")
    st.caption("It cannot identify any other species. If your fish is not in this list, "
               "the result will be wrong even if the app sounds confident.")
    st.caption("Biological information comes from species_info.csv and is marked with its "
               "verification status. Predictions are AI estimates, not expert identification.")

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("1. Upload a fish photo")
    file = st.file_uploader("JPG, PNG or WEBP (max 10 MB)", type=["jpg", "jpeg", "png", "webp"])
    image = None
    if file is not None:
        try:
            image = Image.open(file)
            image.load()
            st.image(image, caption="Your photo", use_container_width=True)
        except Exception:  # noqa: BLE001
            st.error("This file could not be opened as an image. Please try another photo.")
            image = None
    go = st.button("🔍 Identify Fish", type="primary", use_container_width=True, disabled=image is None)

with right:
    st.subheader("2. Result")
    if not go:
        st.info("Upload a clear photo of the whole fish (side view works best), then press **Identify Fish**.")
    else:
        if min(image.size) < 100:
            st.warning("The image is very small, so the result may be unreliable.")
        with st.spinner("Analysing..."):
            probs = model.predict(preprocess(image), verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        top_id, conf = class_names[top], float(probs[top])
        margin = conf - float(probs[second])
        is_unknown = top_id == UNKNOWN_CLASS or conf < LOW_CONF

        if is_unknown:
            st.markdown('<div class="card"><span class="badge bad">Cannot identify reliably</span>'
                        '<p class="species">Unknown / not reliable</p>'
                        '<p>The AI is not confident. This may be a species outside its training data, '
                        'or the photo may be unclear. Try a sharper, whole-fish, side-view photo.</p></div>',
                        unsafe_allow_html=True)
        else:
            rec = info.get(top_id, {})
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

        st.write(f"**Prediction confidence:** {conf*100:.1f}%")
        st.progress(min(max(conf, 0.0), 1.0))

        st.markdown("**Top 3 predictions**")
        top3 = pd.DataFrame({
            "Species": [name_of(class_names[i], info) for i in order[:3]],
            "Confidence (%)": [round(float(probs[i]) * 100, 1) for i in order[:3]],
        })
        st.dataframe(top3, hide_index=True, use_container_width=True)

        if not is_unknown:
            rec = info.get(top_id)
            if rec:
                st.markdown(f"""<div class="card"><h4>Scientific classification</h4>
Family: <b>{e(rec['family'])}</b> &nbsp;|&nbsp; Order: <b>{e(rec['order'])}</b></div>
<div class="card"><h4>Key identification characteristics</h4>{e(rec['identification_characteristics'])}</div>
<div class="card"><h4>Habitat</h4>{e(rec['habitat'])}<h4 style="margin-top:.7rem">Distribution</h4>{e(rec['distribution'])}</div>""",
                            unsafe_allow_html=True)
                sims = [s for s in str(rec.get("similar_species", "")).split(";") if s.strip()]
                if sims:
                    st.markdown("**Visually similar species:** " + ", ".join(name_of(s.strip(), info) for s in sims))
                st.markdown(f'<p class="small">Information status: {e(rec.get("info_status", "unknown"))}</p>',
                            unsafe_allow_html=True)
            else:
                st.warning("No biological information is stored for this species in species_info.csv.")

st.divider()
st.caption("AI predictions are estimates and not a substitute for expert taxonomic identification.")
