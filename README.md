# AquaPorichoy

**Bangladesh's aquatic species identifier — snap a photo, know your species.**
("AquaPorichoy" = "aquatic identity"; it covers fish plus the other aquatic
species of Bangladesh — dolphins, crustaceans and mollusks too.)
*(formerly Bangladesh Fish Species Identification AI)*

Upload a photo of a fish or other aquatic species -> the app suggests the species (common +
scientific name, family, order, habitat, distribution, identification characters, confidence,
similar species) and warns when it cannot identify the animal reliably.

Starter species (8): rohu, katla, mrigal, grass_carp, silver_carp, common_carp, nile_tilapia, hilsa.

## Folder structure
```
fish-identification-ai/
  dataset_raw/            <- YOU add images here: one folder per species (not uploaded to GitHub)
  models/                 <- fish_model.keras + class_names.json go here after training
  notebooks/fish_training_colab.ipynb
  app.py                  <- Streamlit web app
  train.py                <- training + evaluation
  prepare_dataset.py      <- train/val/test split without leakage
  make_species_csv.py     <- regenerates species_info.csv (optional)
  species_info.csv        <- biological database (DRAFT: verify!)
  requirements.txt
  .streamlit/config.toml  <- blue/cyan theme
```

## Step 1 - Collect images (see dataset links in the chat reply)
1. Create `dataset_raw/rohu`, `dataset_raw/katla`, ... Folder names must equal `class_id` in `species_info.csv`.
2. Download a dataset, keep only the species you need, put them in the matching folders.
3. Use ORIGINAL photos only (delete pre-augmented copies and background-removed copies).
4. Aim for 200+ images per species. Check every dataset's licence and cite it in your report.
5. Mix sources: if one species comes only from one dataset, the model may learn the dataset's look
   (background, lighting) instead of the fish. Add your own market/pond photos with varied backgrounds.
6. Recommended: add a folder `unknown_other` with 200+ photos of other fish and non-fish objects
   so the app can say "unknown".

## Step 2 - Train in Google Colab
1. Make folder `fish-identification-ai` in Google Drive (My Drive). Upload `train.py`,
   `prepare_dataset.py`, and the `dataset_raw` folder.
2. Open https://colab.research.google.com -> File -> Upload notebook -> `notebooks/fish_training_colab.ipynb`.
3. Runtime -> Change runtime type -> T4 GPU. Run cells 1-6 in order.
4. Note the TensorFlow version printed in Cell 1. If it is not 2.19.x, edit `tensorflow-cpu==...` in `requirements.txt`.

## Step 3 - Put the trained model in the project
Copy `fish_model.keras` and `class_names.json` into `models/`. (Model is ~10-15 MB, fine for GitHub.)

## Step 4 - Test locally (optional)
`pip install -r requirements.txt` then `streamlit run app.py`

## Step 5 - Upload to GitHub
1. Create an account at https://github.com, then New repository -> name `fish-identification-ai`, **Public**.
2. Click "uploading an existing file", drag in everything EXCEPT `dataset_raw/` (including the hidden
   `.streamlit` folder and the `models` folder), then Commit.

## Step 6 - Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub (authorize access).
2. Create app -> "Deploy a public app from GitHub".
3. Repository: `YOUR-USERNAME/fish-identification-ai` | Branch: `main` | Main file path: `app.py`.
4. Advanced settings -> Python version **3.11** (TensorFlow 2.19 needs 3.9-3.12).
5. Deploy. The first build takes a few minutes. Your public URL looks like `https://<name>.streamlit.app`.

## Step 7 - Test the deployed app
Test with (a) a clear photo of each species, (b) a species NOT in the list, (c) a non-fish photo,
(d) a blurry photo, (e) a phone photo. Use photos NOT used in training. Wrong/over-confident results
on (b)-(d) mean you need an `unknown_other` class and more varied images.

## Adding a new species
1. Add `dataset_raw/<new_class_id>/` with images. 2. Add a row to `species_info.csv` with the same `class_id`.
3. Re-run Colab Cells 3-6. 4. Replace the two files in `models/` on GitHub; the app redeploys.

## Open data version

The app now prefers an open-data species table over the hand-written CSV.
If `data/species_master.csv` exists it is used; otherwise the app falls back
to `species_info.csv`, so nothing breaks if the open data has not been built.

**Sources**
- [GBIF](https://www.gbif.org) — species matching / taxonomy (kingdom → genus),
  occurrence counts for Bangladesh and worldwide, and occurrences with photos.
  Species pages look like `https://www.gbif.org/species/<key>`.
- [iNaturalist](https://www.inaturalist.org) — taxon class (from taxon
  ancestors), sample photos, and research-grade observation photos.
- [FishBase](https://www.fishbase.se) — linked per fish species as
  `https://www.fishbase.se/summary/<Genus-species>.html` for deeper reading
  (linked, not scraped). Cite FishBase when you use its data.
- [SeaLifeBase](https://www.sealifebase.se) — the FishBase sister database for
  non-fish aquatic species; linked the same way for dolphins, prawns, crabs
  and mollusks (`fishbase_url` stays blank for those species).

The catalogue now holds **58 species** (v1.1 expansion): **40 fish,
8 crustaceans, 7 mollusks and 3 aquatic mammals** — all found in Bangladesh.
It is **not fish-only**: each row has a `group` (Fish, Aquatic mammal,
Crustacean, Mollusk, ...). The seed list includes Bangladesh-relevant
non-fish species — Ganges River Dolphin, Irrawaddy Dolphin, Finless
Porpoise, Giant River Prawn (Golda), Bagda Chingri, Indian White Prawn,
Mud Crab, Orange Mud Crab, Three-spot Swimming Crab, Green Mussel, Blood
Clam (Teppa), Apple Snail and Indian Ocean Squid — enriched through exactly
the same GBIF/iNaturalist pipeline.

**Build / refresh the data**
```
python3 data/build_species_data.py
```
Reads `data/seed_species.csv` (the species list), fetches GBIF + iNaturalist
data with a file cache in `data/cache/`, merges the existing `species_info.csv`
habitat / distribution / identification text as manual overrides, and writes
`data/species_master.csv` with the full classification, habitat, distribution,
Bangladesh + world occurrence counts (biodiversity), photo-record counts, a
sample photo, and source links per species. Two model classes
(`freshwater_shark`, `long_whiskered_catfish`) have only *tentative*
scientific-name mappings (Wallago attu / Sperata aor) — they are flagged
NEEDS VERIFICATION in the output; check them against the training photos
before publishing.

**Download training photos (openly licensed only)**
```
python3 data/download_photos.py --per-species 100
python3 data/download_photos.py --per-species 20 --species rohu,hilsa
```
Downloads research-grade iNaturalist photos and GBIF image occurrences into
`dataset_open/<class_id>/` (git-ignored). Photos with no licence or an
"all rights reserved" licence are skipped; `dataset_open/photo_metadata.csv`
records the licence, photographer/observer and source page for every photo,
so attribution travels with the dataset. Note that many photos are CC BY-NC —
fine for a non-commercial / educational project, but check before any
commercial use.

**What the app shows now**
The app has a single input — one chat-style command bar where you attach a
photo and can add an optional written note in the same bar, then press
Identify once. The result profile shows the species name with its full
classification (kingdom → genus), habitat, distribution, GBIF biodiversity
counts and source links from the master table, with your note shown
alongside as context (the note does not change the prediction). There is
deliberately no name search or catalogue browser — identification is
photo-only. (Hilsa and Common Carp have full data rows ready for when the
model is retrained to predict them.) The interface uses a light animated
river-gradient theme (flowing gradient banner and command bar, fading
result cards; animations switch off for reduced-motion users).

**Honest limitation:** the catalogue can cover many species long before the
classifier can. Expanding *identification* beyond the current 8 model classes
requires retraining: download photos with the script above, put them in the
dataset folders, and follow the existing `train.py` / Colab flow (Steps 2–3),
then replace `models/fish_model.keras` and `models/class_names.json`.
Catalogue/data coverage ≠ classifier coverage.

### Gate plan (aquatic vs human / other living / non-living)

Identification is designed as two stages:
1. **Gate** — aquatic animal vs `human` vs `other_living` vs
   `non_living`/unknown.
2. **Species classifier** — only runs on photos that pass the gate.

There is **no trained gate model yet**. Until one is trained, the app uses an
interim gate: the species model's `unknown_other` class plus its confidence
thresholds, and the "cannot identify" message now says explicitly that the
photo may be a human, another animal, a plant, or an object. That is a
fallback, not a real gate — do not claim otherwise.

Building the gate dataset:
```
python3 data/download_negatives.py            # target: 200+ per gate class
python3 data/download_negatives.py --per-class 20 --classes other_living,aquatic
```
- Target **200+ images per gate class** (`aquatic`, `human`, `other_living`,
  `non_living`), mirroring this repo's existing `unknown_other` guidance.
- `other_living` (birds, insects, plants, fungi) and `aquatic` (seed species)
  are downloaded from iNaturalist research-grade, licence-required photos,
  with attribution in `dataset_gate/photo_metadata.csv`.
- `human` and `non_living` are **not** downloaded by the script: there is no
  clean, licence-safe API source for them, and face datasets must never be
  used. `dataset_gate/SOURCES.md` (written by the script) documents the
  alternatives — COCO (`person` + object categories) and Open Images
  (`Person` + object classes) — to download manually under their terms.
- The gate, like the expanded species list, only starts identifying anything
  after retraining (`train.py`, Colab). Until then it is scaffolding +
  dataset structure, honestly labelled.

## Honest limitations
Accuracy depends on your data. The model only knows its trained species. Biological info in
`species_info.csv` is a draft; verify with FishBase and Bangladesh fisheries references first.
