# Bangladesh Fish Species Identification AI

Upload a fish photo -> the app suggests the species (common + scientific name, family, order,
habitat, distribution, identification characters, confidence, similar species) and warns when it
cannot identify the fish reliably.

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

## Honest limitations
Accuracy depends on your data. The model only knows its trained species. Biological info in
`species_info.csv` is a draft; verify with FishBase and Bangladesh fisheries references first.
