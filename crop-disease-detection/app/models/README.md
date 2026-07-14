# Model files

`crop_disease_model.keras` is git-ignored because it's too large to commit
directly (see `.gitignore` and Step 9 of the notebook). `class_names.json`
is small and committed as-is — it's already the exact class list this model
was trained on.

To run the app locally:

1. Run `notebook/Crop_Disease_Detection.ipynb` through **Step 9** (or open
   it in Colab, which is what it was built for).
2. Download the resulting `crop_disease_model.keras` from Colab's file
   browser (or wherever Step 9 saved it).
3. Place it in this folder: `app/models/crop_disease_model.keras`.
4. Run the app: `streamlit run app/app.py`.

For sharing the trained model without asking people to retrain from
scratch, upload it somewhere that hosts large binaries well and link it
here — e.g. a Hugging Face Hub model repo, a GitHub Release asset, or Git
LFS — rather than committing the `.keras` file to the main repo.
