# Smart Product Pricing Challenge — AmazonMLChallenge2k25

## Overview

This repository contains the code and instructions for the Smart Product Pricing Challenge (ML Challenge 2025). The goal is to build a model that predicts optimal product prices using the provided product text and images. The focus is on creating a reproducible, well-documented solution that uses only the provided training data (external price lookup is strictly prohibited).

This README explains the dataset, required output format, evaluation metric (SMAPE), recommended modeling approaches, and submission instructions.

## Repository structure

- `dataset/` — expected location for `train.csv`, `test.csv`, and sample files
- `src/` — code and utilities (e.g., `src/utils.py` for image downloading)
- `sample_code.py` — optional example that shows how to produce a submission file
- `Documentation_template.md` — template for the required 1-page methodology document
- `README.md` — this file

If any of these files are missing in your local copy, place the dataset files under `dataset/` and the code under `src/` as described above.

## Problem statement (short)

Given product records containing a `sample_id`, `catalog_content` (text that includes title, description, and item pack quantity), and an `image_link`, predict the `price` for each product in the test set. Training data includes the target `price`; the test set does not.

Required output: a CSV file with two columns, `sample_id` and `price` (floating point), matching the ordering and IDs of `dataset/test.csv`.

## Data description

Each row in the dataset contains:

1. `sample_id`: Unique identifier for the sample
2. `catalog_content`: Text field containing title, description and Item Pack Quantity (IPQ) concatenated
3. `image_link`: URL for a product image
4. `price`: Target variable (only present in `train.csv`)

Dataset sizes (provided by the challenge):

- Training: 75,000 samples (with prices)
- Test: 75,000 samples (no prices)

Example files:

- `dataset/sample_test.csv` — sample input format for the test set
- `dataset/sample_test_out.csv` — sample output format for submission

Note: Use `src/utils.py` to download images from `image_link`. Downloads may be throttled; implement retries as needed.

## Output format and submission

Your final submission must be a CSV file named `test_out.csv` (or as required by the portal) with exactly two columns and a header row:

```
sample_id,price
<id_1>,<predicted_price_1>
<id_2>,<predicted_price_2>
...
```

- There must be one prediction per sample in the test set. Missing or extra rows will cause evaluation errors.
- Predicted `price` values must be positive floats.

## Evaluation metric — SMAPE

Submissions are evaluated using Symmetric Mean Absolute Percentage Error (SMAPE). Lower SMAPE is better. The formula is:

$$
	ext{SMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|p_i - a_i|}{\frac{|a_i| + |p_i|}{2}}
$$

where $p_i$ is the predicted price and $a_i$ is the actual price for sample $i$. SMAPE is bounded between 0% and 200%.

Example: actual $= 100$, predicted $= 120$.

$$
	ext{SMAPE} = \frac{|100 - 120|}{( |100| + |120| ) / 2} = \frac{20}{110} \approx 0.1818 \; (18.18\%)
$$

## Constraints and rules

- Do NOT perform any external price lookup or use external price data. Web scraping or API lookups for prices are strictly prohibited and will lead to disqualification.
- Final models and code should be licensed under MIT or Apache 2.0.
- Model parameter count should be up to 8 billion parameters.

## Getting started (recommended environment)

1. Create a Python environment (3.8+ recommended). Example using venv:

```
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # WSL / Unix
pip install -r requirements.txt
```

2. Place the dataset CSVs under `dataset/` (`train.csv`, `test.csv`).

3. Inspect `sample_code.py` for a minimal example that creates a submission with constant or heuristic predictions.

## Using `src/utils.py` to download images

The repository includes `src/utils.py` which provides a `download_images` helper to fetch images from `image_link`. The helper implements retry logic to handle throttling. Example usage (see `src/test.ipynb` for a notebook example):

```python
from src.utils import download_images

# dataset is a DataFrame with an `image_link` column
download_images(df['image_link'], dest_dir='images', max_workers=8)
```

Tip: Download images once and cache them. Image processing (resizing, augmentation, feature extraction) should be deterministic and reproducible.

## Modeling guidance and suggestions

The relationship between product attributes and price is complex. Consider a multi-modal approach that leverages both text and image data.

1. Text features
	- Clean and tokenize `catalog_content`. Extract structured signals like item pack quantity (IPQ), numeric tokens, and units.
	- Apply TF-IDF, bag-of-words, or pre-trained transformer embeddings (e.g., DistilBERT, or other models within the allowed size).
	- Consider special handling for brand names, units (e.g., "pack of 6"), and words indicating quality or rarity.

2. Image features
	- Use a lightweight CNN or a pre-trained visual backbone (within model-size constraints) to extract image embeddings.
	- Simple approaches: EfficientNet-lite, MobileNet, or smaller ViT variants.
	- If image download is incomplete, ensure your pipeline degrades gracefully and uses text-only features.

3. Tabular features
	- Extract numeric features such as inferred quantity, presence of keywords, text length, capitalization signals.
	- One-hot encode categorical features where appropriate.

4. Model choices
	- Gradient-boosted trees (e.g., LightGBM, XGBoost) on engineered features often provide strong baselines.
	- Neural approaches: combine text embeddings (transformer) and image embeddings in a small dense head.
	- Ensembles: average or stack multiple model predictions (GBM + NN) for robustness.

5. Loss and target transformation
	- Price distributions are often right-skewed. Log-transform the target and predict log(price) with an inverse transform at inference to stabilize training. When using SMAPE, be careful: SMAPE operates on the original scale, so evaluate on the original scale.
	- Clip unrealistic predictions (e.g., enforce price >= tiny_positive_value).

6. Outlier handling
	- Inspect extreme prices and rare item types. Consider capping or separate models for premium categories.

7. Cross-validation
	- Use k-fold CV (e.g., 5 folds) and keep test-time ensembling consistent with CV splits.
