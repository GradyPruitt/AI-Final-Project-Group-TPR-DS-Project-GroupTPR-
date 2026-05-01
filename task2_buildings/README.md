# Task 2 — DoriaNET Hurricane Building Damage Classification

Final project Task 2. Three deep computer vision architectures (ResNet-50,
EfficientNet-B0, ViT-B/16) compete on the DoriaNET dataset of post-Hurricane
Dorian aerial imagery, classifying each building into 6 damage levels (0–5).

## Status / what's left to do

**Done:**
- Preprocessing: parses 271 frames + 1458 buildings, computes bounding boxes,
  builds frame-level train/val/test splits.
- PyTorch Dataset that crops each building from its frame using mask bbox,
  applies ImageNet-style transforms.
- Three model classes with ImageNet-pretrained backbones and replaced heads.
- Training loop with class-weighted cross-entropy (class imbalance handling),
  early stopping, full test evaluation.
- Comparison sweep script that runs all three and produces a report-ready table.
- Standard metrics: top-1 accuracy, within-±1 accuracy, macro F1, MAE in
  level units, per-class accuracy.

**Left to do:**
- Run the comparison sweep on a real machine (the sandbox can't download
  pretrained weights).
- Possibly: implement EMD² loss as in the reference paper (currently using
  weighted cross-entropy — class weights handle imbalance, but EMD² would
  better reflect the ordinal nature of the labels). Hook for this is in
  `train.py` — add another option to the loss section.
- EDA notebook for Task 2 (sketch in the project plan; not built yet).
- Wire up presentation plots / figures.

## Reference

Cheng, C.-S., Behzadan, A. H., Noshadravan, A. (2021).
*Deep learning for post-hurricane aerial damage assessment of buildings.*
Computer-Aided Civil and Infrastructure Engineering, 36(6), 695–710.
DOI: 10.1111/mice.12658

PDF in the dataset folder. Their approach uses a two-stage SPDA model
(localization + classification). We simplify to just the classification
stage by using the ground-truth masks directly. Their reported test
accuracy on Dataset 1: **61% top-1, 90% within-±1**. That's the bar to
clear.

## Project layout

```
task2_buildings/
├── src/
│   ├── preprocessing.py      JSON+masks → tidy manifest CSV
│   ├── dataset.py            PyTorch Dataset (crop building from frame+mask)
│   ├── models.py             ResNet50, EfficientNet-B0, ViT-B/16 classifiers
│   ├── metrics.py            accuracy, within-±1, macro F1, MAE, confusion matrix
│   └── train.py              training loop + test evaluation
├── scripts/
│   ├── preprocess.py         CLI: raw dir → manifest CSV
│   ├── train_baseline.py     CLI: train a single model
│   └── run_comparison.py     CLI: train all three, build comparison table
├── data/
│   └── processed/            manifest CSV lands here
├── notebooks/                (TODO: 01_eda.ipynb)
├── runs/                     created per training run
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate              # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Step 1 — Build the manifest

Point at the unzipped DoriaNET project folder (the one containing FRAME/, MASK/, JSON/):

```bash
python scripts/preprocess.py --raw-dir ~/Downloads/DoriaNETproject
```

This produces `data/processed/buildings_manifest.csv` with one row per
building: paths, damage level, mask bounding box, train/val/test split.

Train/val/test split is at the **frame level** so no frame appears in two
splits. Default proportions: 70% train / 15% val / 15% test.

## Step 2 — Train one model (sanity check)

```bash
python scripts/train_baseline.py --model resnet50 --epochs 20
```

The first run downloads ImageNet pretrained weights (~100MB for ResNet-50,
~340MB for ViT). Training ~20 epochs takes maybe 10-15 minutes on a laptop
CPU, much faster on GPU.

## Step 3 — Run the full comparison

```bash
python scripts/run_comparison.py --epochs 20
```

Trains all three models sequentially. Output `runs/comparison_<ts>/comparison_table.md`
is the report-ready table.

## Notes on design choices

**Frame-level splitting.** Buildings within the same frame are visually
correlated. Splitting at the building level would leak training context
into val/test through neighbors in the same frame. We split frames first,
then propagate the assignment to all buildings in that frame.

**Class weights vs EMD² loss.** The dataset is moderately imbalanced
(Level 5 has 108 examples, Level 3 has 464). We use inverse-frequency
weighted cross-entropy as a simple imbalance fix. The reference paper uses
EMD² loss, which additionally accounts for the ordinal nature of the
labels (predicting Level 4 when truth is Level 3 is a smaller error than
predicting Level 0 when truth is Level 5). Implementing EMD² is on the
TODO list — it's a few lines, see paper Section 2.3.

**Why these three models?** ResNet-50 is the classical CNN baseline.
EfficientNet-B0 is a modern CNN that's much smaller (5M vs 25M params)
and often matches ResNet's accuracy via better architecture design. ViT-B/16
is a non-convolutional alternative — it treats the image as a sequence of
16×16 patches and processes them with self-attention. CNN-vs-Transformer
on small data is an interesting comparison; ViTs often need more data than
CNNs to generalize well, which is informative for our use case.

**Why simplify to single-stage classification?** The reference paper uses a
two-stage SPDA: Mask R-CNN for localization, then MobileNet for classification.
Implementing both and fairly comparing three localizers + three classifiers
is out of scope for our timeline. We use the dataset's ground-truth masks
to skip localization, focusing the comparison on the classification step
that's what the assignment requires (three deep CV models).
