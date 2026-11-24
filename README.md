# Chest X-ray Classification: COVID-19 · Viral Pneumonia · Normal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LittleIronclad/covid-xray-classification/blob/main/notebooks/covid_xray_classification.ipynb)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

Three-class classification of chest radiographs, comparing models of increasing capacity — a pixel-space **k-NN** baseline, a **custom CNN trained from scratch**, and a fine-tuned **ResNet-50** — with **Grad-CAM** to check what the networks actually look at.

## Highlights

- **Leakage-free evaluation.** The official test split is never augmented or used for model selection; a stratified validation split is taken *before* augmentation.
- **Class imbalance handled in training, not in the test set.** A weighted sampler with on-the-fly augmentation in training; balanced accuracy and macro-F1 for reporting.
- **Test-time augmentation (TTA).** Predictions are averaged over flipped and rotated views of each test image.
- **Interpretability.** Grad-CAM maps for both networks, used to check whether attention falls on the lung fields or on dataset artefacts.
- **Reproducible.** Fixed seeds, deterministic cuDNN, a single notebook that runs top-to-bottom on a free Colab GPU in about 10 minutes.

## Models

| Model | Description |
|---|---|
| k-NN | 64×64 grayscale pixels, `k` chosen on validation |
| **Custom CNN** | 4 × (7×7 conv + spectral norm + ReLU + max-pool) → FC 256 → 84 → 3; ~234k parameters, SGD with momentum |
| ResNet-50 | ImageNet-pretrained, final layer replaced, fully fine-tuned with Adam (lr 1e-4) |

Spectral normalization bounds the Lipschitz constant of each convolution. With only ~200 training images it noticeably stabilised training of the from-scratch network.

## Results

Untouched test set: 66 images (26 COVID, 20 Normal, 20 Viral Pneumonia).

> Results are being regenerated with the leakage-free pipeline and will be added here.

## Repository structure

```
notebooks/covid_xray_classification.ipynb   end-to-end pipeline: EDA → models → TTA → Grad-CAM → results
requirements.txt
```

## How to run

**Colab (recommended):** click the badge above → *Runtime → Change runtime type → GPU* → *Run all*. The dataset is downloaded automatically via `kagglehub`.

**Locally:**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/covid_xray_classification.ipynb
```
To use a local copy of the data, unzip it so that `Covid19-dataset/` sits in the repository root (or set `DATA_DIR`). On Apple Silicon Macs the notebook uses the GPU via MPS automatically.

## Data

[COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) by Pranav Raikokte (Kaggle). It has 317 chest X-rays, pre-split into train (251) and test (66). The data is not redistributed in this repository.

## Limitations

- **Small test set.** With 66 images, a single error changes accuracy by about 1.5 points, so small differences between models are within noise.
- **Shortcut risk.** Classes come from different public sources, and image resolution and appearance differ systematically by class. High scores may partly reflect image *source* rather than pathology. Grad-CAM helps to detect this, but a real claim would require external validation on data from another institution.
- **Horizontal flips** are anatomically unrealistic for chest radiographs. They are used only as a light regulariser.
- This is a research and education project, **not a diagnostic tool**.

## Context & credits

Two-person team project for the Scientific Research Seminar *"Artificial Neural Networks and Machine Learning"* at HSE University (2025).

- **Alexey Potemkin**: designed, implemented and trained the custom CNN; co-developed data exploration and preprocessing.
- **Teammate**: k-NN baseline and ResNet-50 transfer learning; co-developed data exploration and preprocessing.

This repository is a cleaned-up, English-language version of the original course notebook. The main change is the evaluation protocol: no augmentation of the test set, a validation split taken before augmentation, balanced metrics, and TTA.

## License

MIT. See [LICENSE](LICENSE).
