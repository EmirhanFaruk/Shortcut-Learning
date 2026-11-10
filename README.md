# Skin Lesion Classification & Shortcut Learning

A deep learning study investigating **shortcut learning in medical image classification**. This project trains melanoma-vs-benign classifiers and investigates whether they learn meaningful lesion features or exploit spurious visual cues that correlate with the target label.

The project introduces a controlled visual shortcut, measures model reliance on it, investigates how shortcut adoption changes with cue predictiveness and size, and evaluates a mitigation strategy.

## Overview

Deep learning models can achieve high classification accuracy without learning the features we actually want them to learn. In medical imaging, this is particularly concerning: a model may rely on irrelevant visual artifacts rather than clinically meaningful characteristics.

This project investigates that problem through a controlled experiment using dermoscopic skin lesion images.

The study addresses four main questions:

1. How well do different CNN architectures classify melanoma versus benign lesions?
2. Can a model learn a spurious visual cue instead of the lesion itself?
3. How does shortcut reliance change as the cue becomes more predictive or visually prominent?
4. Can cue-randomizing augmentation reduce shortcut reliance?

## Key Features

- **Transfer learning:** ResNet-50 and EfficientNet-B0 pretrained on ImageNet.
- **Controlled shortcut injection:** Introduces an artificial visual cue correlated with the classification label.
- **Shortcut reliance measurement:** Evaluates models under clean, aligned, and swapped cue conditions.
- **Grad-CAM visualization:** Investigates which image regions influence model predictions.
- **Adoption phase diagram:** Studies shortcut reliance across different cue predictiveness and sizes.
- **Mitigation experiment:** Tests cue-randomizing augmentation against a standard training baseline.
- **Experiment management:** Timestamped logging, per-epoch checkpointing, automatic resume, and result reporting.

## Methodology

### 1. Clean classifier comparison

Two architectures are trained on the original lesion images without an injected shortcut:

- ResNet-50
- EfficientNet-B0

This establishes a baseline for melanoma-versus-benign classification.

### 2. Controlled shortcut learning

An artificial corner cue is introduced into the training images. The cue agrees with the target label 95% of the time.

The trained model is evaluated under three conditions:

| Test regime | Cue behaviour |
|---|---|
| None | No artificial cue |
| Aligned | Cue agrees with the label |
| Swapped | Cue indicates the opposite label |

Shortcut reliance is measured as:

**Reliance = Accuracy(aligned) − Accuracy(none)**

A larger value indicates greater dependence on the artificial cue.

The swapped condition provides an additional test: a model that relies heavily on the cue should perform substantially worse when the cue-label relationship is reversed.

### 3. Shortcut adoption study

The cue's predictiveness and visual size are varied independently.

The predictiveness experiment uses values of rho from 0.5 to 0.97.

The resulting phase diagrams investigate how shortcut reliance changes as the artificial cue becomes increasingly informative.

### 4. Mitigation

A cue-randomizing augmentation strategy is evaluated against a standard training baseline.

The objective is to reduce the relationship between the artificial cue and the classification label, encouraging the model to rely more on the lesion itself.

## Experimental Results

The following results are from an example experimental run.

### Clean classifier comparison

| Model | Accuracy | AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 0.878 | 0.915 | 0.723 | 0.908 |
| ResNet-50 | 0.866 | 0.893 | 0.676 | 0.903 |

EfficientNet-B0 achieved the highest AUC in this experiment.

### Shortcut injection

With cue predictiveness set to rho = 0.95:

| Test regime | Accuracy | AUC | Sensitivity |
|---|---:|---:|---:|
| None | 0.839 | 0.884 | 0.004 |
| Aligned | 0.997 | 0.997 | 0.983 |
| Swapped | 0.433 | 0.436 | 0.004 |

The results demonstrate substantial shortcut reliance.

Although the model achieved 99.7% accuracy when the cue was aligned with the label, malignant sensitivity dropped to 0.4% when the cue was removed.

This illustrates why high test accuracy alone does not guarantee that a medical image classifier has learned meaningful features.

### Shortcut adoption

The rho sweep produced the following results:

| Cue predictiveness (rho) | Reliance | Swap drop |
|---:|---:|---:|
| 0.50 | -0.002 | -0.005 |
| 0.60 | 0.015 | 0.032 |
| 0.70 | 0.011 | 0.067 |
| 0.80 | 0.028 | 0.103 |
| 0.90 | 0.096 | 0.323 |
| 0.97 | 0.156 | 0.573 |

Shortcut dependence increased substantially as cue predictiveness increased, particularly at higher rho values.

### Mitigation results

| Metric | Baseline | Mitigated |
|---|---:|---:|
| Reliance | 0.156 | -0.001 |
| Swap drop | 0.532 | -0.001 |
| Clean-test accuracy | 0.840 | 0.891 |
| Clean-test sensitivity | 0.004 | 0.546 |

Cue-randomizing augmentation substantially reduced measured shortcut reliance in this experiment.

These results are experimental observations from a controlled setting, not evidence of clinical performance or a general solution to unknown shortcuts.

## Visualizations

The project generates several diagnostic figures:

- Training curves and validation metrics.
- ROC curves and confusion matrices.
- Grad-CAM attention maps.
- Examples of images with and without injected cues.
- Accuracy comparisons across the three testing regimes.
- Shortcut adoption phase diagrams.
- Baseline versus mitigation comparisons.

The Grad-CAM comparison between clean and shortcut-trained models is particularly useful for examining changes in model attention.

## Dataset

This project uses the **HAM10000** dermoscopic image dataset.

Dataset source: [HAM10000 – Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)

The dataset is not included in this repository.

Download the images and metadata separately, following the dataset's licensing and usage conditions.

Expected directory structure:

```text
HAM10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

The preparation script maps the original diagnostic categories into two classes:

- **Malignant:** mel, bcc, akiec
- **Benign:** nv, bkl, df, vasc

Dataset splitting is performed by lesion ID rather than individual image to avoid images of the same lesion appearing across training and testing sets.

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.x
- NVIDIA GPU recommended for full experiments (CPU supported for smoke testing)

### Quick start

Clone the repository:

```bash
git clone https://github.com/EmirhanFaruk/Shortcut-Learning.git
cd Shortcut-Learning
```

For ssh:

```bash
git clone git@github.com:EmirhanFaruk/Shortcut-Learning.git
cd Shortcut-Learning
```

Run the pipeline:

```bash
chmod +x run_all.sh
./run_all.sh
```

The script creates a virtual environment, installs dependencies, prepares the dataset when necessary, and executes the pipeline.

### Smoke test

Before running the full experiments, test the pipeline using synthetic data:

```bash
./run_all.sh --smoke
```

This does not require HAM10000.

### Additional execution options

Run fewer epochs and skip the sweep:

```bash
./run_all.sh --epochs 5 --skip-sweep
```

Run individual stages:

```bash
# Classifier comparison and shortcut experiment
python3 src/run_experiment.py --epochs 8

# Shortcut adoption study
python3 src/sweep.py --vary rho
python3 src/sweep.py --vary saliency

# Mitigation experiment
python3 src/mitigation.py

# Generate the combined report
python3 src/report.py
```

For additional options:

```bash
./run_all.sh --help
```

## Output Structure

Running the pipeline generates the following directories:

```text
outputs/
├── models/
├── visualizations/
├── sweep/
├── SUMMARY.md
└── REPORT.md

logs/
```

The outputs include:

- Trained model checkpoints (`.pth`).
- Evaluation metrics (`results.json`).
- Grad-CAM visualizations and diagnostic plots.
- Sweep results and phase diagrams.
- Automatically generated experiment summaries.
- Timestamped execution logs.

Generated datasets, checkpoints, logs, and visualizations are not required to run the project and are kept separate from the source code.

## Reproducibility and Experiment Management

The pipeline includes:

- Timestamped logging.
- Per-epoch checkpointing.
- Automatic resumption after interrupted training.
- Per-stage completion tracking.
- Automatic report generation.
- Experiment result summaries.

These features make it easier to reproduce experiments and continue longer training runs without restarting completed stages.

## Limitations

This project is a controlled research and educational study, not a clinically validated diagnostic system.

Important limitations include:

- The injected shortcut is artificial and explicitly controlled.
- The mitigation strategy assumes the shortcut is already known.
- Results from a single experimental run do not establish statistical reproducibility across random seeds.
- Performance on HAM10000 does not establish generalization to other datasets or clinical environments.
- Grad-CAM provides an approximate visualization of model attribution and does not prove that a model uses clinically meaningful reasoning.

Further work could investigate multiple random seeds, additional datasets, more realistic spurious correlations, and methods for detecting unknown shortcuts.

## References

- Winkler et al. (2019). Research on confounding visual artifacts in deep learning for dermoscopic image classification. *JAMA Dermatology*.
- Geirhos et al. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*.
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. *Scientific Data*.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).

## Disclaimer

This project is intended for research and educational purposes only. It is not intended for clinical diagnosis, medical decision-making, or replacement of professional medical evaluation.