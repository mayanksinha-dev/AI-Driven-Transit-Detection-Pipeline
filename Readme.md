# AI-Driven Transit Detection Pipeline

An end-to-end preprocessing pipeline for detecting exoplanet transit candidates from raw TESS light curve FITS files. This pipeline transforms raw astronomical observations into structured machine learning datasets consisting of engineered transit features and phase-folded light curves.

---

## Overview

Pipeline-A performs automated preprocessing of raw TESS light curves and generates outputs suitable for downstream exoplanet classification models.

### Workflow

![Pipelinr Workflow](<workflow.png>)

---
git push -u origin main
## Features

### 1. Flux Extraction

Extracts the following data from TESS FITS files:

- TIME
- PDCSAP_FLUX
- QUALITY

---

### 2. Quality Filtering

Removes observations flagged by TESS quality indicators.

```python
QUALITY == 0
```

---

### 3. MAD Outlier Removal

Uses Median Absolute Deviation (MAD) to remove anomalous flux measurements.

```text
Threshold = 5 × MAD
```

---

### 4. Savitzky-Golay Detrending

Removes long-term stellar variability while preserving potential transit signals.

---

### 5. BLS Transit Search

Performs Box Least Squares periodogram analysis to estimate:

- Orbital Period
- Transit Duration
- Transit Depth
- Transit Time
- BLS Power
- Transit SNR

---

### 6. Statistical Feature Extraction

Generated features:

```text
mean_flux
std_flux
amplitude
skewness
kurtosis
mad
flux_q25
flux_q75
iqr
bls_period
bls_duration
bls_depth
bls_transit_time
transit_snr
bls_power
```

---

### 7. Phase Folding

Each candidate light curve is phase folded using the detected BLS period.

Outputs are normalized and resampled to a fixed length for downstream machine learning applications.

---

## Project Structure

```text
pipeline_a/

├── transformers/
│   ├── flux_extractor.py
│   ├── quality_filter.py
│   ├── mad_filter.py
│   ├── savgol_detrender.py
│   ├── bls_search.py
│   ├── feature_extractor.py
│   └── phase_folder.py
│
├── dataset_pipeline.py
├── run_pipeline.py
│
├── raw_tess/
│   └── *.fits
│
├── curves/
│   └── TIC_*.npz
│
└── features.csv
```

---

## Outputs

### features.csv

Tabular feature dataset for classical machine learning models.

Example:

```csv
mean_flux,std_flux,amplitude,...
1.0002,0.0041,0.023,...
```

---

### curves/

Each processed target is stored individually:

```text
curves/

├── TIC_280655495.npz
├── TIC_229650439.npz
├── TIC_229455001.npz
└── ...
```

Each file contains:

```python
phase
flux
period
duration
depth
transit_time
snr
power
```

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn astropy
```

---

## Usage

Place TESS FITS files inside:

```text
raw_tess/
```

Run:

```bash
python run_pipeline.py
```

Generated outputs:

```text
features.csv

curves/
    TIC_*.npz
```

---

## Example

Load a processed light curve:

```python
import numpy as np

data = np.load(
    "curves/TIC_280655495.npz"
)

phase = data["phase"]
flux = data["flux"]
```

Visualize:

```python
import matplotlib.pyplot as plt

plt.plot(
    phase,
    flux
)

plt.xlabel("Phase")
plt.ylabel("Normalized Flux")
plt.title("Phase Folded Light Curve")

plt.show()
```

---

## Applications

This pipeline can be used as a preprocessing stage for:

- XGBoost-based exoplanet classifiers
- Random Forest models
- Deep learning transit detectors
- CNN/BiLSTM architectures
- Ensemble exoplanet detection systems

---

## Data Source

**TESS (Transiting Exoplanet Survey Satellite)**

- Mission: NASA
- Archive: MAST
- Product: PDCSAP_FLUX Light Curves