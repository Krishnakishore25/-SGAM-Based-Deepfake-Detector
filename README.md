# -SGAM-Based-Deepfake-Detector
A Spatial Guided Deepfake Detection mechanism is developed to find the deepfake images by learning the spatial and semantic artifacts in the images. This is developed using CNN and Transformer.

## Step 1 — Install Required Libraries

Run the following command before executing the notebook.

### Install All Dependencies

```bash
pip install torch torchvision torchaudio timm numpy pandas matplotlib seaborn scikit-learn tqdm
```

---

# Step 2 — Verify GPU Availability

Run this code:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected output:

```text
True
Tesla T4
```

---

# Step 3 — Mount Google Drive (Google Colab Only)

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

# Step 4 — Create Working Directory

```python
import os

BASE_DIR = "/content/drive/MyDrive/Research_Implementation"

os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)
```

---

# Step 5 — Prepare Dataset Structure

Organize the dataset in the following format:

```text
dataset/
│
├── train/
│   ├── real/
│   └── fake/
│
├── validation/
│   ├── real/
│   └── fake/
│
└── test/
    ├── real/
    └── fake/
```

---

# Step 6 — Open the Notebook

Open the notebook:

```text
Deepfake_SpatialGuidedAttentionMechanism+DeiT+Classifier.ipynb
```

Then run all cells sequentially.

---

# Step 7 — Training Configuration

Example configuration:

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
```

---

# Step 8 — Recommended Imports

```python
import os
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
```

---

# Step 9 — Save Model

```python
torch.save(model.state_dict(), "best_model.pth")
```

---

# Step 10 — Load Model

```python
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```

---

# Step 11 — Generated Outputs

During training, the following outputs are automatically generated:

```text
Experiment 1/
│
├── best_model.pth
├── history.csv
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
└── metrics.txt
```

---

# Common Errors & Fixes

## CUDA Out of Memory

Reduce batch size:

```python
BATCH_SIZE = 16
```

or

```python
BATCH_SIZE = 8
```

---

## Module Not Found Error

Example:

```text
ModuleNotFoundError: No module named 'timm'
```

Install missing package:

```bash
pip install timm
```

---

## Dataset Not Found

Check dataset path carefully:

```python
train_dir = "/content/drive/MyDrive/Research_Implementation/dataset/train"
```

---

# Recommended Google Colab Runtime

Go to:

```text
Runtime → Change Runtime Type
```

Select:

```text
GPU
```

Recommended GPU Types:

- Tesla T4
- V100
- A100

---

# Recommended Versions

| Library | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| torchvision | 0.15+ |
| timm | 0.9+ |
| numpy | 1.24+ |
| pandas | 2.0+ |
| scikit-learn | 1.3+ |

---

# Recommended Training Flow

```text
1. Install dependencies
2. Mount Google Drive
3. Load dataset
4. Build CNN backbone
5. Build SGAM module
6. Build DeiT transformer
7. Train model
8. Save best model
9. Evaluate performance
10. Generate graphs & metrics
```

---

# Notes

- GPU training is highly recommended.
- Training on CPU will be very slow.
- Always save checkpoints to Google Drive.
- Use smaller batch sizes if GPU memory is limited.
- Keep dataset balanced between real and fake classes.
