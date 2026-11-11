# Multimodal Emotion Fusion

A Python implementation of decision-level multimodal emotion classification across four modalities (`EEG`, `Image`, `Speech`, and `Video`) and three emotion classes (`Negative`, `Neutral`, and `Positive`). The project includes modality-specific model architectures, feature extraction utilities, a PyTorch training helper, Dempster-Shafer evidence combination, and a reliability-weighted Mamdani fuzzy inference fusion engine with rule-level explainability.

## Project Structure

```text
emotion_fusion_project/
├── branches.py                 # Modality models, feature extractors, and inference helpers
├── fuzzy_fusion.py             # Mamdani fuzzy fusion, reliability weighting, and Dempster-Shafer fusion
├── main.py                     # Simulation benchmark and demo sample runner
├── train.py                    # Macro-F1 metric and PyTorch branch training loop
├── requirements.txt            # Python package dependencies
└── outputs/
    └── simulation_output.txt   # Generated simulation report and demo explanation
```

## Components

### 1. Decision Fusion (`fuzzy_fusion.py`)
- **`MamdaniFusion`**: Reliability-scaled Mamdani fuzzy inference system using triangular input membership functions (`L`, `M`, `H`) and four output membership sets (`Low`, `Medium`, `High`, `VeryHigh`). Evaluates 15 rules (`R1`–`R15`) covering multi-modality consensus, missing-modality scenarios, and conflict handling, followed by centroid defuzzification and a weighted-mean fallback when no rule fires.
- **`MamdaniFusion.explain`**: Returns the active rules and firing strengths per emotion class for a single sample.
- **`reliability_from_validation`**: Normalizes per-modality validation macro-F1 scores relative to chance (`1/3`) into `[0, 1]` reliability weights.
- **`ds_fuse`**: Implements both undiscounted and reliability-discounted Dempster-Shafer evidence combination across available modalities.

### 2. Modality Branches and Feature Extraction (`branches.py`)
- **`EEGFNN`**: Feed-forward neural network taking 750-dimensional FFT features (`in_features=750`).
- **`FaceCNN`**: Convolutional network with `BatchNorm2d` and `DropBlock2d` for grayscale facial images.
- **`SpeechCNN`**: Convolutional network with adaptive average pooling for speech MFCC feature maps.
- **`VideoBranch`**: Extracts mean/std visual embeddings using a frozen ImageNet-pretrained `ResNet18` backbone and classifies with `StandardScaler` + RBF `SVC` (`probability=True`), alongside a static `audio_features` MFCC statistics helper.
- **Preprocessing utilities**: `eeg_fft_features` (4th-order Butterworth 1–45 Hz bandpass filter, Hann window, rFFT magnitude) and `speech_to_mfcc_map` (audio trimming, normalization, 40-band MFCC extraction).

### 3. Training Utilities (`train.py`)
- **`macro_f1`**: Computes unweighted macro-F1 across `k=3` classes.
- **`train_branch`**: Trains a PyTorch branch with Adam and inverse-frequency class-weighted `CrossEntropyLoss`, selects the checkpoint with the highest validation macro-F1, and writes epoch logs to `outputs/train_log.txt` by default (configurable via `log_path`).

### 4. Simulation Benchmark (`main.py`)
- **`run_simulation`**: Generates calibrated simulated posteriors across `seeds=5` random seeds (`2000` validation samples and `n_test=3000` test samples per seed) and evaluates five fusion methods (`Mean`, `W.mean`, `DS undisc`, `DS disc`, `Mamdani`) under three availability conditions (`All four`, `30% missing`, and `EEG missing`), followed by a single-sample rule-explanation demo.

## Layer-by-Layer Explanation

### A. Neural Network and Classifier Branches (`branches.py`)

Each modality branch follows a modular feature-to-logits (or feature-to-probability) structure:

#### 1. `EEGFNN` (EEG Feed-Forward Network)
Processes flattened EEG FFT power/magnitude features (default input dimension `in_features=750`, output `n_classes=3`):

| Layer | Configuration | Output Shape |
|-------|---------------|--------------|
| Input | Bandpass-filtered (1–45 Hz, 4th-order Butterworth), Hann-windowed rFFT magnitude | `(B, 750)` |
| Hidden Block 1 | `Linear(750, 256)` -> `ReLU` -> `Dropout(p=0.3)` | `(B, 256)` |
| Hidden Block 2 | `Linear(256, 128)` -> `ReLU` -> `Dropout(p=0.3)` | `(B, 128)` |
| Classifier | `Linear(128, 3)` | `(B, 3)` |

#### 2. `FaceCNN` (Facial Expression CNN)
Built from repeated `conv_bn_relu(cin, cout)` blocks (`Conv2d(kernel_size=3, padding=1)` -> `BatchNorm2d` -> `ReLU`). Based on the classifier input dimension (`9216 = 256 * 6 * 6`) after three `2x2` max-pooling stages, the network expects `(B, 1, 48, 48)` grayscale face inputs:

| Layer | Configuration | Output Shape (for `48x48` input) |
|-------|---------------|----------------------------------|
| Conv Block 1 | `conv_bn_relu(1, 64)` -> `conv_bn_relu(64, 64)` -> `MaxPool2d(2)` | `(B, 64, 24, 24)` |
| Conv Block 2 | `conv_bn_relu(64, 128)` -> `conv_bn_relu(128, 128)` -> `MaxPool2d(2)` | `(B, 128, 12, 12)` |
| Conv Block 3 | `conv_bn_relu(128, 256)` -> `conv_bn_relu(256, 256)` -> `DropBlock2d(p=0.1, block_size=3)` -> `MaxPool2d(2)` | `(B, 256, 6, 6)` |
| Flatten | `Flatten()` | `(B, 9216)` |
| Dense Hidden | `Linear(9216, 256)` -> `ReLU` -> `Dropout(p=0.5)` | `(B, 256)` |
| Classifier | `Linear(256, 3)` | `(B, 3)` |

#### 3. `SpeechCNN` (Speech MFCC CNN)
Processes normalized single-channel MFCC time-frequency maps from `speech_to_mfcc_map` (default `(B, 1, 40, 128)`):

| Layer | Configuration | Output Shape (for `40x128` input) |
|-------|---------------|-----------------------------------|
| Conv Block 1 | `conv_bn_relu(1, 32)` -> `MaxPool2d(2)` | `(B, 32, 20, 64)` |
| Conv Block 2 | `conv_bn_relu(32, 64)` -> `MaxPool2d(2)` | `(B, 64, 10, 32)` |
| Conv Block 3 | `conv_bn_relu(64, 128)` -> `MaxPool2d(2)` | `(B, 128, 5, 16)` |
| Global Pooling | `AdaptiveAvgPool2d(1)` -> `Flatten()` | `(B, 128)` |
| Dense Hidden | `Linear(128, 128)` -> `ReLU` -> `Dropout(p=0.4)` | `(B, 128)` |
| Classifier | `Linear(128, 3)` | `(B, 3)` |

#### 4. `VideoBranch` (Frozen ResNet18 + RBF SVM)
- **Visual Backbone**: `ResNet18` (`IMAGENET1K_V1` weights) with its final linear layer replaced by `nn.Identity()` and all parameters frozen (`requires_grad=False`), extracting a 512-dimensional embedding per video frame.
- **Temporal Aggregation (`visual_features`)**: Concatenates frame-wise mean and standard deviation (`[mean, std]`) into a 1024-dimensional video descriptor.
- **Audio Helper (`audio_features`)**: Extracts 40 MFCCs (`sr=22050`) and concatenates temporal mean and standard deviation into an 80-dimensional descriptor.
- **Classification Head (`clf`)**: `scikit-learn` pipeline combining `StandardScaler` with `SVC(kernel="rbf", C=10, probability=True, random_state=42)`.

---

### B. End-to-End Fusion Pipeline (`fuzzy_fusion.py` and `main.py`)

Data flows sequentially through five processing layers from raw/simulated posteriors to final class decisions:

1. **Input and Availability Layer**:
   - Accepts per-modality class posterior probabilities `P` of shape `(N, 4, 3)` (4 modalities by 3 classes) and a boolean availability mask `A` of shape `(N, 4)` indicating which modalities are present.
2. **Reliability Calibration Layer (`reliability_from_validation`)**:
   - Converts each modality's validation macro-F1 score into a normalized reliability weight `r` in `(0, 1]` by scaling excess performance over random chance (`1/3`): `q = max((val_macro_f1 - 1/3) / (2/3), 1e-6)`, followed by `q / q.max()`.
3. **Fuzzification Layer (`MamdaniFusion._alphas`)**:
   - Maps each posterior probability into three triangular fuzzy input sets:
     - `L`: `(0.0, 0.0, 0.5)` (Low)
     - `M`: `(0.35, 0.55, 0.75)` (Medium, scaled by modality reliability `r`)
     - `H`: `(0.6, 1.0, 1.0)` (High, scaled by modality reliability `r`)
   - Unavailable modalities (`A == False`) are masked to `0.0`.
4. **Fuzzy Rule Evaluation and Output Aggregation Layer (`R1`–`R15`)**:
   - Evaluates 15 Mamdani rules per class using `min`/`max` fuzzy operators across multi-modality agreement (`R1`, `R6`, `R7`, `R8`), EEG-anchored consensus (`R2`, `R3`, `R4`), universal low support (`R5`), modality conflict (`R9`, `R10`), missing-modality rules (`R11`–`R14`), and single-modality fallback (`R15`).
   - Takes the maximum firing strength for each consequent level (`Low`, `Medium`, `High`, `VeryHigh`) and clips the corresponding triangular output membership functions defined on a 201-point grid over `[0, 1]`.
   - Takes the upper envelope (`max` across consequents) to form the aggregated fuzzy output curve `B` per class.
5. **Defuzzification, Tie-Breaking, and Fallback Layer (`MamdaniFusion.predict`)**:
   - Computes the centroid score `S_c = sum(B * s) / sum(B)` for each class `c`.
   - If the maximum aggregated area across classes is `<= 1e-9` (no rule fired strongly enough), falls back to the reliability-weighted mean of available posteriors.
   - Resolves near-ties within `tie_eps=1e-3` by choosing the tied class with the largest aggregated area, and outputs `(pred, support, S, fallback)` where `support` normalizes `S` across classes.
   - `MamdaniFusion.explain` exposes the exact non-zero rule firing strengths (`> 1e-6`) for each class.

## Installation

Install the required dependencies listed in `requirements.txt` (`numpy`, `torch`, `torchvision`, `scikit-learn`, `scipy`, `librosa`):

```bash
python -m venv .venv
```

Activate the virtual environment and install packages:

- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Usage

### Run the Simulation and Demo

```bash
python main.py
```

This prints the benchmark summary and demo sample output to the terminal and writes the full report (including mean, standard deviation, per-seed accuracies, input posteriors, and activated rules) to `outputs/simulation_output.txt`.

### Latest Simulation Results (`seeds=5`, `n_test=3000`)

| Scenario    | Mean (%)        | W.mean (%)      | DS undisc (%)   | DS disc (%)     | Mamdani (%)     |
|-------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| All four    | 97.47 (std 0.17)| 97.76 (std 0.14)| 98.47 (std 0.08)| 97.10 (std 0.13)| 94.16 (std 0.46)|
| 30% missing | 92.64 (std 0.20)| 92.63 (std 0.18)| 93.39 (std 0.24)| 92.27 (std 0.32)| 89.93 (std 0.37)|
| EEG missing | 89.45 (std 0.51)| 89.12 (std 0.55)| 90.31 (std 0.47)| 89.28 (std 0.54)| 86.91 (std 0.79)|

### Demo Sample Output

For modality posteriors with reliability `[1.0, 0.739, 0.689, 0.445]`:

- **Prediction**: `Positive`
- **Normalized support**: `[0.0, 0.188, 0.812]`
- **Defuzzified score (`S_c`)**: `[0.0, 0.146, 0.631]`
- **Activated rules**:
  - `Negative`: `none`
  - `Neutral`: `{'R5': 0.344}`
  - `Positive`: `{'R2': 0.344, 'R7': 0.086, 'R8': 0.713, 'R9': 0.4}`

### Training a Branch on Custom Data

No dataset is bundled with the repository. To train a neural branch on prepared NumPy arrays (`Xtr`, `ytr`, `Xva`, `yva`):

```python
from branches import EEGFNN
from train import train_branch

model = EEGFNN(in_features=750)
trained_model, best_f1 = train_branch(model, Xtr, ytr, Xva, yva, epochs=30)
```

Epoch metrics and the best validation macro-F1 score are written to `outputs/train_log.txt` unless `log_path=None` is passed.

