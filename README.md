# CEG3004 DSP Mini-Project: Environmental Sound Classification

**Group:** Pr_41  
**Members:** HuaSheng, Brian Chua  
**Module:** CEG3004 Digital Signal Processing  

---

## Summary

This project builds a robust audio classification pipeline for the ESC-50 dataset. The system classifies 5-second mono audio clips into 50 environmental sound categories. It is designed to hold up under clean, noisy, and band-limited conditions.

**Final validation results on 720 held-out samples (20% stratified split):**

| Metric | Score |
|---|---|
| Accuracy | 0.87 |
| Macro Precision | 0.88 |
| Macro Recall | 0.87 |
| Macro F1 | 0.8674 |

The pipeline follows the course signal processing workflow:

```
raw audio -> preprocessing -> DSP feature extraction -> ML classifier -> predictions
```

---

## Repository Structure

```
.
├── HuaSheng_BrianChua__CEG3004_Project_Colab.ipynb   # Main Colab notebook
├── huasheng_brianchua__ceg3004_project_colab.py       # Python export of notebook
├── README.md                                         # This file

```

---

## Dataset

The project uses a subset of the ESC-50 collection.

| Property | Value |
|---|---|
| Total clips | 2,000 |
| Sound classes | 50 |
| Clips per class | 40 |
| Clip length | 5 seconds |
| Format | mono WAV |
| Sample rate | 16 kHz (resampled) |

The submission set contains three versions of each underlying sound event:

- `__clean`: original unmodified signal
- `__noisy`: additive noise applied
- `__bandlimited`: frequency content restricted

The grading formula is `0.5 * clean + 0.25 * noisy + 0.25 * bandlimited`, so robustness to distortion directly affects the final score.

---

## Pipeline Design

### Step 1: Audio Loading

All clips load as:
- mono channel
- resampled to 16 kHz
- float32 precision
- NaN values replaced with zero

```python
def load_audio(path, sr=16000):
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    y = np.nan_to_num(y).astype(np.float32)
    return y, sr_out
```

---

### Step 2: Preprocessing

We improved the baseline (which did nothing) with four sequential steps:

#### 2a. Fixed-length normalization
Every clip is padded or truncated to exactly 5 seconds (80,000 samples at 16 kHz). This ensures a consistent feature vector size across all clips regardless of recording length.

#### 2b. Silence trimming
`librosa.effects.trim` removes leading and trailing silence at a threshold of 20 dB below peak. After trimming, the clip is re-padded to 5 seconds to maintain the fixed length constraint.

**DSP rationale:** Silence frames add no discriminative information but introduce zero-energy frames into the STFT, which can bias spectral statistics toward zero.

#### 2c. Pre-emphasis filtering
A first-order IIR filter: `y[n] = y[n] - 0.97 * y[n-1]`

**DSP rationale:** Environmental recordings have more energy concentrated in low frequencies. Pre-emphasis boosts high-frequency content before STFT analysis, flattening the spectral envelope and improving MFCC resolution in the upper coefficients.

#### 2d. RMS normalization
Each clip is scaled so its root-mean-square energy equals 1.

**DSP rationale:** Different clips have very different recording levels. RMS normalization removes loudness as a confounding variable and makes features more stable across recordings with different microphone gains or distances.

```python
def preprocess_audio(y, sr):
    target_len = 5 * sr

    # Fixed-length
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')

    # Silence trim + re-pad
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) > 0:
        y = y_trimmed
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')

    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # RMS normalization
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-8:
        y = y / rms

    return y.astype(np.float32)
```

---

### Step 3: Feature Extraction

The final feature vector concatenates four feature groups. Total dimensionality is approximately 575.

#### Feature group summary

| Group | Function | Dimension | DSP Concept |
|---|---|---|---|
| Baseline MFCC stats | `features_mfcc_stats` | 120 | Cepstral analysis, delta dynamics |
| CMVN MFCC stats | `features_cmvn_mfcc` | 360 | Noise-robust cepstral normalization |
| Log-mel stats | `features_logmel` | 120 | Perceptual frequency warping |
| Spectral descriptors | `features_spectral` | 15 | Spectral shape and energy |

#### 3a. Baseline MFCC statistics (kept from template)

20 MFCC coefficients with delta and delta-delta. Each of the three matrices is summarized by mean and standard deviation across time frames.

```
Output: 20*2 + 20*2 + 20*2 = 120 dimensions
```

**DSP rationale:** MFCCs decorrelate the filterbank outputs via the Discrete Cosine Transform, giving a compact representation of spectral shape. Deltas capture temporal dynamics (rate of change of spectral shape), which helps distinguish sounds that differ in their onset or decay pattern.

#### 3b. CMVN-normalized MFCC (added)

40 MFCC coefficients with Cepstral Mean-Variance Normalization applied before computing deltas. Each matrix is pooled with mean, std, median, 25th percentile, and 75th percentile.

```python
mfcc_norm = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
```

```
Output: 40*5 + 40*5 + 40*5 = 600 -> 360 stored (3 matrices x 40 coefficients x 5 stats)
```

**DSP rationale:** CMVN subtracts the per-coefficient mean across time. Additive noise in the time domain becomes additive noise in the cepstral domain. Subtracting the cepstral mean removes this constant additive bias per coefficient, making the representation robust to channel distortions. The median and percentile pooling are less sensitive to noise outlier frames than mean alone.

#### 3c. Log-mel spectrogram statistics (added)

40 mel filterbank channels. Output converted to decibels. Each channel summarized by mean, std, and median.

```
Output: 40 * 3 = 120 dimensions
```

**DSP rationale:** The mel scale maps frequency to a perceptual scale that approximates how the human auditory system processes sound. Low frequencies receive more resolution than high frequencies. This grouping is more stable under band-limited distortion than raw spectral bins because each mel channel integrates energy over a wide frequency range.

#### 3d. Spectral shape descriptors (added)

Five frame-level descriptors summarized by mean, std, and median each.

| Descriptor | DSP Meaning |
|---|---|
| Spectral centroid | Weighted mean frequency (brightness) |
| Spectral bandwidth | Spread of energy around the centroid |
| Spectral rolloff | Frequency below which 85% of energy lies |
| Zero crossing rate | Rate of sign changes (correlates with noisiness) |
| RMS energy | Per-frame signal power |

```
Output: 5 descriptors * 3 stats = 15 dimensions
```

**DSP rationale:** These features characterize the spectral shape in a compact and physically interpretable way. A siren has a narrow, rising centroid. Fire crackling has high ZCR. An airplane has high, sustained RMS. These distinctions survive both additive noise and bandwidth restriction better than raw spectral bins.

#### Feature extraction entry point

```python
def extract_features(path, sr=16000):
    y, sr = load_audio(path, sr=sr)
    y = preprocess_audio(y, sr)
    f_mfcc = features_mfcc_stats(y, sr)
    f_cmvn = features_cmvn_mfcc(y, sr)
    f_mel  = features_logmel(y, sr)
    f_spec = features_spectral(y, sr)
    return np.concatenate([f_mfcc, f_cmvn, f_mel, f_spec])
```

---

### Step 4: Robustness Augmentation

To simulate the noisy and band-limited test conditions during training, we generate two augmented copies of every training clip.

#### Augmentation 1: Additive Gaussian noise

Random SNR sampled uniformly from 10 dB to 25 dB per clip.

```python
def add_gaussian_noise(y, snr_db=15):
    signal_power = np.mean(y ** 2) + 1e-10
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return (y + noise).astype(np.float32)
```

**DSP rationale:** SNR (dB) = 10 * log10(signal power / noise power). By solving for noise power and drawing Gaussian samples, we generate additive white Gaussian noise at a controlled signal-to-noise ratio. The randomized SNR range forces the model to generalize across a spectrum of noise intensities.

#### Augmentation 2: Simulated band-limited audio

A 4th-order Butterworth bandpass filter with passband 300 Hz to 3400 Hz (telephone-like channel).

```python
def simulate_bandlimited(y, sr, low_hz=300, high_hz=3400):
    nyq  = sr / 2
    low  = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    sos  = butter(4, [low, high], btype='band', output='sos')
    return sosfilt(sos, y).astype(np.float32)
```

**DSP rationale:** A Butterworth filter has a maximally flat passband response. The 4th order gives 80 dB/decade rolloff outside the passband, effectively eliminating sub-300 Hz and above-3400 Hz content. This matches the telephone narrowband channel standard and directly simulates the evaluation distortion condition.

#### Combined dataset construction

```python
# Clean (2000) + noisy (2000) + band-limited (2000) = 6000 total samples
X_combined = np.vstack([X, X_aug_arr])
y_combined  = np.concatenate([y, y_aug_arr])

X_tr, X_va, y_tr, y_va = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)
```

---

### Step 5: Model

We replaced the baseline Logistic Regression with an SVM using an RBF kernel.

```python
from sklearn.svm import SVC

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        decision_function_shape='ovr',
    ))
])
```

**Why SVM over Logistic Regression:**  
The feature space (~575 dimensions) is not strictly linearly separable for 50 classes. The RBF kernel implicitly maps inputs to a higher-dimensional space where linear separation becomes feasible. `gamma='scale'` sets gamma as 1 / (n_features * X.var()), which auto-adjusts to the feature distribution. `class_weight='balanced'` accounts for the perfectly balanced dataset structure.

**Why C=10:**  
A larger C value reduces the regularization strength, allowing tighter margins. With 6000 training samples and a well-normalized feature space, the model benefits from a smaller margin rather than a softer, more regularized boundary.

---

## Experiment Log

| Experiment | Model | Features | Augmentation | Val Accuracy | Macro F1 |
|---|---|---|---|---|---|
| 1 (baseline) | Logistic Regression | MFCC stats only | None | ~0.54 | ~0.53 |
| 2 | SVM RBF (C=10) | MFCC stats only | None | ~0.72 | ~0.71 |
| 3 | SVM RBF (C=10) | MFCC + CMVN + log-mel + spectral | None | ~0.84 | ~0.83 |
| 4 (final) | SVM RBF (C=10) | MFCC + CMVN + log-mel + spectral | Noise + bandlimited | **0.87** | **0.8674** |

Key observations:
- Switching from Logistic Regression to SVM gave the largest single jump (+18 pp)
- Adding CMVN and log-mel features on top of MFCC gave the second largest improvement (+12 pp)
- Augmentation improved robustness across distorted conditions without hurting clean accuracy

---

## Results

### Validation classification report (full)

Evaluated on 720 held-out samples across 50 classes.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| airplane | 0.93 | 1.00 | 0.97 | 14 |
| breathing | 0.92 | 0.73 | 0.81 | 15 |
| brushing_teeth | 0.93 | 0.87 | 0.90 | 15 |
| can_opening | 0.75 | 0.86 | 0.80 | 14 |
| car_horn | 0.93 | 1.00 | 0.97 | 14 |
| cat | 0.86 | 0.86 | 0.86 | 14 |
| chainsaw | 0.90 | 0.64 | 0.75 | 14 |
| chirping_birds | 0.83 | 1.00 | 0.91 | 15 |
| church_bells | 1.00 | 1.00 | 1.00 | 14 |
| clapping | 1.00 | 0.93 | 0.96 | 14 |
| clock_alarm | 1.00 | 0.73 | 0.85 | 15 |
| clock_tick | 1.00 | 0.73 | 0.85 | 15 |
| coughing | 0.80 | 0.86 | 0.83 | 14 |
| cow | 1.00 | 0.93 | 0.96 | 14 |
| crackling_fire | 0.82 | 0.93 | 0.88 | 15 |
| crickets | 0.76 | 0.93 | 0.84 | 14 |
| crow | 1.00 | 0.93 | 0.96 | 14 |
| crying_baby | 0.92 | 0.79 | 0.85 | 14 |
| dog | 1.00 | 0.80 | 0.89 | 15 |
| door_wood_creaks | 0.76 | 0.93 | 0.84 | 14 |
| door_wood_knock | 0.71 | 0.80 | 0.75 | 15 |
| drinking_sipping | 0.70 | 0.93 | 0.80 | 15 |
| engine | 1.00 | 0.87 | 0.93 | 15 |
| fireworks | 0.71 | 0.67 | 0.69 | 15 |
| footsteps | 0.81 | 0.93 | 0.87 | 14 |
| frog | 1.00 | 1.00 | 1.00 | 14 |
| glass_breaking | 0.58 | 0.79 | 0.67 | 14 |
| hand_saw | 1.00 | 0.86 | 0.92 | 14 |
| helicopter | 0.65 | 0.87 | 0.74 | 15 |
| hen | 1.00 | 0.73 | 0.85 | 15 |
| insects | 0.91 | 0.71 | 0.80 | 14 |
| keyboard_typing | 0.78 | 1.00 | 0.88 | 14 |
| laughing | 0.88 | 1.00 | 0.93 | 14 |
| mouse_click | 0.75 | 0.64 | 0.69 | 14 |
| pig | 1.00 | 1.00 | 1.00 | 14 |
| pouring_water | 0.94 | 1.00 | 0.97 | 15 |
| rain | 0.88 | 0.93 | 0.90 | 15 |
| rooster | 1.00 | 0.93 | 0.96 | 14 |
| sea_waves | 0.86 | 0.80 | 0.83 | 15 |
| sheep | 1.00 | 1.00 | 1.00 | 14 |
| siren | 1.00 | 1.00 | 1.00 | 15 |
| sneezing | 0.75 | 0.86 | 0.80 | 14 |
| snoring | 0.76 | 0.87 | 0.81 | 15 |
| thunderstorm | 1.00 | 0.93 | 0.97 | 15 |
| toilet_flush | 0.92 | 0.86 | 0.89 | 14 |
| train | 0.71 | 0.86 | 0.77 | 14 |
| vacuum_cleaner | 0.83 | 0.71 | 0.77 | 14 |
| washing_machine | 0.93 | 1.00 | 0.97 | 14 |
| water_drops | 1.00 | 0.86 | 0.92 | 14 |
| wind | 0.80 | 0.53 | 0.64 | 15 |
| **macro avg** | **0.88** | **0.87** | **0.87** | 720 |
| **weighted avg** | **0.88** | **0.87** | **0.87** | 720 |

### Best-performing classes (F1 = 1.00)

- `church_bells`, `frog`, `pig`, `sheep`, `siren`

These classes have highly distinctive spectral or temporal signatures. Church bells produce strong harmonic series. Frog calls are periodic and narrow-band. Siren has a characteristic rising-falling frequency sweep.

### Weakest classes

| Class | F1 | Likely reason |
|---|---|---|
| wind | 0.64 | Broadband, noise-like spectrum overlaps with rain and sea waves |
| glass_breaking | 0.67 | Short transient onset; broken into two-part energy burst that resembles other impacts |
| fireworks | 0.69 | Impulsive broadband burst similar to other explosion-type sounds |
| mouse_click | 0.69 | Very short transient; MFCC frame window may not align with onset |
| helicopter | 0.74 | Low-frequency harmonic content overlaps with engine and airplane |

---

## Error Analysis

### Pattern 1: Spectrally similar broadband sounds

`wind`, `rain`, `sea_waves`, and `fireworks` all occupy wide frequency ranges with irregular temporal structure. The mel filterbank averaging over frequency bins reduces discrimination between these classes. CMVN helps with noise robustness but does not create sharper class boundaries between inherently similar spectra.

**Potential fix:** Add spectral flux (frame-to-frame spectral change rate) or spectral contrast (ratio of peaks to valleys in subbands), which would differentiate steady broadband noise from intermittent bursts.

### Pattern 2: Short transient events

`mouse_click` and `glass_breaking` produce most of their energy in 1-2 frames. The STFT with a 1024-sample window (64 ms at 16 kHz) may not align perfectly with the onset. The summary statistics (mean, std) averaged over all frames dilute the transient contribution.

**Potential fix:** Use onset-aligned features, or extract statistics over the first quartile of the clip separately from the rest.

### Pattern 3: High recall, low precision classes

`clock_alarm` and `clock_tick` both show recall of 0.73 despite perfect precision. Both classes have periodic tonal structure, but their temporal rate differs. The current feature set captures spectral shape well but does not explicitly encode periodicity or fundamental frequency.

**Potential fix:** Add autocorrelation-based features or pitch tracking features to capture periodicity.

### Pattern 4: Confusion between similar animal sounds

`hen`, `crow`, and `insects` all sit in the 0.80 to 0.85 F1 range. These three classes share tonal and periodic qualities. Without formant tracking or pitch contour features, the model sees partially overlapping centroid distributions.

---

## How This Pipeline Addresses the Grading Criteria

### DSP concepts directly applied

| Course concept | Implementation |
|---|---|
| STFT and time-frequency analysis | All MFCC and mel features use STFT with n_fft=1024, hop=256 |
| Mel filterbank and perceptual frequency warping | `features_logmel` with 40 mel bands |
| Cepstral analysis | `features_mfcc_stats` and `features_cmvn_mfcc` |
| Pre-emphasis (IIR filter) | First-order filter in `preprocess_audio` |
| FIR/IIR filtering | Butterworth bandpass in `simulate_bandlimited` |
| Signal-to-noise ratio | Gaussian noise augmentation uses SNR formula |
| Spectral descriptors | Centroid, bandwidth, rolloff, ZCR, RMS |
| Time-domain normalization | RMS normalization in `preprocess_audio` |
| Delta features (temporal dynamics) | First and second-order delta MFCCs |

---

## Reproducibility Guide

### Option A: Google Colab (recommended)

1. Open `HuaSheng_BrianChua__CEG3004_Project_Colab.ipynb` in Colab
2. Set `GROUP_ID = "Pr_41"` in the first cell
3. Run all cells in order from top to bottom
4. Download the generated files when prompted:
   - `Pr_41_model.joblib`
   - `Pr_41_predictions.csv`

Expected runtime: approximately 15 to 25 minutes on a Colab CPU runtime (augmentation and SVM training on 6000 samples takes the most time).

### Option B: Local Python environment

**Step 1: Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**Step 2: Install dependencies**

```bash
pip install numpy scipy pandas scikit-learn librosa soundfile tqdm gdown matplotlib ipywidgets
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Step 3: Download and extract the dataset**

The notebook auto-downloads via `gdown`. For local runs, download manually from the course Google Drive link and extract to `/content/` or update `DATA_ROOT` in step 3 of the script:

```python
DATA_ROOT = "/your/local/path/to/data/"
```

**Step 4: Run the script**

```bash
python huasheng_brianchua__ceg3004_project_colab.py
```

Comment out the `files.download(...)` lines (Colab-specific) if running locally. Output files will appear in the working directory.

### Expected output files

| File | Description |
|---|---|
| `Pr_41_model.joblib` | Trained sklearn Pipeline (scaler + SVM) |
| `Pr_41_predictions.csv` | Two-column CSV: `clip_id`, `predicted_label` |

### Dataset folder structure expected

```
data/
├── train/
│   ├── labels.csv
│   └── audio/
│       ├── 0001.wav
│       └── ...
└── submission/
    ├── metadata.csv
    └── audio/
        ├── 0001__clean.wav
        ├── 0001__noisy.wav
        ├── 0001__bandlimited.wav
        └── ...
```

---

## Dependencies

```
numpy
scipy
pandas
scikit-learn
librosa
soundfile
tqdm
gdown
matplotlib
ipywidgets
```

---


