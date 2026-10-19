# Acoustic Spray-Can Shaking Detector: YAMNet Transfer-Learning & Expanded Negative Data Experiments

This experimental workspace (`experiments/yamnet_transfer_learning/`) provides a fully isolated, reproducible framework comparing three distinct acoustic detection models for spray-can shaking:

- **Model A**: Current Production CNN (`backend/models/best_detector.keras`)
- **Model B**: Expanded-Data CNN (same architecture trained/fine-tuned on expanded ESC-50 and INMP441 negative audio)
- **Model C**: Transfer-Learning Classifier trained on frozen 1,024-dimensional YAMNet embeddings

> **Safety Notice**: Production models (`backend/models/best_detector.keras`) and microcontroller firmware (`firmware/esp32_mic/`) remain **100% untouched and unmodified**. All experiments, datasets, cached embeddings, checkpoints, and reports reside strictly within this directory.

---

## 1. Directory Structure

```
experiments/yamnet_transfer_learning/
├── README.md                              # This reproduction guide
├── inspection_report.md                   # Step 1 comprehensive project audit
├── data/
│   ├── dataset_manifest.json              # Full provenance manifest of all 1,849 audio files
│   ├── data_quality_report.json           # Corruption, silence, clipping & duplicate audit
│   ├── esc50_raw/                         # 255 curated ESC-50 negative audio WAV files
│   ├── own_recordings/                    # 1,594 INMP441 mic hardware recordings
│   ├── expanded_train.npz                 # 3,131 train windows (22.05 kHz and 16.0 kHz)
│   ├── expanded_val.npz                   # 342 validation windows
│   ├── expanded_test.npz                  # 469 held-out test windows (disjoint recordings)
│   └── yamnet_embeddings/                 # Cached 1,024-d YAMNet features & config
├── models/
│   ├── model_b_expanded_cnn/              # Checkpoints and eval reports for Model B
│   └── model_c_yamnet_classifier/         # Checkpoints and eval reports for Model C
├── results/
│   ├── metrics_summary.csv                # Machine-readable side-by-side performance table
│   ├── metrics_summary.json               # Full metrics JSON dictionary
│   ├── confusion_matrix_model_a.png       # Model A confusion matrix heatmap
│   ├── confusion_matrix_model_b.png       # Model B confusion matrix heatmap
│   ├── confusion_matrix_model_c.png       # Model C confusion matrix heatmap
│   ├── confusion_matrices_combined.png    # 3-panel side-by-side visual comparison
│   ├── esp32_yamnet_feasibility_report.md # Step 6 microcontroller feasibility analysis
│   └── comparison_report.md               # Step 7 final experimental report and recommendation
└── scripts/
    ├── prepare_expanded_dataset.py        # Curates ESC-50 + INMP441, filters, splits npz
    ├── train_expanded_cnn.py              # Trains/fine-tunes Model B with data augmentation
    ├── extract_yamnet_embeddings.py       # Computes & caches YAMNet features
    ├── train_yamnet_classifier.py         # Trains Model C 6-class classifier head
    ├── evaluate_and_compare.py            # Step 5 fair benchmark across Models A, B, and C
    └── test_esp32_yamnet_feasibility.py   # Step 6 hardware memory and latency benchmark
```

---

## 2. Six-Class Label Mapping

The exact 6-class ordering and IDs are preserved across all datasets, models, and evaluation scripts:

| ID | Class Name | Description |
|---|---|---|
| `0` | `ambient_and_speech` | Background room noise, conversation, fan hum |
| `1` | `loud_impact` | Claps, door slams, dropping heavy objects, footsteps |
| `2` | `continuous_environmental` | Wind, engine hum, running water, sirens |
| `3` | `hard_negative_rattle` | Key jangling, coin shakes, pill bottles, toolboxes |
| `4` | `aerosol_spray_hiss` | Continuous spray valve discharge without shaking |
| `5` | `spray_can_shaking` | **Target event**: Acoustic marble rattle inside spray can |

---

## 3. Environment Setup

The experiments run on Python 3.10+ (tested on Python 3.11 with TensorFlow 2.x):

```bash
# Activate virtual environment if applicable
# Install required dependencies
pip install "tensorflow>=2.15.0" "tensorflow-hub>=0.16.0" "tf-keras>=2.15.0" "setuptools<=69.5.1"
pip install librosa scikit-learn matplotlib numpy soundfile requests
```

---

## 4. Exact Reproduction Commands

Run each step sequentially from the project root (`c:\Users\User\Desktop\CogniLearn\Spray`):

### Step 1: Data Preparation & Quality Audit
Downloads curated ESC-50 negatives, scans INMP441 microphone recordings, performs silence/clipping/duplicate filtering, and generates disjoint recording-level train/val/test partitions:
```bash
python experiments/yamnet_transfer_learning/scripts/prepare_expanded_dataset.py
```

### Step 2: Train Model B (Expanded-Data CNN)
Trains the production CNN architecture with warm-start weights on the expanded dataset with training-only gain, time-shift, and noise augmentations:
```bash
python experiments/yamnet_transfer_learning/scripts/train_expanded_cnn.py
```

### Step 3: Extract & Cache YAMNet Embeddings
Loads official TF Hub YAMNet, processes 16 kHz waveforms, and caches 1,024-dimensional mean-pooled features:
```bash
python experiments/yamnet_transfer_learning/scripts/extract_yamnet_embeddings.py
```

### Step 4: Train Model C (YAMNet Transfer Learning Head)
Trains a 6-class dense MLP head on frozen YAMNet embeddings using early stopping and balanced class weighting:
```bash
python experiments/yamnet_transfer_learning/scripts/train_yamnet_classifier.py
```

### Step 5: Fair Benchmark & Comparison
Evaluates Models A, B, and C on the exact same held-out test recordings (`expanded_test.npz`):
```bash
python experiments/yamnet_transfer_learning/scripts/evaluate_and_compare.py
```

### Step 6: Microcontroller Feasibility Analysis
Quantifies YAMNet parameter size, activation memory arena, and FLOP latency against the ESP32-WROVER-B hardware limits:
```bash
python experiments/yamnet_transfer_learning/scripts/test_esp32_yamnet_feasibility.py
```
