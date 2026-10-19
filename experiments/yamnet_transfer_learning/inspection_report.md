# Project Inspection & Baseline Architecture Audit Report
**Date:** 2026-10-17  
**Workspace:** `experiments/yamnet_transfer_learning/`  
**Task:** Safe YAMNet Transfer-Learning and Negative-Data Expansion Experiment  

---

## 1. Existing Dataset & Training Infrastructure

### 1.1 Dataset Folders & Ground-Truth Manifest
- **Raw Audio Root:** `backend/data/raw/` containing 48 subfolders and 974 audio files across authentic field recordings, hardware captures, synthetic physical mimics, and ambient noise.
- **File Formats:** Uncompressed 16-bit PCM `.wav`, `.mp3`, and `.m4a` files with sample rates ranging from 16,000 Hz to 48,000 Hz.
- **Recording-Level Split Manifest:** `backend/data/processed/split_manifest.json` specifies exact disjoint partitions for `train`, `val`, and `test` at the source recording level.
  - **Audit Result:** 974 / 974 audio files referenced across all partitions exist on disk (0 missing).
- **Processed Window Partitions:**
  - `backend/data/processed/train.pt`: 5,224 windows (58.9 MB)
  - `backend/data/processed/val.pt`: 523 windows (5.9 MB)
  - `backend/data/processed/test.pt`: 1,652 windows (18.6 MB)
  - `backend/data/processed/dataset_info.json`: Detailed class counts per partition.
  - `backend/data/processed/test_conditions.json`: Acoustic condition metadata (distance, angle, background noise level).
- **ESP32 Hardware Captures:**
  - `backend/tmp_esp32_audio/`: 686 real-world 1.0-second 22,050 Hz WAV captures recorded directly by the INMP441 I2S microphone on the ESP32.
  - `backend/recordings/`: 53 full multi-second alert event recordings saved by the real-time event recorder.

### 1.2 Training Scripts & Active Pipelines
- `backend/train_and_evaluate.py`: PyTorch training, validation early stopping, conservative threshold tuning, and held-out test evaluation.
- `backend/dataset_builder.py`: Window extraction, audio loading, resampling, Log-Mel spectrogram generation, and SpecAugment.
- `backend/stage2_classifier_tf.py`: Keras / TensorFlow functional model architecture and prediction runtime.
- `backend/convert_pytorch_to_tf.py` & `backend/compare_pytorch_vs_tf.py`: Weights transfer and cross-framework numeric parity verification.
- `backend/training_manager.py`: Automated candidate model training, evaluation, and zero-downtime hot-reload engine.

---

## 2. Production Model Architecture & Class Mapping

### 2.1 Active Checkpoint: `backend/models/best_detector.keras`
- **Model Format:** Keras 3 / TensorFlow 2.x SavedModel format (`best_detector.keras`, 2,117,321 bytes).
- **Total Parameters:** 505,222 (~1.93 MB unquantized float32)
  - Trainable parameters: 504,262
  - Non-trainable parameters: 960 (BatchNormalization running mean/variance)
- **Input Tensor:** `spectrogram_input: shape=(None, 64, 44, 1), dtype=float32`
- **Output Tensors:**
  - `probabilities: shape=(None, 6), dtype=float32` (Softmax class probabilities)
  - `logits: shape=(None, 6), dtype=float32` (Raw pre-activation logits)
  - `features_flat: shape=(None, 2048), dtype=float32` (Bottleneck embedding vector)

### 2.2 Layer Architecture
```
Layer (type)                    Output Shape           Param #     Details
====================================================================================
spectrogram_input (InputLayer)  (None, 64, 44, 1)      0           64 Mel bins x 44 time frames
conv1_conv (Conv2D)             (None, 64, 44, 32)     320         3x3 conv, stride 1, same pad
conv1_bn (BatchNormalization)   (None, 64, 44, 32)     128         eps=1e-5, momentum=0.9
conv1_relu (ReLU)               (None, 64, 44, 32)     0           Rectified linear
conv1_pool (MaxPooling2D)       (None, 32, 22, 32)     0           2x2 pool, stride 2
conv1_drop (Dropout)            (None, 32, 22, 32)     0           p = 0.15
conv2_conv (Conv2D)             (None, 32, 22, 64)     18,496      3x3 conv, stride 1, same pad
conv2_bn (BatchNormalization)   (None, 32, 22, 64)     256         eps=1e-5, momentum=0.9
conv2_relu (ReLU)               (None, 32, 22, 64)     0           Rectified linear
conv2_pool (MaxPooling2D)       (None, 16, 11, 64)     0           2x2 pool, stride 2
conv2_drop (Dropout)            (None, 16, 11, 64)     0           p = 0.20
conv3_conv (Conv2D)             (None, 16, 11, 128)    73,856      3x3 conv, stride 1, same pad
conv3_bn (BatchNormalization)   (None, 16, 11, 128)    512         eps=1e-5, momentum=0.9
conv3_relu (ReLU)               (None, 16, 11, 128)    0           Rectified linear
conv3_pool (MaxPooling2D)       (None, 8, 5, 128)      0           2x2 pool, stride 2
conv3_drop (Dropout)            (None, 8, 5, 128)      0           p = 0.25
conv4_conv (Conv2D)             (None, 8, 5, 128)      147,584     3x3 conv, stride 1, same pad
conv4_bn (BatchNormalization)   (None, 8, 5, 128)      512         eps=1e-5, momentum=0.9
conv4_relu (ReLU)               (None, 8, 5, 128)      0           Rectified linear
adaptive_avg_pool (Lambda)      (None, 4, 4, 128)      0           PyTorch AdaptiveAvgPool2d((4,4))
transpose_nchw (Lambda)         (None, 128, 4, 4)      0           Transpose for PyTorch index parity
features_flatten (Flatten)      (None, 2048)           0           Flatten to 2,048 feature dims
classifier_dense1 (Dense)       (None, 128)            262,272     Fully connected layer
classifier_bn1 (BatchNormalization) (None, 128)        512         eps=1e-5, momentum=0.9
classifier_relu1 (ReLU)         (None, 128)            0           Rectified linear
classifier_drop1 (Dropout)      (None, 128)            0           p = 0.40
classifier_logits (Dense)       (None, 6)              774         6-class output logits
probabilities (Softmax)         (None, 6)              0           Normalized probabilities
====================================================================================
```

### 2.3 Preserved Six-Class Order & Semantics
| Class Index | Class Name | Description / Sound Sources |
|---|---|---|
| `0` | `ambient_and_speech` | Speech, vocalizations, room tone, keyboard typing, fan hum, breathing |
| `1` | `loud_impact` | Door slams, desk bangs, dropped objects, claps, hammer hits, glass breaking |
| `2` | `continuous_environmental`| Rain, water drops, train rumbles, HVAC hum, vehicle engines |
| `3` | `hard_negative_rattle` | Coins, keys, pill bottles, screws in tin, bottle caps, loose glassware |
| `4` | `aerosol_spray_hiss` | Continuous & pulsed aerosol spraying, nozzle discharge, sputtering |
| `5` | `spray_can_shaking` | **ONLY TARGET:** Verified aerosol spray can shaken with internal mixing ball |

---

## 3. Preprocessing, Windowing, Augmentation, and Gating

### 3.1 Audio Parameters
- **Audio Sample Rate:** 22,050 Hz
- **Window Duration:** 1.0 second (22,050 samples)
- **Hop Duration:** 0.25 seconds (5,512 samples) in streaming mode
- **FFT Size (`n_fft`):** 1024
- **Hop Length (`hop_length`):** 512 samples (~23.2 ms per frame)
- **Mel Filter Bins (`n_mels`):** 64
- **Frequency Range:** $f_{min} = 100.0\text{ Hz}$, $f_{max} = 11,000.0\text{ Hz}$
- **Spectrogram Time Frames:** $1 + \lfloor 22050 / 512 \rfloor = 44\text{ frames}$
- **Log-Mel Formula:** $\log_{10}(\text{Mel} + 10^{-6})$

### 3.2 Decision Gating & Cooldown Rules
- **Conservative Shaking Threshold:** $P(\text{spray\_can\_shaking}) \ge 0.80$
- **Dominance Margin:** $P(\text{spray}) - \max(P(\text{mimics})) \ge 0.35$
- **Max Competing Confidence:** $\max(P(\text{hiss}), P(\text{rattle}), P(\text{impact})) \le 0.20$
- **Uncertainty Entropy Cutoff:** $H_{\text{norm}} \le 0.70$
- **Temporal Persistence:** 2 consecutive candidate windows (0.50s prompt detection)

---

## 4. INT8 Quantization & ESP32 Integration

- **TFLite Evaluation Models:**
  - `backend/models/tflite_eval/best_detector_int8.tflite`: Post-training INT8 quantized model (532,528 bytes, ~520 KB).
  - Validated by automated parity tests in `backend/tests/test_edge_deployment_parity.py`.
- **ESP32 Firmware Runtime:**
  - `firmware/esp32_mic/model_data.cpp`: Embedded byte array of the TFLite model.
  - `firmware/esp32_mic/edge_detector.cpp`: TensorFlow Lite for Microcontrollers (TFLM) runtime using a 260 KB Tensor Arena in PSRAM.
  - `firmware/esp32_mic/stage1_filter.h`: Kinematic time-sequence pre-filter running in $<1\text{ ms}$ on raw PCM.

---

## 5. Verification Checklist & Missing Files
- [x] Production Keras model `best_detector.keras` exists and validated.
- [x] PyTorch model `best_detector.pth` exists and validated.
- [x] All 974 source audio files in manifest exist and verified on disk.
- [x] Preprocessed PyTorch partitions `train.pt`, `val.pt`, `test.pt` match `dataset_info.json`.
- [x] TensorFlow 2.21.0, PyTorch 2.14.1, and `tensorflow-hub` 0.16.1 installed and operational.
- [x] Dedicated experimental workspace `experiments/yamnet_transfer_learning/` created and isolated from production assets.
