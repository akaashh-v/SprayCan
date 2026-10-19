# TensorFlow Migration Guide: Acoustic Spray-Can Shaking Detector

This document details the completed migration of the **Acoustic Spray-Can Shaking Detector** Stage 2 CNN Classifier from **PyTorch** to **TensorFlow / Keras**.

---

## 1. Executive Summary

- **Conversion Strategy:** Direct High-Fidelity Mathematical Weight Transference (PyTorch state_dict → Keras layer variables).
- **Model Checkpoints:**
  - Original PyTorch model: [`backend/models/best_detector.pth`](file:///c:/Users/User/Desktop/CogniLearn/Spray/backend/models/best_detector.pth) (**Strictly preserved, untouched**).
  - Converted TensorFlow model: [`backend/models/best_detector.keras`](file:///c:/Users/User/Desktop/CogniLearn/Spray/backend/models/best_detector.keras)
  - Converted TensorFlow weights: [`backend/models/best_detector.weights.h5`](file:///c:/Users/User/Desktop/CogniLearn/Spray/backend/models/best_detector.weights.h5)
- **Validation Outcome:**
  - Class prediction agreement: **100.0% (25/25)** across all representative audio classes.
  - Verdict gating agreement: **100.0% (25/25)**.
  - Maximum probability divergence across all classes: **`5.36 × 10⁻⁷`** (essentially identical).
  - Inference latency: **2.86 ms / window** on CPU (with compiled `tf.function` graph).

---

## 2. Preserved Specifications & Invariants

### 2.1 Preserved Class Hierarchy (6 Classes)
```python
CLASS_NAMES = [
    "ambient_and_speech",        # 0
    "loud_impact",               # 1
    "continuous_environmental",  # 2
    "hard_negative_rattle",      # 3
    "aerosol_spray_hiss",        # 4
    "spray_can_shaking",         # 5 (Target)
]
```

### 2.2 Preserved Audio Preprocessing
- **Sample Rate:** 22,050 Hz
- **Window Size:** 1.0 second (22,050 samples)
- **Hop Duration:** 0.25 seconds (4 evaluations/sec)
- **Log-Mel Spectrogram:** 64 Mel frequency bands × 44 time frames
- **STFT Configuration:** Hann window, `N_FFT = 1024`, `HOP_LENGTH = 512`, `F_MIN = 100.0 Hz`, `F_MAX = 11,000.0 Hz`
- **Spectrogram Normalization:** Per-spectrogram zero-mean, unit-variance: `(log_mel - mean) / (std + 1e-6)`

---

## 3. Mathematical Layer Mapping

| Layer | PyTorch Weight Key | PyTorch Shape | TensorFlow Layer Name | Keras Weight Shape |
|---|---|---|---|---|
| Conv 1 | `conv1.0.weight` / `bias` | `(32, 1, 3, 3)` / `(32,)` | `conv1_conv` | `(3, 3, 1, 32)` / `(32,)` |
| BatchNorm 1 | `conv1.1.{weight, bias, running_mean, running_var}` | `(32,)` each | `conv1_bn` | `[gamma, beta, mean, var]` `(32,)` |
| Conv 2 | `conv2.0.weight` / `bias` | `(64, 32, 3, 3)` / `(64,)` | `conv2_conv` | `(3, 3, 32, 64)` / `(64,)` |
| BatchNorm 2 | `conv2.1.{weight, bias, running_mean, running_var}` | `(64,)` each | `conv2_bn` | `(64,)` each |
| Conv 3 | `conv3.0.weight` / `bias` | `(128, 64, 3, 3)` / `(128,)` | `conv3_conv` | `(3, 3, 64, 128)` / `(128,)` |
| BatchNorm 3 | `conv3.1.{weight, bias, running_mean, running_var}` | `(128,)` each | `conv3_bn` | `(128,)` each |
| Conv 4 | `conv4.0.weight` / `bias` | `(128, 128, 3, 3)` / `(128,)` | `conv4_conv` | `(3, 3, 128, 128)` / `(128,)` |
| BatchNorm 4 | `conv4.1.{weight, bias, running_mean, running_var}` | `(128,)` each | `conv4_bn` | `(128,)` each |
| Adaptive Pool | `AdaptiveAvgPool2d((4, 4))` | Function on `(8, 5)` | `adaptive_avg_pool` | Exact slicing & mean pooling |
| Transpose | Channel-first Flatten order | `(B, 128, 4, 4)` | `transpose_nchw` | `(B, 4, 4, 128) -> (B, 128, 4, 4)` |
| Dense 1 | `classifier.1.weight` / `bias` | `(128, 2048)` / `(128,)` | `classifier_dense1` | `(2048, 128)` / `(128,)` |
| BatchNorm Dense | `classifier.2.{weight, bias, running_mean, running_var}` | `(128,)` each | `classifier_bn1` | `(128,)` each |
| Dense Logits | `classifier.5.weight` / `bias` | `(6, 128)` / `(6,)` | `classifier_logits` | `(128, 6)` / `(6,)` |

---

## 4. New Files & Utilities

1. **[`backend/stage2_classifier_tf.py`](file:///c:/Users/User/Desktop/CogniLearn/Spray/backend/stage2_classifier_tf.py)**
   - Implements `build_spray_can_cnn_tf()` using Keras Functional API.
   - Implements `SprayCanShakingCNNTF` wrapper class matching the exact API of `SprayCanShakingCNN` (`predict_proba`, `forward`, `extract_features`, `load_weights`, `save`).
2. **[`backend/convert_pytorch_to_tf.py`](file:///c:/Users/User/Desktop/CogniLearn/Spray/backend/convert_pytorch_to_tf.py)**
   - High-fidelity conversion script transferring weights from `best_detector.pth` into `best_detector.keras`.
3. **[`backend/compare_pytorch_vs_tf.py`](file:///c:/Users/User/Desktop/CogniLearn/Spray/backend/compare_pytorch_vs_tf.py)**
   - Side-by-side comparative validation script testing probability divergence, prediction accuracy, and CPU latency.
4. **[`backend/tests/test_tensorflow_migration.py`](file:///c:/Users/User/Desktop/CogniLearn/Spray/backend/tests/test_tensorflow_migration.py)**
   - Automated test suite verifying parity, class preservation, feature extraction, and dual-backend operation.

---

## 5. How to Run & Switch Backends

The detector engine is designed with a **dual-backend adapter**. By default, it continues running on PyTorch (`MODEL_BACKEND = "pytorch"`).

### 5.1 Running with TensorFlow
To activate the TensorFlow detector backend:

**Option A: Environment Variable (Recommended for testing without code edits)**
```powershell
$env:MODEL_BACKEND = "tensorflow"
python run.py
```
*(On Linux/macOS: `export MODEL_BACKEND=tensorflow && python run.py`)*

**Option B: In `backend/config.py`**
```python
MODEL_BACKEND = "tensorflow"  # Switch default from "pytorch" to "tensorflow"
```

**Option C: Direct Python instantiation**
```python
from backend.detector_engine import SprayCanDetectorEngine

# Instantiate explicitly with TensorFlow backend
engine = SprayCanDetectorEngine(backend="tensorflow")
```

### 5.2 Instant Rollback to PyTorch
To revert immediately to the original PyTorch detector:
```powershell
$env:MODEL_BACKEND = "pytorch"
python run.py
```
or in `backend/config.py`:
```python
MODEL_BACKEND = "pytorch"
```
The original weights checkpoint `backend/models/best_detector.pth` remains untouched and fully intact.

---

## 6. How to Run Validation Tests

### Run Conversion:
```bash
python backend/convert_pytorch_to_tf.py
```

### Run Side-by-Side Audio Benchmark:
```bash
python backend/compare_pytorch_vs_tf.py
```

### Run Automated Test Suite:
```bash
python -m unittest backend/tests/test_tensorflow_migration.py -v
```
