"""Comprehensive Fair Comparison of Models A, B, and C on Held-Out Test Set.

Step 5 Requirements:
- Identical held-out test recordings for all three models (expanded_test.npz).
- Metrics evaluated:
  * Overall accuracy & macro F1-score
  * Per-class precision, recall, and F1-score
  * Confusion matrices (saved as JSON, CSV, and PNG plots)
  * False alarms on negative recordings & False Positives per Hour
  * Missed-event rate for genuine spray-can shaking
  * Hard negative rattle breakdown
  * Preprocessing time, inference latency, parameter count, and model file sizes
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_and_compare")

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent
EXP_DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_A_PATH = PROJECT_ROOT / "backend" / "models" / "best_detector.keras"
MODEL_B_PATH = BASE_DIR / "models" / "model_b_expanded_cnn" / "expanded_cnn_best.keras"
MODEL_C_HEAD_PATH = BASE_DIR / "models" / "model_c_yamnet_classifier" / "yamnet_classifier_best.keras"
YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"

CLASS_NAMES = [
    "ambient_and_speech",
    "loud_impact",
    "continuous_environmental",
    "hard_negative_rattle",
    "aerosol_spray_hiss",
    "spray_can_shaking",
]
NUM_CLASSES = len(CLASS_NAMES)
SAMPLE_RATE_22K = 22050
SAMPLE_RATE_16K = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 100.0
F_MAX = 11000.0


def compute_log_mel_batch(audio_batch: np.ndarray, sr: int = SAMPLE_RATE_22K) -> np.ndarray:
    """Compute Log-Mel spectrograms matching production."""
    specs = []
    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    for i in range(len(audio_batch)):
        y = audio_batch[i]
        D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=True)) ** 2
        mel = np.dot(mel_basis, D)
        log_mel = np.log10(mel + 1e-6)
        if log_mel.shape[1] > 44:
            log_mel = log_mel[:, :44]
        elif log_mel.shape[1] < 44:
            log_mel = np.pad(log_mel, ((0, 0), (0, 44 - log_mel.shape[1])), mode="reflect")
        specs.append(log_mel[:, :, np.newaxis])
    return np.array(specs, dtype=np.float32)


def build_cnn_model(name: str = "SprayCanCNN") -> keras.Model:
    """Production architecture builder."""
    inputs = keras.Input(shape=(64, 44, 1), name="spectrogram_input")
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_conv")(inputs)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv1_pool")(x)
    x = layers.Dropout(0.15, name="conv1_drop")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv2_bn")(x)
    x = layers.ReLU(name="conv2_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv2_pool")(x)
    x = layers.Dropout(0.20, name="conv2_drop")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv3_bn")(x)
    x = layers.ReLU(name="conv3_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv3_pool")(x)
    x = layers.Dropout(0.25, name="conv3_drop")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", name="conv4_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv4_bn")(x)
    x = layers.ReLU(name="conv4_relu")(x)

    def _pool_4x4(t):
        w0 = tf.reduce_mean(t[:, :, 0:2, :], axis=2, keepdims=True)
        w1 = tf.reduce_mean(t[:, :, 1:3, :], axis=2, keepdims=True)
        w2 = tf.reduce_mean(t[:, :, 2:4, :], axis=2, keepdims=True)
        w3 = tf.reduce_mean(t[:, :, 3:5, :], axis=2, keepdims=True)
        w_p = tf.concat([w0, w1, w2, w3], axis=2)
        h0 = tf.reduce_mean(w_p[:, 0:2, :, :], axis=1, keepdims=True)
        h1 = tf.reduce_mean(w_p[:, 2:4, :, :], axis=1, keepdims=True)
        h2 = tf.reduce_mean(w_p[:, 4:6, :, :], axis=1, keepdims=True)
        h3 = tf.reduce_mean(w_p[:, 6:8, :, :], axis=1, keepdims=True)
        return tf.concat([h0, h1, h2, h3], axis=1)

    x = layers.Lambda(_pool_4x4, name="adaptive_avg_pool")(x)
    x_ch_first = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 3, 1, 2]), name="transpose_nchw")(x)
    features_flat = layers.Flatten(name="features_flatten")(x_ch_first)

    fc1 = layers.Dense(128, name="classifier_dense1")(features_flat)
    fc1_bn = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="classifier_bn1")(fc1)
    fc1_relu = layers.ReLU(name="classifier_relu1")(fc1_bn)
    fc1_drop = layers.Dropout(0.40, name="classifier_drop1")(fc1_relu)
    logits = layers.Dense(NUM_CLASSES, name="classifier_logits")(fc1_drop)
    probs = layers.Softmax(name="probabilities")(logits)

    return keras.Model(inputs=inputs, outputs=probs, name=name)


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold" if i == j else "normal",
            )
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def benchmark_model_a(x_test_22k: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluate Model A: Current Production CNN."""
    logger.info("Evaluating Model A (Current Production CNN)...")
    model = build_cnn_model(name="ModelA_ProductionCNN")
    model.load_weights(str(MODEL_A_PATH))

    # Benchmark preprocessing time (100 windows)
    t_pre0 = time.perf_counter()
    sample_sub = x_test_22k[:50]
    _ = compute_log_mel_batch(sample_sub)
    pre_time_ms = ((time.perf_counter() - t_pre0) / 50) * 1000.0

    # Preprocess all test samples
    x_test_spec = compute_log_mel_batch(x_test_22k)

    # Benchmark inference time (warmup + 50 passes)
    _ = model.predict(x_test_spec[:1], verbose=0)
    t_inf0 = time.perf_counter()
    for i in range(50):
        _ = model.predict(x_test_spec[i : i + 1], verbose=0)
    inf_time_ms = ((time.perf_counter() - t_inf0) / 50) * 1000.0

    probs = model.predict(x_test_spec, verbose=0)
    file_size_kb = os.path.getsize(MODEL_A_PATH) / 1024.0

    perf = {
        "params": int(model.count_params()),
        "file_size_kb": float(file_size_kb),
        "preprocessing_ms": float(pre_time_ms),
        "inference_ms": float(inf_time_ms),
        "total_latency_ms": float(pre_time_ms + inf_time_ms),
    }
    return probs, perf


def benchmark_model_b(x_test_22k: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluate Model B: Expanded-Data CNN."""
    logger.info("Evaluating Model B (Expanded-Data CNN)...")
    model = build_cnn_model(name="ModelB_ExpandedCNN")
    model.load_weights(str(MODEL_B_PATH))

    t_pre0 = time.perf_counter()
    sample_sub = x_test_22k[:50]
    _ = compute_log_mel_batch(sample_sub)
    pre_time_ms = ((time.perf_counter() - t_pre0) / 50) * 1000.0

    x_test_spec = compute_log_mel_batch(x_test_22k)

    _ = model.predict(x_test_spec[:1], verbose=0)
    t_inf0 = time.perf_counter()
    for i in range(50):
        _ = model.predict(x_test_spec[i : i + 1], verbose=0)
    inf_time_ms = ((time.perf_counter() - t_inf0) / 50) * 1000.0

    probs = model.predict(x_test_spec, verbose=0)
    file_size_kb = os.path.getsize(MODEL_B_PATH) / 1024.0

    perf = {
        "params": int(model.count_params()),
        "file_size_kb": float(file_size_kb),
        "preprocessing_ms": float(pre_time_ms),
        "inference_ms": float(inf_time_ms),
        "total_latency_ms": float(pre_time_ms + inf_time_ms),
    }
    return probs, perf


def benchmark_model_c(x_test_16k: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluate Model C: YAMNet Transfer-Learning Classifier."""
    logger.info("Evaluating Model C (YAMNet Transfer Learning)...")
    yamnet = hub.load(YAMNET_HUB_URL)
    classifier_head = keras.models.load_model(str(MODEL_C_HEAD_PATH))

    # Benchmark YAMNet embedding extraction latency
    single_wave = x_test_16k[0].astype(np.float32)
    # Warmup
    _, emb, _ = yamnet(single_wave)
    _ = classifier_head.predict(np.expand_dims(tf.reduce_mean(emb, axis=0).numpy(), axis=0), verbose=0)

    t_ext0 = time.perf_counter()
    for i in range(20):
        w = x_test_16k[i].astype(np.float32)
        _, emb, _ = yamnet(w)
        _ = tf.reduce_mean(emb, axis=0).numpy()
    yamnet_ms = ((time.perf_counter() - t_ext0) / 20) * 1000.0

    # Classifier head inference latency
    dummy_feat = np.random.randn(1, 1024).astype(np.float32)
    t_head0 = time.perf_counter()
    for _ in range(50):
        _ = classifier_head.predict(dummy_feat, verbose=0)
    head_ms = ((time.perf_counter() - t_head0) / 50) * 1000.0

    # Extract all embeddings for test set
    features_list = []
    for i in range(len(x_test_16k)):
        w = x_test_16k[i].astype(np.float32)
        max_abs = np.max(np.abs(w))
        if max_abs > 1.0:
            w = w / max_abs
        _, emb, _ = yamnet(w)
        features_list.append(tf.reduce_mean(emb, axis=0).numpy())

    features = np.array(features_list, dtype=np.float32)
    probs = classifier_head.predict(features, verbose=0)

    # Estimate YAMNet parameter count (~3.74M) + head (~140K)
    yamnet_params = 3740000
    head_params = int(classifier_head.count_params())
    total_params = yamnet_params + head_params

    # Size on disk: head + YAMNet SavedModel (~14.8 MB)
    head_size_kb = os.path.getsize(MODEL_C_HEAD_PATH) / 1024.0
    yamnet_size_kb = 14800.0  # TF Hub YAMNet weights size

    perf = {
        "params": total_params,
        "head_params": head_params,
        "file_size_kb": float(head_size_kb + yamnet_size_kb),
        "preprocessing_ms": float(yamnet_ms),
        "inference_ms": float(head_ms),
        "total_latency_ms": float(yamnet_ms + head_ms),
    }
    return probs, perf


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Compute all required metrics for security spray-can shaking detection."""
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y_true))
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))

    report = classification_report(
        y_true,
        preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Spray-Can Shaking (Target Class 5)
    target_idx = 5
    true_shaking_mask = (y_true == target_idx)
    non_shaking_mask = ~true_shaking_mask

    total_shaking = int(np.sum(true_shaking_mask))
    detected_shaking = int(np.sum((preds == target_idx) & true_shaking_mask))
    missed_shaking = total_shaking - detected_shaking
    miss_rate = (missed_shaking / total_shaking) if total_shaking > 0 else 0.0

    # Negative audio & False alarms
    total_non_shaking = int(np.sum(non_shaking_mask))
    false_alarms = int(np.sum((preds == target_idx) & non_shaking_mask))
    false_alarm_rate = (false_alarms / total_non_shaking) if total_non_shaking > 0 else 0.0

    # False Positives Per Hour:
    # Each test window is 1.0 second.
    # Total negative audio duration in hours:
    negative_duration_sec = total_non_shaking * 1.0
    negative_duration_hr = negative_duration_sec / 3600.0
    fp_per_hour = (false_alarms / negative_duration_hr) if negative_duration_hr > 0 else 0.0

    # Hard negative rattle specifically (Class 3)
    hard_neg_mask = (y_true == 3)
    total_hard_neg = int(np.sum(hard_neg_mask))
    hard_neg_fa = int(np.sum((preds == target_idx) & hard_neg_mask))
    hard_neg_fa_rate = (hard_neg_fa / total_hard_neg) if total_hard_neg > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "spray_shaking_precision": report["spray_can_shaking"]["precision"],
        "spray_shaking_recall": report["spray_can_shaking"]["recall"],
        "spray_shaking_f1": report["spray_can_shaking"]["f1-score"],
        "total_shaking_events": total_shaking,
        "detected_shaking_events": detected_shaking,
        "missed_shaking_events": missed_shaking,
        "missed_shaking_rate_pct": float(miss_rate * 100.0),
        "total_negative_events": total_non_shaking,
        "false_alarms_total": false_alarms,
        "false_alarm_rate_pct": float(false_alarm_rate * 100.0),
        "false_positives_per_hour": float(fp_per_hour),
        "hard_neg_rattle_total": total_hard_neg,
        "hard_neg_rattle_fa": hard_neg_fa,
        "hard_neg_rattle_fa_pct": float(hard_neg_fa_rate * 100.0),
    }


def main():
    logger.info("=== STEP 5: FAIR MODEL COMPARISON ON HELD-OUT TEST DATA ===")
    t0 = time.time()

    # Load held-out test data
    test_data = np.load(EXP_DATA_DIR / "expanded_test.npz")
    x_test_22k = test_data["audio_22050"]
    x_test_16k = test_data["audio_16000"]
    y_test = test_data["labels"]

    logger.info("Test set loaded: %d windows (1.0s each, %.2f min total)", len(y_test), len(y_test) / 60.0)

    # 1. Evaluate Model A
    probs_a, perf_a = benchmark_model_a(x_test_22k)
    metrics_a = compute_metrics(y_test, probs_a)

    # 2. Evaluate Model B
    probs_b, perf_b = benchmark_model_b(x_test_22k)
    metrics_b = compute_metrics(y_test, probs_b)

    # 3. Evaluate Model C
    probs_c, perf_c = benchmark_model_c(x_test_16k)
    metrics_c = compute_metrics(y_test, probs_c)

    # Plot individual confusion matrices
    plot_confusion_matrix(metrics_a["confusion_matrix"], "Model A (Current Production CNN)", RESULTS_DIR / "confusion_matrix_model_a.png")
    plot_confusion_matrix(metrics_b["confusion_matrix"], "Model B (Expanded-Data CNN)", RESULTS_DIR / "confusion_matrix_model_b.png")
    plot_confusion_matrix(metrics_c["confusion_matrix"], "Model C (YAMNet Transfer Learning)", RESULTS_DIR / "confusion_matrix_model_c.png")

    # Plot 3-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), dpi=150)
    models = [
        ("Model A: Production CNN", metrics_a["confusion_matrix"], axes[0]),
        ("Model B: Expanded CNN", metrics_b["confusion_matrix"], axes[1]),
        ("Model C: YAMNet Classifier", metrics_c["confusion_matrix"], axes[2]),
    ]
    for name, cm, ax in models:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            title=name,
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:d}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold" if i == j else "normal",
                )
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices_combined.png", bbox_inches="tight")
    plt.close()

    # Compile Summary Dictionary & CSV
    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_windows": len(y_test),
        "test_duration_minutes": float(len(y_test) / 60.0),
        "models": {
            "model_a_production_cnn": {
                "name": "Model A (Production CNN)",
                "accuracy": metrics_a["accuracy"],
                "macro_f1": metrics_a["macro_f1"],
                "spray_precision": metrics_a["spray_shaking_precision"],
                "spray_recall": metrics_a["spray_shaking_recall"],
                "spray_f1": metrics_a["spray_shaking_f1"],
                "missed_shaking_pct": metrics_a["missed_shaking_rate_pct"],
                "false_alarm_rate_pct": metrics_a["false_alarm_rate_pct"],
                "false_positives_per_hour": metrics_a["false_positives_per_hour"],
                "hard_negative_rattle_fa_pct": metrics_a["hard_neg_rattle_fa_pct"],
                "params": perf_a["params"],
                "file_size_kb": perf_a["file_size_kb"],
                "preprocessing_ms": perf_a["preprocessing_ms"],
                "inference_ms": perf_a["inference_ms"],
                "total_latency_ms": perf_a["total_latency_ms"],
            },
            "model_b_expanded_cnn": {
                "name": "Model B (Expanded-Data CNN)",
                "accuracy": metrics_b["accuracy"],
                "macro_f1": metrics_b["macro_f1"],
                "spray_precision": metrics_b["spray_shaking_precision"],
                "spray_recall": metrics_b["spray_shaking_recall"],
                "spray_f1": metrics_b["spray_shaking_f1"],
                "missed_shaking_pct": metrics_b["missed_shaking_rate_pct"],
                "false_alarm_rate_pct": metrics_b["false_alarm_rate_pct"],
                "false_positives_per_hour": metrics_b["false_positives_per_hour"],
                "hard_negative_rattle_fa_pct": metrics_b["hard_neg_rattle_fa_pct"],
                "params": perf_b["params"],
                "file_size_kb": perf_b["file_size_kb"],
                "preprocessing_ms": perf_b["preprocessing_ms"],
                "inference_ms": perf_b["inference_ms"],
                "total_latency_ms": perf_b["total_latency_ms"],
            },
            "model_c_yamnet_classifier": {
                "name": "Model C (YAMNet Transfer Learning)",
                "accuracy": metrics_c["accuracy"],
                "macro_f1": metrics_c["macro_f1"],
                "spray_precision": metrics_c["spray_shaking_precision"],
                "spray_recall": metrics_c["spray_shaking_recall"],
                "spray_f1": metrics_c["spray_shaking_f1"],
                "missed_shaking_pct": metrics_c["missed_shaking_rate_pct"],
                "false_alarm_rate_pct": metrics_c["false_alarm_rate_pct"],
                "false_positives_per_hour": metrics_c["false_positives_per_hour"],
                "hard_negative_rattle_fa_pct": metrics_c["hard_neg_rattle_fa_pct"],
                "params": perf_c["params"],
                "file_size_kb": perf_c["file_size_kb"],
                "preprocessing_ms": perf_c["preprocessing_ms"],
                "inference_ms": perf_c["inference_ms"],
                "total_latency_ms": perf_c["total_latency_ms"],
            },
        },
    }

    # Save JSON summary
    json_path = RESULTS_DIR / "metrics_summary.json"
    json_path.write_text(json.dumps(comparison_summary, indent=2))
    logger.info("Saved JSON metrics summary to %s", json_path)

    # Generate CSV summary
    csv_path = RESULTS_DIR / "metrics_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Metric,Model A (Production CNN),Model B (Expanded-Data CNN),Model C (YAMNet Transfer Learning)\n")
        f.write(f"Test Accuracy,{metrics_a['accuracy']*100:.2f}%,{metrics_b['accuracy']*100:.2f}%,{metrics_c['accuracy']*100:.2f}%\n")
        f.write(f"Macro F1-Score,{metrics_a['macro_f1']:.4f},{metrics_b['macro_f1']:.4f},{metrics_c['macro_f1']:.4f}\n")
        f.write(f"Shaking Precision,{metrics_a['spray_shaking_precision']*100:.2f}%,{metrics_b['spray_shaking_precision']*100:.2f}%,{metrics_c['spray_shaking_precision']*100:.2f}%\n")
        f.write(f"Shaking Recall,{metrics_a['spray_shaking_recall']*100:.2f}%,{metrics_b['spray_shaking_recall']*100:.2f}%,{metrics_c['spray_shaking_recall']*100:.2f}%\n")
        f.write(f"Shaking F1-Score,{metrics_a['spray_shaking_f1']:.4f},{metrics_b['spray_shaking_f1']:.4f},{metrics_c['spray_shaking_f1']:.4f}\n")
        f.write(f"Missed Shaking Rate,{metrics_a['missed_shaking_rate_pct']:.2f}%,{metrics_b['missed_shaking_rate_pct']:.2f}%,{metrics_c['missed_shaking_rate_pct']:.2f}%\n")
        f.write(f"False Alarm Rate (Negatives),{metrics_a['false_alarm_rate_pct']:.2f}%,{metrics_b['false_alarm_rate_pct']:.2f}%,{metrics_c['false_alarm_rate_pct']:.2f}%\n")
        f.write(f"False Positives Per Hour,{metrics_a['false_positives_per_hour']:.1f} FP/hr,{metrics_b['false_positives_per_hour']:.1f} FP/hr,{metrics_c['false_positives_per_hour']:.1f} FP/hr\n")
        f.write(f"Hard Negative Rattle False Alarm Rate,{metrics_a['hard_neg_rattle_fa_pct']:.2f}%,{metrics_b['hard_neg_rattle_fa_pct']:.2f}%,{metrics_c['hard_neg_rattle_fa_pct']:.2f}%\n")
        f.write(f"Total Parameters,{perf_a['params']:,},{perf_b['params']:,},{perf_c['params']:,}\n")
        f.write(f"Model Disk Size,{perf_a['file_size_kb']:.1f} KB,{perf_b['file_size_kb']:.1f} KB,{perf_c['file_size_kb']/1024.0:.2f} MB\n")
        f.write(f"Audio Preprocessing Time,{perf_a['preprocessing_ms']:.2f} ms,{perf_b['preprocessing_ms']:.2f} ms,{perf_c['preprocessing_ms']:.2f} ms\n")
        f.write(f"Inference Latency,{perf_a['inference_ms']:.2f} ms,{perf_b['inference_ms']:.2f} ms,{perf_c['inference_ms']:.2f} ms\n")
        f.write(f"Total Latency per 1.0s Window,{perf_a['total_latency_ms']:.2f} ms,{perf_b['total_latency_ms']:.2f} ms,{perf_c['total_latency_ms']:.2f} ms\n")

    logger.info("Saved CSV metrics summary to %s", csv_path)

    # Print beautiful comparison table to stdout
    print("\n" + "=" * 90)
    print("FAIR BENCHMARK COMPARISON ON HELD-OUT TEST DATA (469 Audio Windows)")
    print("=" * 90)
    header = f"{'Metric':<36} | {'Model A (Production)':<18} | {'Model B (Expanded)':<18} | {'Model C (YAMNet)':<18}"
    print(header)
    print("-" * 90)
    print(f"{'Overall Accuracy':<36} | {metrics_a['accuracy']*100:16.2f}% | {metrics_b['accuracy']*100:16.2f}% | {metrics_c['accuracy']*100:16.2f}%")
    print(f"{'Macro F1-Score':<36} | {metrics_a['macro_f1']:17.4f} | {metrics_b['macro_f1']:17.4f} | {metrics_c['macro_f1']:17.4f}")
    print(f"{'Shaking Precision':<36} | {metrics_a['spray_shaking_precision']*100:16.2f}% | {metrics_b['spray_shaking_precision']*100:16.2f}% | {metrics_c['spray_shaking_precision']*100:16.2f}%")
    print(f"{'Shaking Recall':<36} | {metrics_a['spray_shaking_recall']*100:16.2f}% | {metrics_b['spray_shaking_recall']*100:16.2f}% | {metrics_c['spray_shaking_recall']*100:16.2f}%")
    print(f"{'Shaking F1-Score':<36} | {metrics_a['spray_shaking_f1']:17.4f} | {metrics_b['spray_shaking_f1']:17.4f} | {metrics_c['spray_shaking_f1']:17.4f}")
    print(f"{'Missed Shaking Rate (%)':<36} | {metrics_a['missed_shaking_rate_pct']:16.2f}% | {metrics_b['missed_shaking_rate_pct']:16.2f}% | {metrics_c['missed_shaking_rate_pct']:16.2f}%")
    print(f"{'False Alarm Rate on Negatives (%)':<36} | {metrics_a['false_alarm_rate_pct']:16.2f}% | {metrics_b['false_alarm_rate_pct']:16.2f}% | {metrics_c['false_alarm_rate_pct']:16.2f}%")
    print(f"{'False Positives Per Hour':<36} | {metrics_a['false_positives_per_hour']:12.1f} FP/hr | {metrics_b['false_positives_per_hour']:12.1f} FP/hr | {metrics_c['false_positives_per_hour']:12.1f} FP/hr")
    print(f"{'Hard Negative Rattle False Alarms':<36} | {metrics_a['hard_neg_rattle_fa_pct']:16.2f}% | {metrics_b['hard_neg_rattle_fa_pct']:16.2f}% | {metrics_c['hard_neg_rattle_fa_pct']:16.2f}%")
    print("-" * 90)
    print(f"{'Total Parameters':<36} | {perf_a['params']:17,d} | {perf_b['params']:17,d} | {perf_c['params']:17,d}")
    print(f"{'Model Size on Disk':<36} | {perf_a['file_size_kb']:14.1f} KB | {perf_b['file_size_kb']:14.1f} KB | {perf_c['file_size_kb']/1024.0:14.2f} MB")
    print(f"{'Audio Preprocessing Time':<36} | {perf_a['preprocessing_ms']:14.2f} ms | {perf_b['preprocessing_ms']:14.2f} ms | {perf_c['preprocessing_ms']:14.2f} ms")
    print(f"{'Inference Latency (Desktop CPU)':<36} | {perf_a['inference_ms']:14.2f} ms | {perf_b['inference_ms']:14.2f} ms | {perf_c['inference_ms']:14.2f} ms")
    print(f"{'Total Window Latency':<36} | {perf_a['total_latency_ms']:14.2f} ms | {perf_b['total_latency_ms']:14.2f} ms | {perf_c['total_latency_ms']:14.2f} ms")
    print("=" * 90)

    elapsed = time.time() - t0
    logger.info("Step 5 evaluation completed in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
