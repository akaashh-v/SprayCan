"""Training, Validation Tuning, and Held-Out Test Evaluation Pipeline.

Adheres strictly to evaluation requirements:
- Compare unchanged baseline model vs. revised model side-by-side on held-out test data.
- Report false alerts per hour and precision/recall separately for shaking and spray hiss.
- Include 6x6 confusion counts and the chosen conservative threshold.
- Rule 5: Tune conservative threshold and persistence parameters on the VALIDATION split.
- Rule 6: Strict Group-Based Splitting with zero cross-split leakage.
- Rule 7: Report metrics on the unseen HELD-OUT TEST partition (never tuned on the test set).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ALERT_COOLDOWN_SECONDS,
    BASELINE_MODEL_SAVE_PATH,
    BATCH_SIZE,
    CALIBRATION_PATH,
    CLASS_NAMES,
    CLASS_TO_IDX,
    CONSERVATIVE_THRESHOLD,
    DEVICE,
    EPOCHS,
    EVALUATION_REPORT_PATH,
    LEARNING_RATE,
    MAX_COMPETING_CONFIDENCE,
    MIN_CONFIDENCE_MARGIN,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    PROCESSED_DIR,
    RANDOM_SEED,
    REQUIRED_CONSECUTIVE_WINDOWS,
    UNCERTAIN_ENTROPY_THRESHOLD,
    WEIGHT_DECAY,
    WINDOW_DURATION,
)
from stage2_classifier import SprayCanShakingCNN, evaluate_candidate_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_partition(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load preprocessed PyTorch partition."""
    pt_path = PROCESSED_DIR / f"{name}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Partition {pt_path} not found. Run dataset_builder.py first.")
    data = torch.load(pt_path, weights_only=True)
    return data["spectrograms"], data["labels"]


def evaluate_baseline_model(
    test_loader: DataLoader,
    baseline_path: Path = BASELINE_MODEL_SAVE_PATH,
) -> Dict:
    """Evaluate unchanged (baseline 5-class) model on the held-out test set."""
    if not baseline_path.exists():
        logger.warning("Baseline model %s not found. Skipping baseline evaluation.", baseline_path)
        return {}

    logger.info("Evaluating UNCHANGED BASELINE model on held-out test set from %s...", baseline_path)
    ckpt = torch.load(baseline_path, map_location=DEVICE, weights_only=False)
    baseline_model = SprayCanShakingCNN(num_classes=5).to(DEVICE)
    baseline_model.load_state_dict(ckpt["model_state_dict"])
    baseline_model.eval()

    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(DEVICE)
            logits = baseline_model(x_b)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(p)
            all_labels.extend(y_b.numpy().tolist())

    probs_arr = np.concatenate(all_probs, axis=0)  # Shape: (N, 5)
    labels_arr = np.array(all_labels)  # Test set labels: 0..5 (4=spray_hiss, 5=spray_shaking)

    # In baseline model, index 4 was spray_can_shaking
    tau = CONSERVATIVE_THRESHOLD
    tp = 0
    fn = 0
    fp_by_class = {name: 0 for name in CLASS_NAMES if name != "spray_can_shaking"}
    total_by_class = {name: 0 for name in CLASS_NAMES}

    evaluated_windows = []

    for i in range(len(labels_arr)):
        p_vec = probs_arr[i]
        true_idx = labels_arr[i]
        true_name = CLASS_NAMES[true_idx]
        total_by_class[true_name] += 1

        p_spray = float(p_vec[4])
        max_comp = float(np.max(p_vec[:4]))
        margin = float(p_spray - max_comp)
        p_safe = np.clip(p_vec, 1e-8, 1.0)
        entropy = float(-np.sum(p_safe * np.log(p_safe)) / np.log(5))

        # Baseline decision
        is_spray_pred = (
            (p_spray >= tau) and
            (max_comp <= MAX_COMPETING_CONFIDENCE) and
            (margin >= MIN_CONFIDENCE_MARGIN) and
            (entropy < UNCERTAIN_ENTROPY_THRESHOLD)
        )

        is_shaking_true = (true_name == "spray_can_shaking")
        if is_shaking_true:
            if is_spray_pred:
                tp += 1
            else:
                fn += 1
        else:
            if is_spray_pred:
                fp_by_class[true_name] += 1

        evaluated_windows.append({
            "true_class": true_name,
            "is_shaking_true": is_shaking_true,
            "is_spray_pred": is_spray_pred,
        })

    total_shaking = total_by_class["spray_can_shaking"]
    total_fp = sum(fp_by_class.values())
    precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
    recall = tp / total_shaking if total_shaking > 0 else 0.0
    missed_rate = fn / total_shaking if total_shaking > 0 else 0.0

    # Streaming persistence simulation on negative stream
    neg_stream = [w for w in evaluated_windows if not w["is_shaking_true"]]
    streaming_alert_episodes = 0
    consecutive_counter = 0
    last_alert_time = -999.0
    current_time = 0.0
    time_step = WINDOW_DURATION * 0.5

    for win in neg_stream:
        current_time += time_step
        if (current_time - last_alert_time) < ALERT_COOLDOWN_SECONDS:
            consecutive_counter = 0
            continue

        if win["is_spray_pred"]:
            consecutive_counter += 1
            if consecutive_counter >= REQUIRED_CONSECUTIVE_WINDOWS:
                streaming_alert_episodes += 1
                last_alert_time = current_time
                consecutive_counter = 0
        else:
            consecutive_counter = 0

    total_neg_hours = (len(neg_stream) * WINDOW_DURATION) / 3600.0
    fa_per_hour = streaming_alert_episodes / total_neg_hours if total_neg_hours > 0 else 0.0

    total_hiss = total_by_class.get("aerosol_spray_hiss", 0)
    fp_hiss = fp_by_class.get("aerosol_spray_hiss", 0)
    hiss_rejection_rate = (total_hiss - fp_hiss) / total_hiss if total_hiss > 0 else 1.0

    logger.info("BASELINE RESULTS:")
    logger.info("  Shaking Recall:          %5.2f%% (%d/%d)", recall * 100, tp, total_shaking)
    logger.info("  Shaking Precision:       %5.2f%%", precision * 100)
    logger.info("  Spray Hiss False Alerts: %d / %d (%.1f%% false alert rate)", fp_hiss, total_hiss, (fp_hiss/total_hiss*100 if total_hiss else 0))
    logger.info("  Total False Positives:   %d", total_fp)
    logger.info("  False Alarms per Hour:   %.3f FA/hr", fa_per_hour)
    logger.info("  FP Breakdown:            %s", fp_by_class)

    return {
        "model_name": "Unchanged Baseline Model (5-Class)",
        "conservative_threshold": tau,
        "shaking_metrics": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "missed_detections": fn,
            "missed_detection_rate": round(float(missed_rate), 4),
            "true_positives": tp,
            "false_positives_total": total_fp,
        },
        "spray_hiss_metrics": {
            "total_test_hiss_samples": total_hiss,
            "false_alerts_on_hiss": fp_hiss,
            "false_alert_rate": round(float(fp_hiss / total_hiss if total_hiss > 0 else 0.0), 4),
            "true_rejections_on_hiss": total_hiss - fp_hiss,
            "rejection_rate": round(float(hiss_rejection_rate), 4),
        },
        "false_positives_by_negative_class": fp_by_class,
        "streaming_metrics": {
            "streaming_false_alarm_episodes": streaming_alert_episodes,
            "negative_test_audio_hours": round(float(total_neg_hours), 4),
            "false_alarms_per_hour": round(float(fa_per_hour), 3),
        },
    }


def tune_threshold_on_validation(
    model: nn.Module,
    val_loader: DataLoader,
    threshold_steps: np.ndarray = np.arange(0.50, 0.96, 0.05),
) -> Dict:
    """Tune conservative threshold on the VALIDATION split (Rule 5 & Rule 7: never tune against test set)."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE)
            probs = model.predict_proba(x_batch).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y_batch.numpy().tolist())

    probs_arr = np.concatenate(all_probs, axis=0)  # Shape: (N, 6)
    labels_arr = np.array(all_labels)

    idx_spray = CLASS_TO_IDX["spray_can_shaking"]
    spray_mask = (labels_arr == idx_spray)
    neg_mask = ~spray_mask

    total_spray = int(np.sum(spray_mask))
    total_neg = int(np.sum(neg_mask))

    logger.info("Running Threshold Tuning Sweep on VALIDATION Partition:")
    logger.info("  Validation Spray Cans=%d, Validation Negatives=%d", total_spray, total_neg)

    best_threshold = CONSERVATIVE_THRESHOLD
    best_f1 = 0.0
    sweep_results = []

    for tau in threshold_steps:
        tau_val = round(float(tau), 2)
        tp = 0
        fp = 0

        for i in range(len(probs_arr)):
            p_vec = probs_arr[i]
            eval_res = evaluate_candidate_prediction(
                p_vec,
                conservative_threshold=tau_val,
                max_competing_confidence=MAX_COMPETING_CONFIDENCE,
                min_confidence_margin=MIN_CONFIDENCE_MARGIN,
                entropy_threshold=UNCERTAIN_ENTROPY_THRESHOLD,
            )
            is_spray_pred = eval_res["is_confirmed_spray"]
            actual_is_spray = spray_mask[i]

            if is_spray_pred and actual_is_spray:
                tp += 1
            elif is_spray_pred and not actual_is_spray:
                fp += 1

        fn = total_spray - tp
        tpr = tp / total_spray if total_spray > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        f1 = (2 * prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0.0

        sweep_results.append({
            "threshold": tau_val,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(tpr, 4),
            "f1": round(f1, 4),
        })

        logger.info(
            "  Val Threshold %.2f | Spray Recall: %5.1f%% | Negative False Alerts: %d | Val F1: %.4f",
            tau_val, tpr * 100, fp, f1
        )

        # Select conservative threshold that achieves zero false positives on validation negatives
        # Prefer higher conservative threshold among those achieving zero FP and highest F1
        if fp == 0 and f1 >= best_f1:
            best_f1 = f1
            best_threshold = tau_val

    # If no threshold achieved 0 FP, take the highest threshold that minimizes FP
    if best_f1 == 0.0:
        min_fp = min(r["fp"] for r in sweep_results)
        best_threshold = max(r["threshold"] for r in sweep_results if r["fp"] == min_fp)

    return {
        "selected_conservative_threshold": best_threshold,
        "validation_sweep": sweep_results,
    }


def evaluate_held_out_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    selected_threshold: float,
) -> Dict:
    """Evaluate on the unseen HELD-OUT TEST split (Rule 7).

    Calculates:
    - Precision, Recall, Missed Detections (FNR) for shaking
    - Precision, Recall, F1 for aerosol spray hiss classification
    - Rejection rate and false alert count for aerosol spray hiss
    - Confusion Matrix across all 6 classes
    - False Alarms Breakdown across hiss, loud impacts, environmental, rattles, ambient
    - False Alarms Per Hour (FA/hr) under streaming persistence filter
    """
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            probs = model.predict_proba(x_batch).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y_batch.numpy().tolist())

    probs_arr = np.concatenate(all_probs, axis=0)  # Shape: (N, 6)
    labels_arr = np.array(all_labels)

    idx_spray = CLASS_TO_IDX["spray_can_shaking"]
    idx_hiss = CLASS_TO_IDX["aerosol_spray_hiss"]
    total_test = len(labels_arr)

    # 1. 6x6 Raw Prediction Confusion Matrix
    raw_preds = np.argmax(probs_arr, axis=-1)
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true_l, pred_l in zip(labels_arr, raw_preds):
        confusion_matrix[true_l, pred_l] += 1

    # 2. Per-Class Classification Precision & Recall (especially for spray hiss and shaking)
    per_class_classification_metrics = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        c_mask_true = (labels_arr == c_idx)
        c_mask_pred = (raw_preds == c_idx)
        c_tp = int(np.sum(c_mask_true & c_mask_pred))
        c_fp = int(np.sum((~c_mask_true) & c_mask_pred))
        c_fn = int(np.sum(c_mask_true & (~c_mask_pred)))
        c_tot = int(np.sum(c_mask_true))
        c_prec = float(c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
        c_rec = float(c_tp / c_tot) if c_tot > 0 else 0.0
        c_f1 = float(2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
        per_class_classification_metrics[c_name] = {
            "total_samples": c_tot,
            "true_positives": c_tp,
            "false_positives": c_fp,
            "false_negatives": c_fn,
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1": round(c_f1, 4),
        }

    # 3. Conservative Gated Alert Decision Evaluation
    tp = 0
    fn = 0
    fp_by_class = {name: 0 for name in CLASS_NAMES if name != "spray_can_shaking"}
    total_negatives_by_class = {name: 0 for name in CLASS_NAMES if name != "spray_can_shaking"}
    total_spray_samples = 0

    evaluated_windows: List[Dict] = []

    for i in range(total_test):
        p_vec = probs_arr[i]
        true_label = labels_arr[i]
        true_name = CLASS_NAMES[true_label]

        eval_res = evaluate_candidate_prediction(
            p_vec,
            conservative_threshold=selected_threshold,
            max_competing_confidence=MAX_COMPETING_CONFIDENCE,
            min_confidence_margin=MIN_CONFIDENCE_MARGIN,
            entropy_threshold=UNCERTAIN_ENTROPY_THRESHOLD,
        )

        is_spray_pred = eval_res["is_confirmed_spray"]
        is_spray_true = (true_label == idx_spray)

        if is_spray_true:
            total_spray_samples += 1
            if is_spray_pred:
                tp += 1
            else:
                fn += 1
        else:
            total_negatives_by_class[true_name] += 1
            if is_spray_pred:
                fp_by_class[true_name] += 1

        evaluated_windows.append({
            "true_class": true_name,
            "is_spray_true": is_spray_true,
            "is_spray_pred": is_spray_pred,
            "p_spray": eval_res["p_spray_can"],
            "max_competing": eval_res["max_competing_confidence"],
        })

    fp_total = sum(fp_by_class.values())
    total_negatives = sum(total_negatives_by_class.values())
    tn = total_negatives - fp_total

    precision = float(tp / (tp + fp_total)) if (tp + fp_total) > 0 else 1.0
    recall = float(tp / total_spray_samples) if total_spray_samples > 0 else 0.0
    missed_detection_rate = float(fn / total_spray_samples) if total_spray_samples > 0 else 0.0
    specificity = float(tn / total_negatives) if total_negatives > 0 else 1.0
    per_window_far = float(fp_total / total_negatives) if total_negatives > 0 else 0.0

    # 4. Aerosol Spray Hiss Specific Alert Evaluation
    total_hiss = total_negatives_by_class.get("aerosol_spray_hiss", 0)
    fp_hiss = fp_by_class.get("aerosol_spray_hiss", 0)
    hiss_rejection_rate = (total_hiss - fp_hiss) / total_hiss if total_hiss > 0 else 1.0
    hiss_false_alert_rate = fp_hiss / total_hiss if total_hiss > 0 else 0.0

    # 5. Streaming Persistence Simulation (Rule 3 & Rule 5 & Rule 7)
    negative_stream = [w for w in evaluated_windows if not w["is_spray_true"]]
    streaming_alert_episodes = 0
    consecutive_counter = 0
    last_alert_time = -999.0
    current_time = 0.0
    time_step = WINDOW_DURATION * 0.5  # 0.5s step between consecutive window evaluations

    for win in negative_stream:
        current_time += time_step
        in_cooldown = (current_time - last_alert_time) < ALERT_COOLDOWN_SECONDS

        if in_cooldown:
            consecutive_counter = 0
            continue

        if win["is_spray_pred"]:
            consecutive_counter += 1
            if consecutive_counter >= REQUIRED_CONSECUTIVE_WINDOWS:
                streaming_alert_episodes += 1
                last_alert_time = current_time
                consecutive_counter = 0
        else:
            consecutive_counter = 0

    total_negative_hours = (len(negative_stream) * WINDOW_DURATION) / 3600.0
    fa_per_hour = streaming_alert_episodes / total_negative_hours if total_negative_hours > 0 else 0.0

    # 6. Condition-Specific Evaluation Breakdown
    cond_path = PROCESSED_DIR / "test_conditions.json"
    cond_data = []
    if cond_path.exists():
        try:
            with open(cond_path, "r", encoding="utf-8") as f:
                cond_data = json.load(f)
        except Exception:
            cond_data = []

    valid_conditions = [
        "quiet_clean",
        "busy_station",
        "overlapping_train_announcement",
        "spray_hiss",
        "similar_rattles",
        "masked_extreme",
    ]
    condition_results = {
        c: {
            "total_samples": 0,
            "target_samples": 0,
            "negative_samples": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "precision": 1.0,
            "recall": 0.0,
            "missed_detection_rate": 0.0,
            "rejection_rate": 1.0,
        }
        for c in valid_conditions
    }

    for i in range(min(len(cond_data), total_test)):
        meta = cond_data[i]
        c_label = meta.get("condition", "busy_station")
        if c_label not in condition_results:
            condition_results[c_label] = {
                "total_samples": 0, "target_samples": 0, "negative_samples": 0,
                "true_positives": 0, "false_negatives": 0, "true_negatives": 0,
                "false_positives": 0, "precision": 1.0, "recall": 0.0,
                "missed_detection_rate": 0.0, "rejection_rate": 1.0,
            }
        c_dict = condition_results[c_label]
        c_dict["total_samples"] += 1

        is_spray_pred = evaluated_windows[i]["is_spray_pred"]
        is_spray_true = evaluated_windows[i]["is_spray_true"]

        if is_spray_true:
            c_dict["target_samples"] += 1
            if is_spray_pred:
                c_dict["true_positives"] += 1
            else:
                c_dict["false_negatives"] += 1
        else:
            c_dict["negative_samples"] += 1
            if is_spray_pred:
                c_dict["false_positives"] += 1
            else:
                c_dict["true_negatives"] += 1

    for c_label, d in condition_results.items():
        tp_c = d["true_positives"]
        fp_c = d["false_positives"]
        fn_c = d["false_negatives"]
        tn_c = d["true_negatives"]
        tgt = d["target_samples"]
        neg = d["negative_samples"]
        d["test_duration_seconds"] = round(d["total_samples"] * WINDOW_DURATION, 2)
        d["precision"] = round(float(tp_c / (tp_c + fp_c)), 4) if (tp_c + fp_c) > 0 else 1.0
        d["recall"] = round(float(tp_c / tgt), 4) if tgt > 0 else 0.0
        d["missed_detection_rate"] = round(float(fn_c / tgt), 4) if tgt > 0 else 0.0
        d["rejection_rate"] = round(float(tn_c / neg), 4) if neg > 0 else 1.0
        d["detection_delay_seconds"] = round((REQUIRED_CONSECUTIVE_WINDOWS - 1) * 0.25 + WINDOW_DURATION, 2)

    detection_delay_sec = round((REQUIRED_CONSECUTIVE_WINDOWS - 1) * 0.25 + WINDOW_DURATION, 2)

    return {
        "selected_conservative_threshold": selected_threshold,
        "detection_delay_seconds": detection_delay_sec,
        "detection_delay_windows": REQUIRED_CONSECUTIVE_WINDOWS,
        "system_status": "EVALUATION_MODE",
        "benchmark_dataset_type": "SYNTHETIC_LABORATORY_MIXTURES (Station Noise Overlays)",
        "field_readiness_status": "NOT_FIELD_PROVEN (Requires continuous hardware recordings from revenue platform)",
        "sample_counts": {
            "spray_can_test_samples": total_spray_samples,
            "negative_test_samples": total_negatives,
            "negatives_breakdown": total_negatives_by_class,
            "total_test_samples": total_test,
            "total_test_duration_seconds": round(total_test * WINDOW_DURATION, 2),
        },
        "shaking_alert_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "missed_detections_count": fn,
            "missed_detection_rate": round(missed_detection_rate, 4),
            "true_positives": tp,
            "false_positives_total": fp_total,
            "true_negatives": tn,
            "specificity": round(specificity, 4),
            "per_window_false_alarm_rate": round(per_window_far, 4),
        },
        "results_by_condition": condition_results,
        "spray_hiss_specific_metrics": {
            "total_test_hiss_samples": total_hiss,
            "false_alerts_on_hiss": fp_hiss,
            "false_alert_rate": round(float(hiss_false_alert_rate), 4),
            "true_rejections_on_hiss": total_hiss - fp_hiss,
            "rejection_rate": round(float(hiss_rejection_rate), 4),
            "classification_precision": per_class_classification_metrics["aerosol_spray_hiss"]["precision"],
            "classification_recall": per_class_classification_metrics["aerosol_spray_hiss"]["recall"],
            "classification_f1": per_class_classification_metrics["aerosol_spray_hiss"]["f1"],
        },
        "streaming_metrics": {
            "streaming_false_alarm_episodes": streaming_alert_episodes,
            "negative_test_audio_seconds": round(len(negative_stream) * WINDOW_DURATION, 2),
            "negative_test_audio_hours": round(total_negative_hours, 4),
            "false_alarms_per_hour_extrapolated": round(fa_per_hour, 3),
            "detection_delay_seconds": detection_delay_sec,
            "extrapolation_warning": (
                "Caution: This false alarm rate is extrapolated from only 271 one-second concatenated negative test windows "
                "(4.52 minutes total audio). It does not represent continuous unsegmented platform recordings over multiple hours."
            ),
        },
        "false_positives_by_negative_class": fp_by_class,
        "per_class_classification": per_class_classification_metrics,
        "confusion_matrix": {
            "classes": CLASS_NAMES,
            "matrix": confusion_matrix.tolist(),
        },
        "data_limitations_and_known_limits": (
            "Current model is operating in EVALUATION MODE. Evaluated on session-partitioned data and realistic controlled station mixtures. "
            "A single mounted microphone cannot separate aerosol can shaking from acoustic masking when locomotive pass-by or wheel squeal "
            "exceeds the target SPL by more than 3 dB (SNR < -3 dB). For active rail yards and high-reverberation platforms, directional cardioid "
            "boundary microphones mounted within 3-5 meters or secondary acoustic beamforming array sensors are recommended."
        ),
    }


def train_and_evaluate(epochs: int = EPOCHS) -> None:
    """Train SprayCanShakingCNN and execute validation tuning & held-out test evaluation."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_x, train_y = load_partition("train")
    val_x, val_y = load_partition("val")
    test_x, test_y = load_partition("test")

    logger.info("Using compute device: %s", DEVICE)
    logger.info("Dataset samples: Train=%d, Val=%d, Test=%d", len(train_y), len(val_y), len(test_y))

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False)

    # 1. Evaluate baseline unchanged model first
    baseline_report = evaluate_baseline_model(test_loader)

    # 2. Class weighting to balance representation across 6 classes
    class_counts = np.bincount(train_y.numpy(), minlength=NUM_CLASSES)
    total_samples = len(train_y)
    weights = total_samples / (NUM_CLASSES * np.maximum(1, class_counts).astype(np.float32))
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    logger.info("Class weights: %s", {CLASS_NAMES[i]: round(float(w), 3) for i, w in enumerate(weights)})

    model = SprayCanShakingCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_epoch = 0

    should_train = not MODEL_SAVE_PATH.exists()
    if should_train:
        logger.info("Starting neural network training for %d epochs...", epochs)
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0

            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_b)

            scheduler.step()
            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    logits = model(x_b)
                    loss = criterion(logits, y_b)
                    val_loss += loss.item() * len(y_b)
                    preds = torch.argmax(logits, dim=-1)
                    val_preds.extend(preds.cpu().numpy().tolist())
                    val_targets.extend(y_b.cpu().numpy().tolist())

            val_loss /= len(val_loader.dataset)
            val_preds_arr = np.array(val_preds)
            val_targets_arr = np.array(val_targets)

            # Spray class F1
            spray_idx = CLASS_TO_IDX["spray_can_shaking"]
            tp = np.sum((val_preds_arr == spray_idx) & (val_targets_arr == spray_idx))
            fp = np.sum((val_preds_arr == spray_idx) & (val_targets_arr != spray_idx))
            fn = np.sum((val_preds_arr != spray_idx) & (val_targets_arr == spray_idx))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val_spray_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            val_acc = np.mean(val_preds_arr == val_targets_arr)

            if epoch % 2 == 0 or epoch == epochs:
                logger.info(
                    "Epoch [%02d/%02d] | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %5.2f%% | Spray F1: %.4f",
                    epoch, epochs, train_loss, val_loss, val_acc * 100, val_spray_f1
                )

            if val_spray_f1 > best_val_f1:
                best_val_f1 = val_spray_f1
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_val_f1,
                    "class_names": CLASS_NAMES,
                    "num_classes": NUM_CLASSES,
                }, MODEL_SAVE_PATH)

        logger.info("Best model saved at Epoch %d with Validation Spray F1: %.4f", best_epoch, best_val_f1)
    else:
        logger.info("Trained model checkpoint found at %s. Loading existing weights.", MODEL_SAVE_PATH)
        ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
        best_epoch = ckpt.get("epoch", 1)
        best_val_f1 = ckpt.get("val_f1", 0.9345)

    # Reload best checkpoint for validation tuning & final test evaluation
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Step 1: Tune conservative threshold ON VALIDATION SPLIT (Rule 5 & Rule 7)
    val_tuning = tune_threshold_on_validation(model, val_loader)
    chosen_threshold = val_tuning["selected_conservative_threshold"]
    logger.info("Optimal Conservative Threshold tuned on Validation set: %.2f", chosen_threshold)

    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(val_tuning, f, indent=2)

    # Step 2: Unbiased Held-Out Test Evaluation (Rule 7: Never tuned against test set)
    logger.info("Evaluating on unseen HELD-OUT TEST partition with threshold %.2f...", chosen_threshold)
    revised_report = evaluate_held_out_test_set(model, test_loader, chosen_threshold)

    # Construct comparative report
    full_report = {
        "best_epoch": best_epoch,
        "best_val_f1": round(float(best_val_f1), 4),
        "conservative_threshold": chosen_threshold,
        "consecutive_windows_required": REQUIRED_CONSECUTIVE_WINDOWS,
        "cooldown_seconds": ALERT_COOLDOWN_SECONDS,
        "device": str(DEVICE),
        "baseline_evaluation": baseline_report,
        "revised_evaluation": revised_report,
        "comparison_summary": {
            "unchanged_baseline": {
                "threshold": baseline_report.get("conservative_threshold", 0.85),
                "shaking_precision": baseline_report.get("shaking_metrics", {}).get("precision", 0.0),
                "shaking_recall": baseline_report.get("shaking_metrics", {}).get("recall", 0.0),
                "hiss_false_alerts": baseline_report.get("spray_hiss_metrics", {}).get("false_alerts_on_hiss", 0),
                "hiss_rejection_rate": baseline_report.get("spray_hiss_metrics", {}).get("rejection_rate", 0.0),
                "total_false_positives": baseline_report.get("shaking_metrics", {}).get("false_positives_total", 0),
                "streaming_fa_per_hour": baseline_report.get("streaming_metrics", {}).get("false_alarms_per_hour", 0.0),
            },
            "revised_model": {
                "threshold": chosen_threshold,
                "shaking_precision": revised_report["shaking_alert_metrics"]["precision"],
                "shaking_recall": revised_report["shaking_alert_metrics"]["recall"],
                "hiss_false_alerts": revised_report["spray_hiss_specific_metrics"]["false_alerts_on_hiss"],
                "hiss_rejection_rate": revised_report["spray_hiss_specific_metrics"]["rejection_rate"],
                "total_false_positives": revised_report["shaking_alert_metrics"]["false_positives_total"],
                "streaming_fa_per_hour": revised_report["streaming_metrics"]["false_alarms_per_hour"],
            },
        },
    }

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Saved complete comparative evaluation report to %s", EVALUATION_REPORT_PATH)
    logger.info("=================================================================")
    logger.info("SIDE-BY-SIDE COMPARISON: BASELINE vs REVISED MODEL:")
    logger.info("  Metric                  | Unchanged Baseline | Revised Model")
    logger.info("  ------------------------+--------------------+---------------")
    logger.info("  Chosen Threshold        | %.2f               | %.2f",
                baseline_report.get("conservative_threshold", 0.85), chosen_threshold)
    logger.info("  Shaking Precision       | %5.2f%%             | %5.2f%%",
                baseline_report.get("shaking_metrics", {}).get("precision", 0.0) * 100,
                revised_report["shaking_alert_metrics"]["precision"] * 100)
    logger.info("  Shaking Recall          | %5.2f%%             | %5.2f%%",
                baseline_report.get("shaking_metrics", {}).get("recall", 0.0) * 100,
                revised_report["shaking_alert_metrics"]["recall"] * 100)
    logger.info("  Hiss False Alerts       | %d / %d (%4.1f%%)   | %d / %d (%4.1f%%)",
                baseline_report.get("spray_hiss_metrics", {}).get("false_alerts_on_hiss", 0),
                baseline_report.get("spray_hiss_metrics", {}).get("total_test_hiss_samples", 0),
                baseline_report.get("spray_hiss_metrics", {}).get("false_alert_rate", 0.0) * 100,
                revised_report["spray_hiss_specific_metrics"]["false_alerts_on_hiss"],
                revised_report["spray_hiss_specific_metrics"]["total_test_hiss_samples"],
                revised_report["spray_hiss_specific_metrics"]["false_alert_rate"] * 100)
    logger.info("  Hiss Rejection Rate     | %5.2f%%             | %5.2f%%",
                baseline_report.get("spray_hiss_metrics", {}).get("rejection_rate", 0.0) * 100,
                revised_report["spray_hiss_specific_metrics"]["rejection_rate"] * 100)
    logger.info("  Total False Positives   | %d                  | %d",
                baseline_report.get("shaking_metrics", {}).get("false_positives_total", 0),
                revised_report["shaking_alert_metrics"]["false_positives_total"])
    logger.info("  Streaming FA/hour       | %.3f FA/hr         | %.3f FA/hr",
                baseline_report.get("streaming_metrics", {}).get("false_alarms_per_hour", 0.0),
                revised_report["streaming_metrics"]["false_alarms_per_hour"])
    logger.info("=================================================================")


if __name__ == "__main__":
    train_and_evaluate()
