"""Canonical Evaluation Script for Spray Can Sound Detector.

Executes a single, unified, reproducible evaluation run across:
1. Validation Partition (data/processed/val.pt): 506 samples (238 spray targets, 268 negatives)
2. Held-Out Test Partition (data/processed/test.pt): 1006 samples (735 spray targets, 271 negatives)

Measures:
- Per-window Multi-Gated Pipeline metrics across thresholds tau in [0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
- Full Confusion Matrix (TP, FP, FN, TN) for every threshold
- False positives breakdown by specific negative class
- Synthetic mixture condition breakdowns
- Strict duration tracking of negative audio (no extrapolated false alerts per hour)
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEVICE,
    MAX_COMPETING_CONFIDENCE,
    MAX_ENV_CONFIDENCE,
    MIN_CONFIDENCE_MARGIN,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    PROCESSED_DIR,
    UNCERTAIN_ENTROPY_THRESHOLD,
    WINDOW_DURATION,
)
from stage2_classifier import SprayCanShakingCNN, evaluate_candidate_prediction


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_partition_across_thresholds(
    model: SprayCanShakingCNN,
    data_path: Path,
    split_name: str,
    thresholds: List[float] = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
) -> Dict:
    data = torch.load(data_path, weights_only=True)
    x = data["spectrograms"].to(DEVICE)
    y = data["labels"].numpy()

    with torch.no_grad():
        probs = model.predict_proba(x).cpu().numpy()

    shaking_idx = CLASS_TO_IDX["spray_can_shaking"]
    total_samples = len(y)
    total_targets = int(np.sum(y == shaking_idx))
    total_negatives = int(np.sum(y != shaking_idx))

    # Class breakdown of ground truth
    class_counts = {c: int(np.sum(y == CLASS_TO_IDX[c])) for c in CLASS_NAMES}

    threshold_results = []

    for tau in thresholds:
        tp, fp, fn, tn = 0, 0, 0, 0
        fp_by_class = {c: 0 for c in CLASS_NAMES if c != "spray_can_shaking"}
        uncertain_count = 0

        # Detailed 6-class confusion matrix
        # rows = true class, cols = predicted class or rejected
        cm_6x6 = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

        for i in range(total_samples):
            p_vec = probs[i]
            true_idx = int(y[i])
            true_name = CLASS_NAMES[true_idx]
            is_spray_true = (true_idx == shaking_idx)

            eval_res = evaluate_candidate_prediction(
                p_vec,
                conservative_threshold=tau,
                max_competing_confidence=MAX_COMPETING_CONFIDENCE,
                max_env_confidence=MAX_ENV_CONFIDENCE,
                min_confidence_margin=MIN_CONFIDENCE_MARGIN,
                entropy_threshold=UNCERTAIN_ENTROPY_THRESHOLD,
            )

            is_spray_pred = eval_res["is_confirmed_spray"]
            if eval_res["is_uncertain"]:
                uncertain_count += 1

            raw_pred_idx = int(np.argmax(p_vec))
            cm_6x6[true_idx, raw_pred_idx] += 1

            if is_spray_true:
                if is_spray_pred:
                    tp += 1
                else:
                    fn += 1
            else:
                if is_spray_pred:
                    fp += 1
                    fp_by_class[true_name] += 1
                else:
                    tn += 1

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        rec = float(tp / total_targets) if total_targets > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        spec = float(tn / total_negatives) if total_negatives > 0 else 1.0

        threshold_results.append({
            "threshold": round(tau, 2),
            "confusion_counts": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
            "metrics": {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "missed_detection_rate": round(1.0 - rec, 4),
                "specificity": round(spec, 4),
                "f1_score": round(f1, 4),
            },
            "false_positives_by_negative_class": fp_by_class,
            "uncertain_candidate_count": uncertain_count,
            "raw_argmax_confusion_matrix": cm_6x6.tolist(),
        })

    return {
        "split_name": split_name,
        "sample_counts": {
            "total_samples": total_samples,
            "target_samples": total_targets,
            "negative_samples": total_negatives,
            "class_distribution": class_counts,
            "total_audio_duration_seconds": round(total_samples * WINDOW_DURATION, 2),
            "negative_audio_duration_seconds": round(total_negatives * WINDOW_DURATION, 2),
            "negative_audio_duration_minutes": round(total_negatives * WINDOW_DURATION / 60.0, 2),
        },
        "threshold_sweep": threshold_results,
    }


def evaluate_test_conditions(
    model: SprayCanShakingCNN,
    test_path: Path,
    conditions_path: Path,
    tau: float = 0.80,
) -> Dict:
    data = torch.load(test_path, weights_only=True)
    x = data["spectrograms"].to(DEVICE)
    y = data["labels"].numpy()

    with torch.no_grad():
        probs = model.predict_proba(x).cpu().numpy()

    cond_data = []
    if conditions_path.exists():
        with open(conditions_path, "r", encoding="utf-8") as f:
            cond_data = json.load(f)

    shaking_idx = CLASS_TO_IDX["spray_can_shaking"]
    conditions = [
        "quiet_clean",
        "busy_station",
        "overlapping_train_announcement",
        "spray_hiss",
        "similar_rattles",
        "masked_extreme",
    ]

    results = {}
    for c in conditions:
        results[c] = {
            "condition_description": "",
            "dataset_nature": "synthetic_mixture" if c != "quiet_clean" else "laboratory_recording",
            "total_samples": 0,
            "target_samples": 0,
            "negative_samples": 0,
            "duration_seconds": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": None,
            "recall": None,
            "rejection_rate": None,
        }

    results["quiet_clean"]["condition_description"] = "Solo spray can shaking in quiet room (no train noise)"
    results["busy_station"]["condition_description"] = "Continuous station background noise ONLY (NO positive shake audio present)"
    results["overlapping_train_announcement"]["condition_description"] = "Spray can shaking mixed with train rumble & announcements (+12 to 0 dB SNR)"
    results["spray_hiss"]["condition_description"] = "Aerosol spray discharge hiss, sputter, and cap handling"
    results["similar_rattles"]["condition_description"] = "Hard negative rattles (keys, coins, pill bottles, glassware)"
    results["masked_extreme"]["condition_description"] = "Heavy acoustic masking where train pass-by exceeds can SPL by 4 dB (-4 dB SNR)"

    for i in range(len(cond_data)):
        c_label = cond_data[i]["condition"]
        if c_label not in results:
            continue
        d = results[c_label]
        d["total_samples"] += 1
        d["duration_seconds"] = round(d["total_samples"] * WINDOW_DURATION, 2)

        true_idx = int(y[i])
        is_spray_true = (true_idx == shaking_idx)

        eval_res = evaluate_candidate_prediction(
            probs[i],
            conservative_threshold=tau,
            max_competing_confidence=MAX_COMPETING_CONFIDENCE,
            max_env_confidence=MAX_ENV_CONFIDENCE,
            min_confidence_margin=MIN_CONFIDENCE_MARGIN,
            entropy_threshold=UNCERTAIN_ENTROPY_THRESHOLD,
        )
        is_spray_pred = eval_res["is_confirmed_spray"]

        if is_spray_true:
            d["target_samples"] += 1
            if is_spray_pred:
                d["tp"] += 1
            else:
                d["fn"] += 1
        else:
            d["negative_samples"] += 1
            if is_spray_pred:
                d["fp"] += 1
            else:
                d["tn"] += 1

    for c, d in results.items():
        tp = d["tp"]
        fp = d["fp"]
        fn = d["fn"]
        tn = d["tn"]
        tgt = d["target_samples"]
        neg = d["negative_samples"]

        d["precision"] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        # CRITICAL AUDIT FIX: If target_samples == 0, recall is mathematically undefined!
        d["recall"] = round(tp / tgt, 4) if tgt > 0 else "N/A (No target samples in test set)"
        d["rejection_rate"] = round(tn / neg, 4) if neg > 0 else "N/A"

    return results


def run_canonical_evaluation() -> Dict:
    run_id = "CANONICAL_RUN_20261013_RECONCILED"

    model_path = MODEL_SAVE_PATH
    val_path = PROCESSED_DIR / "val.pt"
    test_path = PROCESSED_DIR / "test.pt"
    manifest_path = PROCESSED_DIR / "split_manifest.json"
    cond_path = PROCESSED_DIR / "test_conditions.json"

    # Cryptographic hashes
    artifacts_provenance = {
        "run_id": run_id,
        "model_checkpoint_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "val_partition_path": str(val_path),
        "val_sha256": file_sha256(val_path),
        "test_partition_path": str(test_path),
        "test_sha256": file_sha256(test_path),
        "split_manifest_path": str(manifest_path),
        "split_manifest_sha256": file_sha256(manifest_path),
        "test_conditions_path": str(cond_path),
        "test_conditions_sha256": file_sha256(cond_path),
    }

    # Load Model
    model = SprayCanShakingCNN(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_eval = evaluate_partition_across_thresholds(model, val_path, split_name="validation")
    test_eval = evaluate_partition_across_thresholds(model, test_path, split_name="held_out_test")
    test_conditions = evaluate_test_conditions(model, test_path, cond_path, tau=0.80)

    report = {
        "provenance": artifacts_provenance,
        "operating_configuration": {
            "production_conservative_threshold": 0.80,
            "max_competing_confidence": MAX_COMPETING_CONFIDENCE,
            "max_env_confidence": MAX_ENV_CONFIDENCE,
            "min_confidence_margin": MIN_CONFIDENCE_MARGIN,
            "uncertain_entropy_threshold": UNCERTAIN_ENTROPY_THRESHOLD,
            "required_consecutive_windows": 2,
            "cooldown_seconds": 3.0,
            "detection_delay_seconds": 1.25,
        },
        "validation_evaluation": val_eval,
        "held_out_test_evaluation": test_eval,
        "test_conditions_breakdown_at_tau_0_80": test_conditions,
        "duration_and_reliability_statement": (
            "The held-out test partition contains 271 one-second negative windows, totaling 271 seconds (4.52 minutes) "
            "of concatenated synthetic and laboratory negative audio. This negative duration is insufficient to establish "
            "a reliable false alert rate per hour for an operational railway environment. True continuous false alarm rates "
            "can only be measured by deploying continuous multi-hour acoustic recordings through the installed platform microphone."
        ),
    }

    output_path = Path("models/canonical_evaluation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Canonical report generated at {output_path}")
    return report


if __name__ == "__main__":
    report = run_canonical_evaluation()
    print("Run completed successfully.")
