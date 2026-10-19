"""Comprehensive Comparative Evaluation Benchmark: PyTorch vs TensorFlow.

Evaluates the original PyTorch model (`models/best_detector.pth`) and the converted
TensorFlow model (`models/best_detector.keras`) on the exact same representative audio samples:
1. Class probabilities per audio clip across all 6 classes
2. Prediction class agreements and verdict alignment
3. Discrepancies in false positives, false negatives, or margin
4. End-to-end inference latency benchmark (CPU ms per window)
5. Generates a structured JSON comparison report at `backend/models/tf_pytorch_comparison_report.json`
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from audio_utils import compute_log_mel_spectrogram, load_audio, pad_or_trim
from config import (
    AUDIO_SAMPLES,
    CLASS_NAMES,
    CLASS_TO_IDX,
    CONSERVATIVE_THRESHOLD,
    MODELS_DIR,
    NUM_CLASSES,
    SAMPLE_RATE,
)
from stage2_classifier import SprayCanShakingCNN, evaluate_candidate_prediction
from stage2_classifier_tf import SprayCanShakingCNNTF


def run_comparative_benchmark(
    pth_path: Path = MODELS_DIR / "best_detector.pth",
    tf_keras_path: Path = MODELS_DIR / "best_detector.keras",
    report_out_path: Path = MODELS_DIR / "tf_pytorch_comparison_report.json",
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
) -> Dict[str, Any]:
    """Execute side-by-side comparative validation across representative samples."""
    pth_path = Path(pth_path)
    tf_keras_path = Path(tf_keras_path)
    report_out_path = Path(report_out_path)

    if not pth_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pth_path}")
    if not tf_keras_path.exists():
        raise FileNotFoundError(f"TensorFlow model not found: {tf_keras_path}. Run convert_pytorch_to_tf.py first.")

    print("=" * 80)
    print("  [+] PYTORCH vs TENSORFLOW STAGE 2 DETECTOR BENCHMARK")
    print("=" * 80)
    print(f"  * PyTorch Checkpoint:    {pth_path}")
    print(f"  * TensorFlow Model:      {tf_keras_path}")

    # 1. Load PyTorch model
    print("[*] Loading PyTorch model...")
    pt_ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    pt_state = pt_ckpt["model_state_dict"] if "model_state_dict" in pt_ckpt else pt_ckpt
    pt_model = SprayCanShakingCNN(num_classes=NUM_CLASSES)
    pt_model.load_state_dict(pt_state)
    pt_model.eval()

    # 2. Load TensorFlow model
    print("[*] Loading TensorFlow model...")
    tf_wrapper = SprayCanShakingCNNTF(num_classes=NUM_CLASSES, weights_path=tf_keras_path)

    # 3. Gather representative audio files
    sample_dirs = [
        BACKEND_DIR / "data" / "demo_samples",
        BACKEND_DIR / "data" / "raw" / "spray_can_shaking",
        BACKEND_DIR / "data" / "raw" / "aerosol_spray_hiss",
        BACKEND_DIR / "data" / "raw" / "hard_negatives",
        BACKEND_DIR / "data" / "raw" / "ambient_negatives",
        BACKEND_DIR / "data" / "raw" / "loud_impacts",
        BACKEND_DIR / "data" / "raw" / "continuous_environmental",
    ]

    audio_files: List[Path] = []
    seen_names = set()
    for s_dir in sample_dirs:
        if s_dir.exists():
            for f in sorted(s_dir.glob("*.wav")):
                if f.name not in seen_names:
                    audio_files.append(f)
                    seen_names.add(f.name)
                if len(audio_files) >= 25:
                    break
        if len(audio_files) >= 25:
            break

    print(f"[*] Collected {len(audio_files)} representative audio files for evaluation.")

    # 4. Latency Warmup
    dummy_spec = np.random.randn(64, 44).astype(np.float32)
    dummy_pt = torch.from_numpy(dummy_spec).unsqueeze(0).unsqueeze(0)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = pt_model.predict_proba(dummy_pt)
        _ = tf_wrapper.predict_proba(dummy_spec)

    sample_results = []
    all_prob_diffs = []
    pt_latencies = []
    tf_latencies = []

    print("\n" + "-" * 105)
    print(f"{'Sample File':<32} | {'True / Cat':<16} | {'PT Top (P)':<18} | {'TF Top (P)':<18} | {'Max Delta':<10} | {'Status'}")
    print("-" * 105)

    for a_path in audio_files:
        try:
            audio = load_audio(a_path, sr=SAMPLE_RATE)
            audio = pad_or_trim(audio, AUDIO_SAMPLES)
            log_mel = compute_log_mel_spectrogram(audio, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"[!] Skipping {a_path.name}: {e}")
            continue

        # Benchmark PyTorch Latency & Predictions
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_in = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)
            pt_probs = pt_model.predict_proba(pt_in).numpy()[0]
        t1 = time.perf_counter()
        pt_lat = (t1 - t0) * 1000.0  # ms
        pt_latencies.append(pt_lat)

        # Benchmark TensorFlow Latency & Predictions
        t0 = time.perf_counter()
        tf_probs = tf_wrapper.predict_proba(log_mel)[0]
        t1 = time.perf_counter()
        tf_lat = (t1 - t0) * 1000.0  # ms
        tf_latencies.append(tf_lat)

        # Evaluate candidate prediction logic for both
        pt_eval = evaluate_candidate_prediction(pt_probs, conservative_threshold=CONSERVATIVE_THRESHOLD)
        tf_eval = evaluate_candidate_prediction(tf_probs, conservative_threshold=CONSERVATIVE_THRESHOLD)

        prob_diff = np.abs(pt_probs - tf_probs)
        max_diff = float(np.max(prob_diff))
        all_prob_diffs.append(max_diff)

        pred_pt_idx = int(np.argmax(pt_probs))
        pred_tf_idx = int(np.argmax(tf_probs))
        pred_pt_name = CLASS_NAMES[pred_pt_idx]
        pred_tf_name = CLASS_NAMES[pred_tf_idx]

        category = a_path.parent.name
        pt_summary = f"{pred_pt_name[:10]} ({pt_probs[pred_pt_idx]:.2f})"
        tf_summary = f"{pred_tf_name[:10]} ({tf_probs[pred_tf_idx]:.2f})"
        status = "MATCH" if pred_pt_idx == pred_tf_idx and pt_eval["verdict"] == tf_eval["verdict"] else "DIVERGE"

        print(f"{a_path.name[:32]:<32} | {category[:16]:<16} | {pt_summary:<18} | {tf_summary:<18} | {max_diff:<10.2e} | {status}")

        sample_results.append({
            "filename": a_path.name,
            "category": category,
            "max_prob_diff": max_diff,
            "pt_prediction": pred_pt_name,
            "tf_prediction": pred_tf_name,
            "pt_p_spray": pt_eval["p_spray_can"],
            "tf_p_spray": tf_eval["p_spray_can"],
            "pt_verdict": pt_eval["verdict"],
            "tf_verdict": tf_eval["verdict"],
            "pt_latency_ms": round(pt_lat, 2),
            "tf_latency_ms": round(tf_lat, 2),
            "class_probabilities_pt": {name: round(float(pt_probs[i]), 5) for i, name in enumerate(CLASS_NAMES)},
            "class_probabilities_tf": {name: round(float(tf_probs[i]), 5) for i, name in enumerate(CLASS_NAMES)},
        })

    print("-" * 105)

    # 5. Compute Aggregate Summary Statistics
    max_divergence = float(np.max(all_prob_diffs)) if all_prob_diffs else 0.0
    mean_divergence = float(np.mean(all_prob_diffs)) if all_prob_diffs else 0.0
    agreed_predictions = sum(1 for s in sample_results if s["pt_prediction"] == s["tf_prediction"])
    agreed_verdicts = sum(1 for s in sample_results if s["pt_verdict"] == s["tf_verdict"])
    total_evaluated = len(sample_results)
    pred_agreement_rate = (agreed_predictions / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    verdict_agreement_rate = (agreed_verdicts / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    mean_pt_latency = float(np.mean(pt_latencies)) if pt_latencies else 0.0
    mean_tf_latency = float(np.mean(tf_latencies)) if tf_latencies else 0.0

    summary = {
        "total_samples_evaluated": total_evaluated,
        "max_probability_difference": max_divergence,
        "mean_probability_difference": mean_divergence,
        "prediction_agreement_count": agreed_predictions,
        "prediction_agreement_rate_pct": round(pred_agreement_rate, 2),
        "verdict_agreement_count": agreed_verdicts,
        "verdict_agreement_rate_pct": round(verdict_agreement_rate, 2),
        "mean_latency_pytorch_ms": round(mean_pt_latency, 2),
        "mean_latency_tensorflow_ms": round(mean_tf_latency, 2),
        "latency_speedup_factor": round(mean_pt_latency / max(1e-3, mean_tf_latency), 2),
        "status": "PASSED" if max_divergence < 1e-4 and verdict_agreement_rate >= 99.0 else "INVESTIGATE",
    }

    report = {
        "benchmark_summary": summary,
        "sample_evaluations": sample_results,
    }

    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  [+] BENCHMARK SUMMARY: {summary['status']}")
    print("=" * 80)
    print(f"  * Total Samples:               {total_evaluated}")
    print(f"  * Class Prediction Agreement:  {agreed_predictions}/{total_evaluated} ({summary['prediction_agreement_rate_pct']}%)")
    print(f"  * Verdict Gating Agreement:    {agreed_verdicts}/{total_evaluated} ({summary['verdict_agreement_rate_pct']}%)")
    print(f"  * Max Probability Delta:       {max_divergence:.6e}")
    print(f"  * Mean Probability Delta:      {mean_divergence:.6e}")
    print(f"  * PyTorch Mean Latency:        {summary['mean_latency_pytorch_ms']} ms / window")
    print(f"  * TensorFlow Mean Latency:     {summary['mean_latency_tensorflow_ms']} ms / window")
    print(f"  * Saved JSON Report:           {report_out_path}")
    print("=" * 80 + "\n")

    return report


if __name__ == "__main__":
    run_comparative_benchmark()
