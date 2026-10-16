"""Full Streaming Field Detector Evaluation Framework.

Evaluates continuous railway platform audio through the complete streaming engine:
1. Simulates real-time sliding-window streaming (hop=0.25s, window=1.0s).
2. Exercises Stage 1 burst detector, adaptive spectral noise cancellation,
   Stage 2 multi-gated CNN classifier at production threshold tau=0.80,
   multi-window temporal confirmation (>= 2 windows), and alert cooldown (3.0s).
3. Evaluates EVENT-LEVEL metrics (not just isolated one-second windows):
   - Event detection rate (Recall) and missed event count
   - Detection delay (latency from event onset to alert)
   - False alert episodes per hour of negative audio
   - Condition-level breakdown (quiet, train passby, announcement, rain, passenger crowd)
4. Computes Wilson score and Poisson exact 95% confidence intervals.
5. Preserves all canonical artifacts and logs exact SHA-256 hashes.
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from audio_utils import load_audio
from config import (
    ALERT_COOLDOWN_SECONDS,
    AUDIO_RETENTION_HOURS,
    AUDIO_SAMPLES,
    BASE_DIR,
    CALIBRATION_PATH,
    CLASS_NAMES,
    CONSERVATIVE_THRESHOLD,
    DEVICE,
    FIELD_SESSIONS_DIR,
    MAX_COMPETING_CONFIDENCE,
    MAX_ENV_CONFIDENCE,
    MIN_CONFIDENCE_MARGIN,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    REQUIRED_CONSECUTIVE_WINDOWS,
    SAMPLE_RATE,
    SHADOW_MODE_ENABLED,
    STREAM_HOP_DURATION,
    UNCERTAIN_ENTROPY_THRESHOLD,
    WINDOW_DURATION,
)
from detector_engine import SprayCanDetectorEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    # z = 1.95996 for 95% confidence
    z = 1.95996 if confidence == 0.95 else 1.64485
    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denom
    spread = (z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)) / denom
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (round(lower, 4), round(upper, 4))


def compute_interval_union_duration(intervals: List[Tuple[float, float]], max_duration: float) -> float:
    """Compute the total non-overlapping duration of interval union clamped to [0, max_duration]."""
    if not intervals or max_duration <= 0.0:
        return 0.0
    clamped = []
    for s, e in intervals:
        s_c = max(0.0, min(max_duration, float(s)))
        e_c = max(0.0, min(max_duration, float(e)))
        if e_c > s_c:
            clamped.append((s_c, e_c))
    if not clamped:
        return 0.0
    clamped.sort(key=lambda x: x[0])
    merged = [clamped[0]]
    for cur_s, cur_e in clamped[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return sum(e - s for s, e in merged)


def poisson_rate_confidence_interval(k: int, hours: float, confidence: float = 0.95) -> Optional[Dict]:
    """Calculate exact Poisson 95% confidence interval for rate per hour.

    - If hours <= 0: Exposure is zero, rate and CI are undefined (returns None).
    - If k == 0 and hours > 0: Lower = 0.0, Upper = -ln(alpha) / T ≈ 2.9957 / T (Rule of Three).
    - If k >= 1 and hours > 0: Exact Chi-square interval:
        Lower = 0.5 * chi2(2k, alpha/2) / T
        Upper = 0.5 * chi2(2k+2, 1 - alpha/2) / T
    """
    if hours <= 0.0:
        return None

    alpha = 1.0 - confidence
    rule_of_three_upper = round(-math.log(alpha) / hours, 4)

    try:
        from scipy.stats import chi2
        if k == 0:
            lower = 0.0
            upper = rule_of_three_upper
            interpretation = (
                f"Zero false alerts observed across {hours:.3f} negative hours. "
                f"Poisson exact 95% upper confidence bound is {upper:.4f} FA/hr (≈ 3.0 / T). "
                "This does NOT guarantee zero false alerts in continuous deployment."
            )
        else:
            lower = round(float(0.5 * chi2.ppf(alpha / 2.0, 2 * k) / hours), 4)
            upper = round(float(0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (k + 1)) / hours), 4)
            interpretation = (
                f"Poisson exact 95% confidence interval for {k} false alert(s) across {hours:.3f} hours: "
                f"[{lower:.4f}, {upper:.4f}] FA/hr."
            )
        return {
            "lower": lower,
            "upper": upper,
            "rule_of_three_upper": rule_of_three_upper if k == 0 else None,
            "interpretation": interpretation,
        }
    except Exception:
        if k == 0:
            return {
                "lower": 0.0,
                "upper": rule_of_three_upper,
                "rule_of_three_upper": rule_of_three_upper,
                "interpretation": f"Approximate Rule of Three 95% upper bound: {rule_of_three_upper:.4f} FA/hr.",
            }
        rate = k / hours
        margin = 1.96 * math.sqrt(k) / hours
        return {
            "lower": round(max(0.0, rate - margin), 4),
            "upper": round(rate + margin, 4),
            "rule_of_three_upper": None,
            "interpretation": f"Normal approximation 95% confidence interval: [{max(0.0, rate - margin):.4f}, {rate + margin:.4f}] FA/hr.",
        }


class StreamingFieldEvaluator:
    """Evaluates continuous audio sessions against ground-truth interval annotations."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conservative_threshold: float = 0.80,
        hop_seconds: float = STREAM_HOP_DURATION,
    ):
        self.conservative_threshold = conservative_threshold
        self.hop_seconds = hop_seconds
        self.hop_samples = int(SAMPLE_RATE * hop_seconds)
        self.engine = SprayCanDetectorEngine(
            model_path=model_path,
            conservative_threshold=self.conservative_threshold,
            required_consecutive_windows=REQUIRED_CONSECUTIVE_WINDOWS,
            cooldown_seconds=ALERT_COOLDOWN_SECONDS,
            enable_noise_cancellation=True,
        )

    def evaluate_continuous_session(
        self,
        audio_path: Path,
        annotations: List[Dict],
        tolerance_seconds: float = 1.5,
        episode_merge_window: float = 3.0,
    ) -> Dict:
        """Stream an audio file chunk by chunk and match alert episodes to ground truth events.

        Episode Grouping & Event Matching Rules:
        1. Ground-Truth Event Matching:
           Any raw alert triggers firing within an event's window [start_sec - 0.25, end_sec + tolerance]
           are merged into exactly ONE True Positive Alert Episode for that event.
           A sustained shake lasting longer than the detector cooldown fires multiple raw triggers,
           but yields exactly 1 detected event and 1 true alert episode.
        2. Missed Target Events:
           Target events with zero matching alert triggers are marked detected=False (Missed Event).
        3. False Alert Episode Grouping:
           Raw alert triggers that fall outside all labeled target events are grouped into False Alert
           Episodes using an inter-trigger merge window (episode_merge_window=3.0s). Consecutive false
           triggers within 3.0s of each other merge into a single False Alert Episode.
        4. Cooldown Interaction:
           The detector engine's internal cooldown (3.0s) suppresses raw alert firings within 3.0s of
           a previous alert. If two distinct events occur with an inter-event gap < cooldown, the second
           is suppressed by cooldown. If gap >= cooldown, both events can be detected as distinct episodes.
        """
        audio = load_audio(audio_path, sr=SAMPLE_RATE)
        total_duration_sec = len(audio) / SAMPLE_RATE
        self.engine.reset_state()

        # 1. Parse Ground Truth Target Events
        target_events = []
        for ann in annotations:
            cls_name = ann.get("class_label", "")
            if cls_name == "spray_can_shaking":
                target_events.append({
                    "start_sec": float(ann["start_sec"]),
                    "end_sec": float(ann["end_sec"]),
                    "condition": ann.get("condition", "general"),
                    "detected": False,
                    "first_alert_sec": None,
                    "last_alert_sec": None,
                    "delay_sec": None,
                    "matched_trigger_count": 0,
                })

        # 2. Simulate streaming chunk ingestion
        raw_alert_triggers = []
        win_samples = AUDIO_SAMPLES
        hop_len = self.hop_samples

        for start_idx in range(0, max(1, len(audio) - win_samples + 1), hop_len):
            chunk = audio[start_idx : start_idx + win_samples]
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))
            t_current = start_idx / SAMPLE_RATE

            eval_res = self.engine.evaluate_window(chunk, current_time=t_current)
            if eval_res.get("alert_fired", False):
                raw_alert_triggers.append({
                    "alert_time_sec": round(t_current, 3),
                    "confidence": eval_res.get("confidence", 0.0),
                    "entropy": eval_res.get("entropy", 0.0),
                    "margin": eval_res.get("confidence_margin", 0.0),
                })

        # 3. Match Raw Alert Triggers to Target Events (1-to-1 Episode per Target Event)
        matched_trigger_indices = set()
        tp_episodes = []

        for idx, event in enumerate(target_events):
            t_start_tol = event["start_sec"] - 0.25
            t_end_tol = event["end_sec"] + tolerance_seconds
            ev_triggers = []

            for trig_idx, trig in enumerate(raw_alert_triggers):
                t_alert = trig["alert_time_sec"]
                if t_start_tol <= t_alert <= t_end_tol:
                    ev_triggers.append(trig)
                    matched_trigger_indices.add(trig_idx)

            if ev_triggers:
                event["detected"] = True
                event["first_alert_sec"] = min(tr["alert_time_sec"] for tr in ev_triggers)
                event["last_alert_sec"] = max(tr["alert_time_sec"] for tr in ev_triggers)
                event["delay_sec"] = round(event["first_alert_sec"] - event["start_sec"], 3)
                event["matched_trigger_count"] = len(ev_triggers)
                tp_episodes.append({
                    "episode_type": "true_positive",
                    "matched_event_idx": idx,
                    "event_start_sec": event["start_sec"],
                    "event_end_sec": event["end_sec"],
                    "first_alert_sec": event["first_alert_sec"],
                    "last_alert_sec": event["last_alert_sec"],
                    "delay_sec": event["delay_sec"],
                    "trigger_count": len(ev_triggers),
                    "max_confidence": max(tr["confidence"] for tr in ev_triggers),
                })

        # 4. Group Unmatched Triggers into False Alert Episodes
        unmatched_triggers = [
            trig for trig_idx, trig in enumerate(raw_alert_triggers)
            if trig_idx not in matched_trigger_indices
        ]
        unmatched_triggers.sort(key=lambda x: x["alert_time_sec"])

        fp_episodes = []
        if unmatched_triggers:
            cur_cluster = [unmatched_triggers[0]]
            for trig in unmatched_triggers[1:]:
                if trig["alert_time_sec"] - cur_cluster[-1]["alert_time_sec"] <= episode_merge_window:
                    cur_cluster.append(trig)
                else:
                    fp_episodes.append({
                        "episode_type": "false_positive",
                        "matched_event_idx": None,
                        "first_alert_sec": cur_cluster[0]["alert_time_sec"],
                        "last_alert_sec": cur_cluster[-1]["alert_time_sec"],
                        "trigger_count": len(cur_cluster),
                        "max_confidence": max(c["confidence"] for c in cur_cluster),
                    })
                    cur_cluster = [trig]
            if cur_cluster:
                fp_episodes.append({
                    "episode_type": "false_positive",
                    "matched_event_idx": None,
                    "first_alert_sec": cur_cluster[0]["alert_time_sec"],
                    "last_alert_sec": cur_cluster[-1]["alert_time_sec"],
                    "trigger_count": len(cur_cluster),
                    "max_confidence": max(c["confidence"] for c in cur_cluster),
                })

        # 5. Event-Level Counts & Exposure Calculations
        total_targets = len(target_events)
        detected_targets = sum(1 for e in target_events if e["detected"])
        missed_targets = total_targets - detected_targets
        delays = [e["delay_sec"] for e in target_events if e["delay_sec"] is not None]

        num_false_alert_episodes = len(fp_episodes)
        true_alert_episodes_count = len(tp_episodes)
        total_alert_episodes_count = true_alert_episodes_count + num_false_alert_episodes

        # Negative audio duration: total session duration minus duration of positive events (union of intervals)
        positive_intervals = [(e["start_sec"], e["end_sec"]) for e in target_events]
        positive_duration_sec = compute_interval_union_duration(positive_intervals, total_duration_sec)
        negative_duration_sec = max(0.0, total_duration_sec - positive_duration_sec)
        negative_duration_hours = negative_duration_sec / 3600.0

        if negative_duration_hours > 0.0:
            fa_per_hour = round(num_false_alert_episodes / negative_duration_hours, 4)
            fa_ci = poisson_rate_confidence_interval(num_false_alert_episodes, negative_duration_hours)
        else:
            fa_per_hour = None
            fa_ci = None

        rec = (detected_targets / total_targets) if total_targets > 0 else 0.0
        rec_ci = wilson_score_interval(detected_targets, total_targets) if total_targets > 0 else None

        # Condition breakdown
        cond_stats = {}
        for event in target_events:
            c = event["condition"]
            if c not in cond_stats:
                cond_stats[c] = {"target_events": 0, "detected_events": 0, "missed_events": 0}
            cond_stats[c]["target_events"] += 1
            if event["detected"]:
                cond_stats[c]["detected_events"] += 1
            else:
                cond_stats[c]["missed_events"] += 1

        for c, s in cond_stats.items():
            tgt = s["target_events"]
            det = s["detected_events"]
            s["detection_rate"] = round(det / tgt, 4) if tgt > 0 else "N/A"

        all_episodes = tp_episodes + fp_episodes
        all_episodes.sort(key=lambda ep: ep["first_alert_sec"])

        return {
            "session_audio_path": str(audio_path),
            "session_file_sha256": compute_sha256(audio_path),
            "total_duration_seconds": round(total_duration_sec, 2),
            "negative_duration_seconds": round(negative_duration_sec, 2),
            "negative_duration_hours": round(negative_duration_hours, 4),
            "target_events_count": total_targets,
            "detected_events_count": detected_targets,
            "missed_events_count": missed_targets,
            "event_detection_rate": round(rec, 4) if total_targets > 0 else "N/A (No target events)",
            "event_detection_rate_95ci": rec_ci if total_targets > 0 else None,
            "detection_delays_seconds": {
                "mean": round(float(np.mean(delays)), 3) if delays else None,
                "median": round(float(np.median(delays)), 3) if delays else None,
                "min": round(float(min(delays)), 3) if delays else None,
                "max": round(float(max(delays)), 3) if delays else None,
            },
            "raw_alert_triggers_count": len(raw_alert_triggers),
            "alert_episodes_fired": total_alert_episodes_count,
            "true_alert_episodes": true_alert_episodes_count,
            "false_alert_episodes": num_false_alert_episodes,
            "false_alert_episodes_per_hour": fa_per_hour,
            "false_alert_rate_95ci_per_hour": fa_ci,
            "condition_breakdown": cond_stats,
            "target_events_detail": target_events,
            "alert_episodes_detail": all_episodes,
            "raw_alert_triggers_detail": raw_alert_triggers,
        }


def run_streaming_field_evaluation(
    split_manifest_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict:
    """Run full streaming evaluation across all sessions in field dataset."""
    if run_id is None:
        run_id = f"FIELD_EVAL_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    evaluator = StreamingFieldEvaluator(
        model_path=MODEL_SAVE_PATH,
        conservative_threshold=0.80,
    )

    sessions_to_eval = []
    if FIELD_SESSIONS_DIR.exists():
        for sess_dir in sorted(FIELD_SESSIONS_DIR.iterdir()):
            if not sess_dir.is_dir():
                continue
            audio_path = sess_dir / "audio.wav"
            annot_path = sess_dir / "annotations.json"
            if not audio_path.exists():
                continue
            annots = []
            if annot_path.exists():
                try:
                    with open(annot_path, "r", encoding="utf-8") as f:
                        ann_data = json.load(f)
                        annots = ann_data.get("events", [])
                except Exception:
                    pass
            sessions_to_eval.append((sess_dir.name, audio_path, annots))

    session_results = []
    agg_total_targets = 0
    agg_detected_targets = 0
    agg_missed_targets = 0
    agg_false_alerts = 0
    agg_negative_hours = 0.0
    agg_all_delays = []
    agg_conditions = {}

    for sess_name, a_path, annots in sessions_to_eval:
        res = evaluator.evaluate_continuous_session(a_path, annots)
        res["session_id"] = sess_name
        session_results.append(res)

        agg_total_targets += res["target_events_count"]
        agg_detected_targets += res["detected_events_count"]
        agg_missed_targets += res["missed_events_count"]
        agg_false_alerts += res["false_alert_episodes"]
        agg_negative_hours += res["negative_duration_hours"]
        for ev in res["target_events_detail"]:
            if ev.get("delay_sec") is not None:
                agg_all_delays.append(ev["delay_sec"])
            c = ev.get("condition", "ambient")
            if c not in agg_conditions:
                agg_conditions[c] = {"targets": 0, "detected": 0}
            agg_conditions[c]["targets"] += 1
            if ev.get("detected"):
                agg_conditions[c]["detected"] += 1

    agg_rec = (agg_detected_targets / agg_total_targets) if agg_total_targets > 0 else 0.0
    agg_rec_ci = wilson_score_interval(agg_detected_targets, agg_total_targets) if agg_total_targets > 0 else None
    if agg_negative_hours > 0.0:
        agg_fa_rate = round(agg_false_alerts / agg_negative_hours, 4)
        agg_fa_ci = poisson_rate_confidence_interval(agg_false_alerts, agg_negative_hours)
    else:
        agg_fa_rate = None
        agg_fa_ci = None

    report = {
        "run_id": run_id,
        "evaluation_timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_provenance": {
            "checkpoint_path": str(MODEL_SAVE_PATH),
            "model_sha256": compute_sha256(MODEL_SAVE_PATH),
            "architecture": "SprayCanShakingCNN_5class",
            "operating_conservative_threshold": 0.80,
            "max_competing_confidence": MAX_COMPETING_CONFIDENCE,
            "max_env_confidence": MAX_ENV_CONFIDENCE,
            "min_confidence_margin": MIN_CONFIDENCE_MARGIN,
            "uncertain_entropy_threshold": UNCERTAIN_ENTROPY_THRESHOLD,
            "required_consecutive_windows": REQUIRED_CONSECUTIVE_WINDOWS,
            "cooldown_seconds": ALERT_COOLDOWN_SECONDS,
            "shadow_mode_enabled": True,
            "automatic_enforcement_allowed": False,
        },
        "aggregate_field_metrics": {
            "total_sessions_evaluated": len(session_results),
            "total_negative_audio_hours": round(agg_negative_hours, 4),
            "total_target_events": agg_total_targets,
            "detected_target_events": agg_detected_targets,
            "missed_target_events": agg_missed_targets,
            "event_detection_rate": round(agg_rec, 4) if agg_total_targets > 0 else "N/A (No target events collected yet)",
            "event_detection_rate_95ci": {
                "lower": agg_rec_ci[0],
                "upper": agg_rec_ci[1],
            } if agg_rec_ci else None,
            "detection_delay_seconds": {
                "median": round(float(np.median(agg_all_delays)), 3) if agg_all_delays else None,
                "mean": round(float(np.mean(agg_all_delays)), 3) if agg_all_delays else None,
                "min": round(float(min(agg_all_delays)), 3) if agg_all_delays else None,
                "max": round(float(max(agg_all_delays)), 3) if agg_all_delays else None,
            },
            "total_false_alert_episodes": agg_false_alerts,
            "false_alert_episodes_per_hour": agg_fa_rate,
            "false_alert_episodes_per_hour_95ci": agg_fa_ci,
        },
        "condition_breakdown": {
            c: {
                "target_events": st["targets"],
                "detected_events": st["detected"],
                "detection_rate": round(st["detected"] / st["targets"], 4) if st["targets"] > 0 else "N/A",
            }
            for c, st in agg_conditions.items()
        },
        "statistical_uncertainty_and_guarantee_statement": (
            "CRITICAL RELIABILITY NOTICE: Zero false alerts observed over limited test duration "
            "DOES NOT constitute a zero-risk guarantee. True platform false-alert rates require "
            "continuous multi-day and multi-week recordings through the installed microphone under "
            "all weather, passenger density, and railway operational schedules. "
            "Acoustic alerts are advisory shadow-mode signals only; human visual review is mandatory."
        ),
        "session_evaluations": session_results,
    }

    out_file = Path("models") / f"field_evaluation_report_{run_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Field evaluation report generated at %s", out_file)
    return report


if __name__ == "__main__":
    report = run_streaming_field_evaluation()
    print("Streaming Field Evaluation completed.")
    print("Aggregate Summary:")
    print(json.dumps(report.get("aggregate_field_metrics", {}), indent=2))
