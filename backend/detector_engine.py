"""Unified Two-Stage Precision Aerosol Spray Can Detector Engine.

Implements the conservative multi-evidence decision pipeline:
1. Stage 1: Kinematic shaking burst candidate detection & digital clipping rejection.
   - Rule 1: Volume/loudness used ONLY to identify digital clipping / unusable audio.
   - Treats any positive finding strictly as an unconfirmed Candidate.
   - Requires repeated back-and-forth physical bursts (cadence 2-13 Hz, span >= 0.40s, valley contrast >= 4.5 dB).
2. Stage 2: Deep spectro-temporal CNN classifier (SprayCanShakingCNN, 5 classes).
   - Full 2D Log-Mel spectrogram (64 mels x 44 frames).
   - Rule 3: Multiple independent lines of evidence:
     * P(spray_can_shaking) >= conservative_threshold (default 0.85).
     * Competing classes (loud impact, environmental, hard negative rattle, speech) <= MAX_COMPETING_CONFIDENCE (0.20).
     * Confidence margin P(spray) - max(P(competing)) >= MIN_CONFIDENCE_MARGIN (0.45).
     * Shannon entropy < UNCERTAIN_ENTROPY_THRESHOLD (0.60).
3. Multi-Window Temporal Persistence Gating:
   - Must match across REQUIRED_CONSECUTIVE_WINDOWS (>= 3 consecutive windows, >= 1.0s).
4. Cooldown Enforcement:
   - 3.0s cooldown after confirmed alert.
5. Rule 8: Structured debug logging without storing microphone audio unless explicitly configured.
"""

import datetime
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from audio_utils import (
    compute_log_mel_spectrogram,
    load_audio,
    pad_or_trim,
)
from config import (
    ALERT_COOLDOWN_SECONDS,
    ALERT_TITLE,
    AUDIO_RETENTION_HOURS,
    AUDIO_SAMPLES,
    CALIBRATION_PATH,
    CLASS_NAMES,
    CLIPPING_PEAK_THRESHOLD,
    CLIPPING_RATIO_THRESHOLD,
    CONSERVATIVE_THRESHOLD,
    DEVICE,
    ENABLE_EPHEMERAL_REVIEW,
    ENABLE_NOISE_CANCELLATION,
    EVENT_END_QUIET_WINDOWS,
    HOP_SAMPLES,
    MAX_COMPETING_CONFIDENCE,
    MAX_ENV_CONFIDENCE,
    MAX_REVIEW_QUEUE_ITEMS,
    MIN_CONFIDENCE_MARGIN,
    MODEL_BACKEND,
    MODEL_SAVE_PATH,
    TF_MODEL_SAVE_PATH,
    NOISE_FLOOR_FACTOR,
    NOISE_SUBTRACTION_FACTOR,
    NUM_CLASSES,
    OPERATOR_REVIEW_NOTICE,
    REQUIRED_CONSECUTIVE_WINDOWS,
    REVIEW_QUEUE_DIR,
    SAMPLE_RATE,
    SAVE_MIC_AUDIO,
    SHADOW_ALERTS_LOG,
    SHADOW_MODE_ENABLED,
    FIELD_SESSIONS_DIR,
    STAGE1_MIN_RMS,
    UNCERTAIN_ENTROPY_THRESHOLD,
    WINDOW_DURATION,
)
from noise_canceller import AdaptiveSpectralNoiseCanceller
from stage1_shaking_detector import Stage1ShakingDetector
from stage2_classifier import SprayCanShakingCNN, evaluate_candidate_prediction

logger = logging.getLogger(__name__)


class SprayCanDetectorEngine:
    """Production-grade two-stage aerosol spray can shaking sound detector."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        calibration_path: Optional[Path] = None,
        conservative_threshold: Optional[float] = None,
        required_consecutive_windows: int = REQUIRED_CONSECUTIVE_WINDOWS,
        cooldown_seconds: float = ALERT_COOLDOWN_SECONDS,
        enable_noise_cancellation: bool = ENABLE_NOISE_CANCELLATION,
        sr: int = SAMPLE_RATE,
        backend: Optional[str] = None,
    ):
        self.sr = sr
        self.required_consecutive_windows = required_consecutive_windows
        self.cooldown_seconds = cooldown_seconds
        self.enable_noise_cancellation = enable_noise_cancellation

        # 1. Initialize Stage 1 Kinematic Shaking Detector
        self.stage1 = Stage1ShakingDetector(sr=self.sr)

        # 2. Initialize Real-Time Adaptive Spectral Noise Canceller
        self.noise_canceller = AdaptiveSpectralNoiseCanceller(
            sr=self.sr,
            alpha=NOISE_SUBTRACTION_FACTOR,
            beta=NOISE_FLOOR_FACTOR,
        )

        # 3. Load Conservative Threshold from Calibration if available
        self.calibration_path = Path(calibration_path or CALIBRATION_PATH)
        if conservative_threshold is not None:
            self.conservative_threshold = conservative_threshold
        elif self.calibration_path.exists():
            try:
                with open(self.calibration_path, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                    self.conservative_threshold = float(
                        calib_data.get("selected_conservative_threshold", CONSERVATIVE_THRESHOLD)
                    )
            except Exception as e:
                logger.warning("Could not read calibration file: %s. Using default: %.2f", e, CONSERVATIVE_THRESHOLD)
                self.conservative_threshold = CONSERVATIVE_THRESHOLD
        else:
            self.conservative_threshold = CONSERVATIVE_THRESHOLD

        # 4. Initialize Stage 2 Neural Network (Dual Backend: PyTorch or TensorFlow)
        self.backend = (backend or MODEL_BACKEND).lower()
        default_model_path = TF_MODEL_SAVE_PATH if self.backend == "tensorflow" else MODEL_SAVE_PATH
        self.model_path = Path(model_path or default_model_path)
        self.model = None
        self.model_loaded = False
        self._load_model_weights()

        # 5. State tracking for real-time streaming
        self.shadow_mode = SHADOW_MODE_ENABLED
        self.consecutive_positive_windows = 0
        self.consecutive_quiet_windows = 0
        self.required_quiet_windows = EVENT_END_QUIET_WINDOWS  # Consecutive quiet windows to confirm event end and rearm
        self.alert_episode_active = False
        self._last_alert_time = -999.0
        self.rolling_audio_buffer = np.zeros(AUDIO_SAMPLES, dtype=np.float32)
        self.history = deque(maxlen=50)
        self._was_recent_candidate = False
        self.review_queue = deque(maxlen=MAX_REVIEW_QUEUE_ITEMS)
        self._purge_expired_review_items()

    @property
    def last_alert_time(self) -> float:
        """Timestamp of the most recent alert onset."""
        return self._last_alert_time

    @last_alert_time.setter
    def last_alert_time(self, val: float) -> None:
        self._last_alert_time = float(val)
        if self._last_alert_time > 0:
            self.alert_episode_active = True
            self.consecutive_quiet_windows = 0

    def _load_model_weights(self) -> None:
        """Load trained neural network checkpoint (PyTorch or TensorFlow)."""
        if not self.model_path.exists():
            logger.warning("Model checkpoint %s not found yet. Stage 2 will remain in standby.", self.model_path)
            self.model_loaded = False
            return

        try:
            if self.backend == "tensorflow" or self.model_path.suffix in (".keras", ".h5"):
                from stage2_classifier_tf import SprayCanShakingCNNTF
                self.model = SprayCanShakingCNNTF(num_classes=NUM_CLASSES, weights_path=self.model_path)
                self.backend = "tensorflow"
                self.model_loaded = True
                logger.info("Loaded Stage 2 TensorFlow neural network from %s", self.model_path)
            else:
                self.model = SprayCanShakingCNN(num_classes=NUM_CLASSES).to(DEVICE)
                checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
                state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.backend = "pytorch"
                self.model_loaded = True
                logger.info("Loaded Stage 2 PyTorch neural network from %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load model from %s: %s", self.model_path, e)
            self.model_loaded = False

    def reload_model_if_available(self) -> bool:
        """Check and reload model if training recently completed."""
        if not self.model_loaded and self.model_path.exists():
            self._load_model_weights()
            if self.calibration_path.exists():
                try:
                    with open(self.calibration_path, "r", encoding="utf-8") as f:
                        calib_data = json.load(f)
                        self.conservative_threshold = float(
                            calib_data.get("selected_conservative_threshold", CONSERVATIVE_THRESHOLD)
                        )
                except Exception:
                    pass
        return self.model_loaded

    def reload_model(
        self,
        model_path: Optional[Union[str, Path]] = None,
        backend: Optional[str] = None,
    ) -> bool:
        """Explicitly hot-reload model weights from an active or specified checkpoint path."""
        if backend is not None:
            self.backend = backend.lower()
        if model_path is not None:
            self.model_path = Path(model_path)
            if self.model_path.suffix in (".keras", ".h5"):
                self.backend = "tensorflow"
            elif self.model_path.suffix == ".pth":
                self.backend = "pytorch"
        self._load_model_weights()
        if self.calibration_path.exists():
            try:
                with open(self.calibration_path, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                    self.conservative_threshold = float(
                        calib_data.get("selected_conservative_threshold", CONSERVATIVE_THRESHOLD)
                    )
            except Exception:
                pass
        return self.model_loaded

    def reset_state(self) -> None:
        """Reset temporal state counters, rolling buffer, and noise filter."""
        self.consecutive_positive_windows = 0
        self.consecutive_quiet_windows = 0
        self.alert_episode_active = False
        self._last_alert_time = -999.0
        self.rolling_audio_buffer.fill(0.0)
        self.history.clear()
        self.noise_canceller.reset()

    def evaluate_window(
        self,
        audio_window: np.ndarray,
        current_time: Optional[float] = None,
    ) -> Dict:
        """Evaluate a single 1.0-second audio window through the conservative two-stage logic."""
        if current_time is None:
            current_time = time.time()

        # Ensure correct window length
        if len(audio_window) != AUDIO_SAMPLES:
            audio_window = pad_or_trim(audio_window, AUDIO_SAMPLES)

        # -------------------------------------------------------------
        # Check Alert State & Rearming Logic (Zero Cooldown, Real-Time Continuous)
        # -------------------------------------------------------------
        time_since_alert = current_time - self.last_alert_time

        # Fallback safety timeout: auto-rearm if no candidate shaking observed for > 15s
        if self.alert_episode_active and time_since_alert > 15.0 and not self._was_recent_candidate:
            self.alert_episode_active = False
            self.consecutive_quiet_windows = 0

        # Cooldown completely removed: continuous real-time evaluation with zero delay
        in_cooldown = False
        cooldown_remaining = 0.0

        # -------------------------------------------------------------
        # Rule 1: Loudness may be used ONLY to identify clipping or unusable audio
        # Inspect raw sensor input before any linear/spectral DSP transforms
        # -------------------------------------------------------------
        raw_peak = float(np.max(np.abs(audio_window))) if len(audio_window) > 0 else 0.0
        raw_clip_ratio = float(np.mean(np.abs(audio_window) >= CLIPPING_PEAK_THRESHOLD)) if len(audio_window) > 0 else 0.0
        if raw_clip_ratio >= CLIPPING_RATIO_THRESHOLD or raw_peak >= 1.05:
            s1 = self.stage1.analyze_window(audio_window)
            self.consecutive_positive_windows = 0
            result = {
                "timestamp": round(current_time, 3),
                "rms": s1["rms"],
                "peak": s1["peak"],
                "is_clipped": True,
                "burst_count": 0,
                "cadence_hz": 0.0,
                "temporal_span_sec": 0.0,
                "valley_contrast_db": 0.0,
                "interval_cv": 0.0,
                "canister_spectral_ratio": 0.0,
                "noise_cancellation_active": self.enable_noise_cancellation,
                "snr_db": 0.0,
                "noise_reduction_db": 0.0,
                "noise_floor_rms": 0.0,
                "is_stage1_candidate": False,
                "stage1_reason": s1["reason"],
                "stage2_evaluated": False,
                "p_spray_can": 0.0,
                "p_loud_impact": 0.0,
                "p_continuous_environmental": 0.0,
                "p_hard_negative": 0.0,
                "p_ambient": 0.0,
                "max_competing_confidence": 0.0,
                "max_competing_class": "none",
                "confidence_margin": 0.0,
                "entropy": 0.0,
                "is_uncertain": False,
                "predicted_class": "none",
                "consecutive_windows": 0,
                "required_windows": self.required_consecutive_windows,
                "in_cooldown": in_cooldown,
                "cooldown_remaining_sec": round(cooldown_remaining, 2),
                "alert_fired": False,
                "status": "UNUSABLE_AUDIO_CLIPPING",
                "explanation": f"Unusable audio / digital clipping detected ({s1['reason']}). Alert prevented.",
            }
            self.history.append(result)
            logger.debug("Engine: Unusable audio clipping: %s", s1['reason'])
            return result


        # -------------------------------------------------------------
        # Adaptive Spectral Noise Cancellation (Background Room Noise, Fan Hum, Rain Bed)
        # Background adaptation freezes during candidate persistence, active alerts, or cooldown
        # -------------------------------------------------------------
        freeze_adaptation = bool(
            self.alert_episode_active or
            in_cooldown or
            (self.consecutive_positive_windows > 0) or
            self._was_recent_candidate
        )
        nc_metrics = {
            "snr_db": 0.0,
            "noise_reduction_db": 0.0,
            "noise_floor_rms": 0.0,
            "is_active": self.enable_noise_cancellation,
            "adaptation_frozen": freeze_adaptation,
        }
        if self.enable_noise_cancellation:
            processed_window, nc_metrics = self.noise_canceller.cancel_noise(
                audio_window, freeze_adaptation=freeze_adaptation
            )
        else:
            processed_window = audio_window

        # -------------------------------------------------------------
        # Stage 1: Kinematic Shaking Burst Candidate Pre-Filter & Clipping Check
        # -------------------------------------------------------------
        s1 = self.stage1.analyze_window(processed_window)
        self._was_recent_candidate = bool(s1.get("is_candidate", False))

        # Check if current window is verified quiet (clean ambient platform noise or silence)
        is_verified_quiet = bool(
            (s1["rms"] < STAGE1_MIN_RMS) or
            (not s1["is_candidate"] and s1["rms"] < 0.015 and s1.get("burst_count", 0) == 0)
        )

        if is_verified_quiet:
            self.consecutive_quiet_windows += 1
            if self.alert_episode_active and self.consecutive_quiet_windows >= self.required_quiet_windows:
                # Event-end confirmed: REARM detector state machine!
                self.alert_episode_active = False
                self.consecutive_positive_windows = 0
                logger.debug("Engine: Rearmed after %d verified quiet windows.", self.consecutive_quiet_windows)
        else:
            # Active bursts or elevated energy reset quiet count
            if s1.get("is_candidate", False) or s1.get("burst_count", 0) >= 2 or s1["rms"] >= 0.025:
                self.consecutive_quiet_windows = 0

        # Baseline result template
        result = {
            "timestamp": round(current_time, 3),
            "rms": s1["rms"],
            "peak": s1["peak"],
            "is_clipped": s1["is_clipped"],
            "burst_count": s1["burst_count"],
            "cadence_hz": s1["cadence_hz"],
            "temporal_span_sec": s1["temporal_span_sec"],
            "valley_contrast_db": s1["valley_contrast_db"],
            "interval_cv": s1.get("interval_cv", 0.0),
            "canister_spectral_ratio": s1.get("canister_spectral_ratio", 0.0),
            "noise_cancellation_active": self.enable_noise_cancellation,
            "adaptation_frozen": freeze_adaptation,
            "snr_db": nc_metrics["snr_db"],
            "noise_reduction_db": nc_metrics["noise_reduction_db"],
            "noise_floor_rms": nc_metrics["noise_floor_rms"],
            "is_stage1_candidate": s1["is_candidate"],
            "stage1_reason": s1["reason"],
            "stage2_evaluated": False,
            "p_spray_can": 0.0,
            "p_loud_impact": 0.0,
            "p_continuous_environmental": 0.0,
            "p_hard_negative": 0.0,
            "p_ambient": 0.0,
            "max_competing_confidence": 0.0,
            "max_competing_class": "none",
            "confidence_margin": 0.0,
            "entropy": 0.0,
            "is_uncertain": False,
            "predicted_class": "none",
            "consecutive_windows": self.consecutive_positive_windows,
            "required_windows": self.required_consecutive_windows,
            "in_cooldown": in_cooldown,
            "cooldown_remaining_sec": round(cooldown_remaining, 2),
            "alert_episode_active": self.alert_episode_active,
            "is_verified_quiet": is_verified_quiet,
            "alert_fired": False,
            "status": "STATUS: SILENT",
            "explanation": "",
        }

        # Rule 1: Digital clipping rejection (loudness used ONLY to identify clipping/unusable audio)
        if s1["is_clipped"]:
            self.consecutive_positive_windows = 0
            result["status"] = "UNUSABLE_AUDIO_CLIPPING"
            result["explanation"] = f"Unusable audio / digital clipping detected ({s1['reason']}). Alert prevented."
            self.history.append(result)
            logger.debug("Engine: Unusable audio clipping: %s", s1['reason'])
            return result

        # Silent or energy floor rejection
        if s1["rms"] < STAGE1_MIN_RMS:
            self.consecutive_positive_windows = 0
            self.consecutive_quiet_windows += 1
            if self.alert_episode_active and self.consecutive_quiet_windows >= self.required_quiet_windows:
                self.alert_episode_active = False
                logger.debug("Engine: Event ended after %d quiet windows. Detector rearmed.", self.consecutive_quiet_windows)
            result["status"] = "STATUS: SILENT"
            result["alert_episode_active"] = self.alert_episode_active
            result["is_event_active"] = self.alert_episode_active
            result["explanation"] = f"Ambient silence or low energy (RMS {s1['rms']:.4f} < {STAGE1_MIN_RMS})."
            self.history.append(result)
            return result

        # -------------------------------------------------------------
        # STAGE 1 GATE ENFORCEMENT:
        # Rule 3: Sound's time sequence MUST match repeated back-and-forth shaking,
        # NOT a single impact or an unrelated continuous sound.
        # If not a candidate, treat strictly as non-candidate. CANNOT ALERT.
        # -------------------------------------------------------------
        if not s1["is_candidate"]:
            self.consecutive_positive_windows = max(0, self.consecutive_positive_windows - 1)
            self.consecutive_quiet_windows += 1
            if self.alert_episode_active and self.consecutive_quiet_windows >= self.required_quiet_windows:
                self.alert_episode_active = False
                logger.debug("Engine: Event ended after %d non-candidate windows. Detector rearmed.", self.consecutive_quiet_windows)
            result["alert_fired"] = False
            result["is_stage1_candidate"] = False
            result["alert_episode_active"] = self.alert_episode_active
            result["is_event_active"] = self.alert_episode_active
            result["status"] = "NO_SHAKING_CADENCE"
            result["explanation"] = f"Stage 1 rejected: {s1['reason']}"
            self.history.append(result)
            return result

        # -------------------------------------------------------------
        # STAGE 2 CONFIRMATION:
        # Audio is a confirmed Stage 1 candidate (repeated bursts + span + contrast).
        # Now verify if the full spectro-temporal pattern matches spray can shaking.
        # -------------------------------------------------------------
        result["is_stage1_candidate"] = True

        self.reload_model_if_available()
        if not self.model_loaded:
            self.consecutive_positive_windows = 0
            result["status"] = "STAGE1_CANDIDATE (MODEL_NOT_READY)"
            result["explanation"] = "Candidate detected by Stage 1, but Stage 2 neural network checkpoint is not loaded."
            self.history.append(result)
            return result

        # Compute full 2D Log-Mel spectrogram on noise-cancelled audio (64 mels x 44 frames)
        log_mel = compute_log_mel_spectrogram(processed_window, sr=self.sr)

        if self.backend == "tensorflow":
            probs = self.model.predict_proba(log_mel)[0]
        else:
            tensor_in = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                probs = self.model.predict_proba(tensor_in).cpu().numpy()[0]
            del tensor_in

        s2_eval = evaluate_candidate_prediction(
            probs,
            conservative_threshold=self.conservative_threshold,
            max_competing_confidence=MAX_COMPETING_CONFIDENCE,
            max_env_confidence=MAX_ENV_CONFIDENCE,
            min_confidence_margin=MIN_CONFIDENCE_MARGIN,
            entropy_threshold=UNCERTAIN_ENTROPY_THRESHOLD,
        )

        result["stage2_evaluated"] = True
        result["p_spray_can"] = s2_eval["p_spray_can"]
        result["p_aerosol_spray_hiss"] = s2_eval.get("p_aerosol_spray_hiss", 0.0)
        result["p_loud_impact"] = s2_eval["p_loud_impact"]
        result["p_continuous_environmental"] = s2_eval["p_continuous_environmental"]
        result["p_hard_negative"] = s2_eval["p_hard_negative"]
        result["p_ambient"] = s2_eval["p_ambient"]
        result["max_competing_confidence"] = s2_eval["max_competing_confidence"]
        result["max_competing_class"] = s2_eval["max_competing_class"]
        result["confidence_margin"] = s2_eval["confidence_margin"]
        result["entropy"] = s2_eval["entropy"]
        result["is_uncertain"] = s2_eval["is_uncertain"]
        result["predicted_class"] = s2_eval["predicted_class"]

        # Rule 8: Structured Debug Logging of decision and confidences (never storing audio)
        logger.debug(
            "Evaluation: p_spray=%.3f, p_hiss=%.3f, max_comp=%.3f (%s), margin=%.3f, entropy=%.2f, verdict=%s",
            s2_eval["p_spray_can"],
            s2_eval.get("p_aerosol_spray_hiss", 0.0),
            s2_eval["max_competing_confidence"],
            s2_eval["max_competing_class"],
            s2_eval["confidence_margin"],
            s2_eval["entropy"],
            s2_eval["verdict"],
        )

        # Case A: Ambiguous / High entropy sound or uncertain evidence
        if s2_eval["is_uncertain"] or s2_eval["verdict"] == "UNCERTAIN_AMBIGUOUS_SOUND":
            self.consecutive_positive_windows = max(0, self.consecutive_positive_windows - 1)
            self.consecutive_quiet_windows += 1
            if self.alert_episode_active and self.consecutive_quiet_windows >= self.required_quiet_windows:
                self.alert_episode_active = False
                logger.debug("Engine: Shaking event ended after ambiguous gap. Detector rearmed.")
            result["status"] = "UNCERTAIN_AMBIGUOUS_SOUND"
            result["is_uncertain"] = True
            result["alert_episode_active"] = self.alert_episode_active
            result["is_event_active"] = self.alert_episode_active
            result["explanation"] = (
                f"Ambiguous acoustic evidence (Entropy {s2_eval['entropy']:.2f}, Margin {s2_eval['confidence_margin']:.2f}). "
                "No alert issued. Dispatched to operator review queue as uncertain candidate."
            )
            self._log_to_review_queue(
                audio_window=audio_window,
                evaluation=s2_eval,
                current_time=current_time,
                event_type="uncertain_candidate",
            )
            self.history.append(result)
            return result

        # Case B: Confirmed Spray Can Pattern
        if s2_eval["is_confirmed_spray"]:
            self.consecutive_positive_windows += 1
            self.consecutive_quiet_windows = 0
            result["consecutive_windows"] = self.consecutive_positive_windows

            # Persistence Gating: Must persist across consecutive windows
            if self.consecutive_positive_windows >= self.required_consecutive_windows:
                if not self.alert_episode_active:
                    # New distinct shaking event detected!
                    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    result["alert_fired"] = True
                    result["alert_title"] = ALERT_TITLE
                    result["status"] = f"ALERT: {ALERT_TITLE.upper()} (SHADOW MODE - ADVISORY ONLY)"
                    result["timestamp_iso"] = now_iso
                    result["confidence"] = round(s2_eval["p_spray_can"], 4)
                    result["shadow_mode"] = self.shadow_mode
                    result["dispatch_authorized"] = False
                    result["automatic_enforcement"] = False
                    result["human_verification_required"] = True
                    result["explanation"] = (
                        f"possible aerosol-can shaking detected (Confidence: {s2_eval['p_spray_can']*100:.1f}%, "
                        f"Timestamp: {now_iso}). Shadow Mode Active: {OPERATOR_REVIEW_NOTICE}"
                    )
                    result["operator_action_required"] = True
                    self._log_to_review_queue(
                        audio_window=audio_window,
                        evaluation=s2_eval,
                        current_time=current_time,
                        event_type="confirmed_alert",
                    )
                    self._log_shadow_alert(
                        timestamp_iso=now_iso,
                        evaluation=s2_eval,
                        current_time=current_time,
                    )
                    self.last_alert_time = current_time
                    self.alert_episode_active = True
                    self.consecutive_quiet_windows = 0
                    logger.info("SHADOW ALERT TRIGGERED: %s at %s (Advisory Only)", ALERT_TITLE, now_iso)
                else:
                    # Sustained shake continues within the active event:
                    # Model inference ran completely; event remains active without duplicate alert trigger.
                    result["alert_fired"] = False
                    result["status"] = "SUSTAINED_SHAKING_ACTIVE"
                    result["confidence"] = round(s2_eval["p_spray_can"], 4)
                    result["explanation"] = (
                        f"Sustained aerosol spray can shaking continues within active event "
                        f"(Confidence: {s2_eval['p_spray_can']*100:.1f}%, Persistence: {self.consecutive_positive_windows}). Event ongoing."
                    )
                result["alert_episode_active"] = True
                result["is_event_active"] = True
            else:
                result["status"] = f"STAGE1_CANDIDATE (Persistence: {self.consecutive_positive_windows}/{self.required_consecutive_windows})"
                result["explanation"] = (
                    f"Candidate matches spray can (P={s2_eval['p_spray_can']*100:.1f}%), "
                    f"awaiting persistence ({self.consecutive_positive_windows}/{self.required_consecutive_windows} windows) to confirm shaking."
                )
            self.history.append(result)
            return result

        # Case C: Not a confirmed spray can (Hiss, Impact, Environmental, Rattle, Speech, Low margin)
        if s2_eval["verdict"] in [
            "AEROSOL_SPRAY_HISS_REJECTED",
            "HARD_NEGATIVE_RATTLE_REJECTED",
            "LOUD_IMPACT_REJECTED",
            "CONTINUOUS_ENVIRONMENTAL_REJECTED",
        ]:
            self.consecutive_positive_windows = 0  # Strict immediate hard reset on verified confounder
        else:
            # Ambient speech syllable, background voice overlap, or brief uncertain fluctuation:
            # Leaky decay instead of brutal instant zeroing
            self.consecutive_positive_windows = max(0, self.consecutive_positive_windows - 1)

        self.consecutive_quiet_windows += 1
        if self.alert_episode_active and self.consecutive_quiet_windows >= self.required_quiet_windows:
            self.alert_episode_active = False
            logger.debug("Engine: Shaking event ended on non-spray sound. Detector rearmed.")

        result["alert_episode_active"] = self.alert_episode_active
        result["is_event_active"] = self.alert_episode_active

        if s2_eval["verdict"] == "AEROSOL_SPRAY_HISS_REJECTED":
            result["status"] = "AEROSOL_SPRAY_HISS_REJECTED"
            result["explanation"] = f"Rejected as aerosol spray hiss / nozzle discharge (P(hiss)={s2_eval.get('p_aerosol_spray_hiss', 0.0)*100:.1f}%). Spraying is not shaking. No alert."
        elif s2_eval["verdict"] == "LOUD_IMPACT_REJECTED":
            result["status"] = "LOUD_IMPACT_REJECTED"
            result["explanation"] = f"Rejected as sudden loud impact / shock (P(impact)={s2_eval['p_loud_impact']*100:.1f}%). No alert."
        elif s2_eval["verdict"] == "CONTINUOUS_ENVIRONMENTAL_REJECTED":
            result["status"] = "CONTINUOUS_ENVIRONMENTAL_REJECTED"
            result["explanation"] = f"Rejected as continuous environmental sound (P(env)={s2_eval['p_continuous_environmental']*100:.1f}%). No alert."
        elif s2_eval["verdict"] == "HARD_NEGATIVE_RATTLE_REJECTED":
            result["status"] = "HARD_NEGATIVE_RATTLE_REJECTED"
            result["explanation"] = f"Rejected as hard negative rattle (P(rattle)={s2_eval['p_hard_negative']*100:.1f}%). No alert."
        elif s2_eval["verdict"] == "AMBIENT_AND_SPEECH":
            result["status"] = "AMBIENT_AND_SPEECH"
            result["explanation"] = f"Rejected as speech / ambient vocalization (P(ambient)={s2_eval['p_ambient']*100:.1f}%). No alert."
        else:
            result["status"] = "UNCERTAIN_NO_ALERT"
            result["explanation"] = (
                f"Model cannot distinguish sound confidently (P(spray)={s2_eval['p_spray_can']*100:.1f}%, "
                f"margin={s2_eval['confidence_margin']*100:.1f}% < {MIN_CONFIDENCE_MARGIN*100:.0f}%). Returning uncertain/no alert."
            )

        self.history.append(result)
        return result

    def process_stream_chunk(self, chunk: np.ndarray, current_time: Optional[float] = None) -> Dict:
        """Feed a live streaming audio chunk into rolling buffer and evaluate."""
        if len(chunk) == 0:
            return self.evaluate_window(np.zeros(AUDIO_SAMPLES, dtype=np.float32), current_time)

        # Push chunk into rolling window
        chunk_len = len(chunk)
        if chunk_len >= AUDIO_SAMPLES:
            self.rolling_audio_buffer = chunk[-AUDIO_SAMPLES:].astype(np.float32)
        else:
            self.rolling_audio_buffer = np.roll(self.rolling_audio_buffer, -chunk_len)
            self.rolling_audio_buffer[-chunk_len:] = chunk.astype(np.float32)

        return self.evaluate_window(self.rolling_audio_buffer, current_time=current_time)

    def process_audio_file(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        hop_sec: float = 0.25,
    ) -> Dict:
        """Process an entire audio file in sliding windows and generate comprehensive diagnostics."""
        audio = load_audio(source, sr=self.sr)
        self.reset_state()
        total_len = len(audio)

        # For short clips (<= 2.0s), use smaller hop (0.15s) so multiple overlapping windows are tested
        actual_hop_sec = 0.15 if total_len <= int(self.sr * 2.0) else hop_sec
        hop_len = int(self.sr * actual_hop_sec)
        win_len = AUDIO_SAMPLES

        if total_len < win_len:
            audio = pad_or_trim(audio, win_len)
            total_len = len(audio)

        evaluations: List[Dict] = []
        alerts: List[Dict] = []
        time_offset = 0.0

        for start_idx in range(0, total_len - win_len + 1, hop_len):
            window = audio[start_idx : start_idx + win_len]
            t_win = start_idx / self.sr
            eval_res = self.evaluate_window(window, current_time=t_win)
            evaluations.append(eval_res)
            if eval_res["alert_fired"]:
                alerts.append(eval_res)

        # Overall summary diagnostics
        candidate_count = sum(1 for e in evaluations if e["is_stage1_candidate"])
        stage2_confirmed_count = sum(1 for e in evaluations if e.get("p_spray_can", 0) >= self.conservative_threshold)
        max_spray_p = max((e.get("p_spray_can", 0.0) for e in evaluations), default=0.0)
        max_hiss_p = max((e.get("p_aerosol_spray_hiss", 0.0) for e in evaluations), default=0.0)
        max_impact_p = max((e.get("p_loud_impact", 0.0) for e in evaluations), default=0.0)
        max_env_p = max((e.get("p_continuous_environmental", 0.0) for e in evaluations), default=0.0)
        max_rattle_p = max((e.get("p_hard_negative", 0.0) for e in evaluations), default=0.0)
        has_clipping = any(e.get("is_clipped", False) for e in evaluations)

        duration_sec = total_len / self.sr
        alert_triggered = len(alerts) > 0

        if alert_triggered:
            primary_verdict = "CONFIRMED_SPRAY_CAN_ALERT"
            summary_desc = (
                f"Alert confirmed! Detected {len(alerts)} sustained spray can event(s). "
                f"Max P(spray)={max_spray_p*100:.1f}%."
            )
        elif has_clipping:
            primary_verdict = "REJECTED_DIGITAL_CLIPPING"
            summary_desc = "Audio contains saturated digital clipping / unusable signal. Alert suppressed."
        elif max_hiss_p >= 0.30:
            primary_verdict = "REJECTED_AEROSOL_SPRAY_HISS"
            summary_desc = f"Classified as aerosol spray hiss / nozzle discharge (Max P(hiss)={max_hiss_p*100:.1f}%). Spraying is not shaking. Alert suppressed."
        elif max_impact_p >= 0.35:
            primary_verdict = "REJECTED_LOUD_IMPACT"
            summary_desc = f"Classified as loud impact / shock (Max P(impact)={max_impact_p*100:.1f}%). Alert suppressed."
        elif max_env_p >= 0.35:
            primary_verdict = "REJECTED_CONTINUOUS_ENVIRONMENTAL"
            summary_desc = f"Classified as continuous environmental sound (Max P(env)={max_env_p*100:.1f}%). Alert suppressed."
        elif max_rattle_p >= 0.35:
            primary_verdict = "REJECTED_HARD_NEGATIVE_RATTLE"
            summary_desc = f"Classified as hard negative rattle (coins/keys/pill bottle, Max P(rattle)={max_rattle_p*100:.1f}%). Alert suppressed."
        elif candidate_count == 0:
            primary_verdict = "REJECTED_NO_SHAKING_CADENCE"
            summary_desc = "No repeated back-and-forth shaking cadence detected by Stage 1."
        elif max_spray_p < self.conservative_threshold:
            primary_verdict = "UNCERTAIN_NO_ALERT"
            summary_desc = (
                f"Candidate shaking detected but Stage 2 confidence P(spray)={max_spray_p*100:.1f}% "
                f"did not meet conservative threshold ({self.conservative_threshold*100:.0f}%). Returning uncertain/no alert."
            )
        else:
            primary_verdict = "REJECTED_INSUFFICIENT_PERSISTENCE"
            summary_desc = (
                f"Spray-like bursts observed, but did not persist across "
                f"{self.required_consecutive_windows} consecutive windows. Ruled out as transient rattle."
            )

        return {
            "duration_sec": round(duration_sec, 2),
            "total_duration_sec": round(duration_sec, 2),
            "total_windows_evaluated": len(evaluations),
            "candidate_windows_count": candidate_count,
            "stage1_candidates_count": candidate_count,
            "stage2_confirmed_windows": stage2_confirmed_count,
            "alerts_fired_count": len(alerts),
            "confirmed_alerts_count": len(alerts),
            "alert_triggered": alert_triggered,
            "primary_verdict": primary_verdict,
            "final_verdict": primary_verdict,
            "summary_description": summary_desc,
            "max_spray_confidence": round(max_spray_p, 4),
            "max_hard_neg_confidence": round(max(max_rattle_p, max_hiss_p, max_impact_p, max_env_p), 4),
            "max_confidence": {
                "spray_can_shaking": round(max_spray_p, 4),
                "aerosol_spray_hiss": round(max_hiss_p, 4),
                "loud_impact": round(max_impact_p, 4),
                "continuous_environmental": round(max_env_p, 4),
                "hard_negative_rattle": round(max_rattle_p, 4),
            },
            "evaluations": evaluations,
            "timeline": evaluations,
            "alerts": alerts,
        }

    def _log_to_review_queue(
        self,
        audio_window: np.ndarray,
        evaluation: Dict,
        current_time: float,
        event_type: str,
    ) -> None:
        """Log ambiguous or alerted event to the ephemeral operator review queue.

        Privacy rule:
        - Stores only anonymous acoustic window for human validation.
        - Automatically purged according to AUDIO_RETENTION_HOURS (24 hours).
        """
        if not ENABLE_EPHEMERAL_REVIEW:
            return

        self._purge_expired_review_items()

        event_id = f"evt_{int(time.time() * 1000)}"
        timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        wav_filename = f"{event_id}_{event_type}.wav"
        wav_path = REVIEW_QUEUE_DIR / wav_filename

        try:
            # Ephemeral audio buffer: save 1.0s window for human verification
            sf.write(str(wav_path), audio_window, self.sr, subtype="PCM_16")
        except Exception as e:
            logger.warning("Could not persist review audio snippet: %s", e)
            wav_filename = None

        record = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp_iso": timestamp_iso,
            "system_time": round(current_time, 3),
            "p_spray_can": float(evaluation.get("p_spray_can", 0.0)),
            "p_aerosol_spray_hiss": float(evaluation.get("p_aerosol_spray_hiss", 0.0)),
            "p_hard_negative": float(evaluation.get("p_hard_negative", 0.0)),
            "p_continuous_environmental": float(evaluation.get("p_continuous_environmental", 0.0)),
            "p_ambient": float(evaluation.get("p_ambient", 0.0)),
            "entropy": float(evaluation.get("entropy", 0.0)),
            "margin": float(evaluation.get("confidence_margin", 0.0)),
            "predicted_class": evaluation.get("predicted_class", "unknown"),
            "verdict": evaluation.get("verdict", "unknown"),
            "audio_file": wav_filename,
            "human_reviewed": False,
            "human_verdict": None,
            "operator_notes": None,
            "retention_policy": f"Ephemeral review buffer. Auto-purged after {AUDIO_RETENTION_HOURS}h.",
        }

        meta_path = REVIEW_QUEUE_DIR / f"{event_id}.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception as e:
            logger.warning("Could not write review metadata: %s", e)

        self.review_queue.append(record)

    def _log_shadow_alert(
        self,
        timestamp_iso: str,
        evaluation: Dict,
        current_time: float,
    ) -> None:
        """Log advisory shadow mode alert to append-only JSONL log."""
        try:
            alert_entry = {
                "alert_id": f"shadow_{int(time.time() * 1000)}",
                "timestamp_iso": timestamp_iso,
                "system_time": round(current_time, 3),
                "shadow_mode": True,
                "dispatch_authorized": False,
                "automatic_enforcement": False,
                "human_verification_required": True,
                "confidence_p_spray": float(evaluation.get("p_spray_can", 0.0)),
                "confidence_margin": float(evaluation.get("confidence_margin", 0.0)),
                "entropy": float(evaluation.get("entropy", 0.0)),
                "operator_notice": OPERATOR_REVIEW_NOTICE,
            }
            with open(SHADOW_ALERTS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_entry) + "\n")
        except Exception as e:
            logger.warning("Could not append shadow alert log: %s", e)

    def _purge_expired_review_items(self) -> None:
        """Purge review records and audio files older than AUDIO_RETENTION_HOURS."""
        cutoff_epoch = time.time() - (AUDIO_RETENTION_HOURS * 3600)
        if REVIEW_QUEUE_DIR.exists():
            try:
                for item in REVIEW_QUEUE_DIR.glob("*"):
                    if item.stat().st_mtime < cutoff_epoch:
                        try:
                            item.unlink()
                        except Exception:
                            pass
            except Exception as e:
                logger.warning("Error running privacy retention purge on review queue: %s", e)

        if FIELD_SESSIONS_DIR.exists():
            try:
                for sess_dir in FIELD_SESSIONS_DIR.iterdir():
                    if sess_dir.is_dir() and sess_dir.stat().st_mtime < cutoff_epoch:
                        # Auto-purge expired continuous session folder
                        import shutil
                        shutil.rmtree(sess_dir, ignore_errors=True)
            except Exception as e:
                logger.warning("Error running privacy retention purge on field sessions: %s", e)

    def get_review_queue(self) -> List[Dict]:
        """Fetch all events in the review queue ordered by newest first."""
        self._purge_expired_review_items()
        records = []
        if REVIEW_QUEUE_DIR.exists():
            for json_file in sorted(
                REVIEW_QUEUE_DIR.glob("evt_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        records.append(json.load(f))
                except Exception:
                    pass
        return records[:MAX_REVIEW_QUEUE_ITEMS]

    def resolve_review_event(
        self,
        event_id: str,
        human_verdict: str,
        operator_notes: str = "",
    ) -> bool:
        """Record a trained human operator's verdict on an uncertain or alerted event."""
        json_path = REVIEW_QUEUE_DIR / f"{event_id}.json"
        if not json_path.exists():
            return False
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["human_reviewed"] = True
            record["human_verdict"] = human_verdict
            record["operator_notes"] = operator_notes
            record["reviewed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            return True
        except Exception as e:
            logger.error("Failed to update review record: %s", e)
            return False
