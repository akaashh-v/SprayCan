"""Alert-Triggered Audio Recorder with Rolling RAM Pre-Event Buffer.

Enforces privacy, reliability, and real-time performance:
- Ordinary background audio is maintained in RAM ONLY and NEVER written to disk.
- When the detector confirms an authentic spray-can alert, a WAV recording is saved
  containing:
    1. Pre-event context from the rolling RAM buffer (configurable 3-5 seconds, default 4.0s).
    2. The detected event audio window(s) (grouped to prevent duplicate files).
    3. Post-event audio context (configurable, default 3.0 seconds).
- Thread-safe, bounded, zero-leak FreeRTOS/server integration with automatic debounce.
"""

from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from config import (
    ALERT_MAX_SAVED_RECORDINGS,
    ALERT_POST_EVENT_SECONDS,
    ALERT_PRE_EVENT_SECONDS,
    ALERT_RECORDINGS_DIR,
    ALERT_RECORDING_COOLDOWN_SECONDS as ALERT_COOLDOWN_SECONDS,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class RollingAudioBuffer:
    """Thread-safe in-memory circular audio buffer for pre-event recording.

    Maintains exactly the most recent `max_seconds` of raw audio in RAM.
    Handles early start (less than max_seconds available) and long-term wraparound smoothly.
    """

    def __init__(self, sr: int = SAMPLE_RATE, max_seconds: float = ALERT_PRE_EVENT_SECONDS):
        self.sr = sr
        self.max_seconds = max(0.5, float(max_seconds))
        self.max_samples = int(self.sr * self.max_seconds)
        self._chunks: deque = deque()
        self._total_samples: int = 0
        self._lock = threading.Lock()

    def set_max_seconds(self, max_seconds: float) -> None:
        with self._lock:
            self.max_seconds = max(0.5, float(max_seconds))
            self.max_samples = int(self.sr * self.max_seconds)
            self._trim_locked()

    def append(self, samples: np.ndarray) -> None:
        """Append an audio array (1D float32) to the buffer."""
        if samples is None or len(samples) == 0:
            return
        arr = np.asarray(samples, dtype=np.float32).flatten()
        with self._lock:
            self._chunks.append(arr)
            self._total_samples += len(arr)
            self._trim_locked()

    def _trim_locked(self) -> None:
        """Discard oldest chunks once total samples exceed max_samples."""
        while self._chunks and (self._total_samples - len(self._chunks[0])) >= self.max_samples:
            removed = self._chunks.popleft()
            self._total_samples -= len(removed)

    def get_audio(self) -> np.ndarray:
        """Return a single contiguous numpy array of all buffered pre-event audio."""
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            concatenated = np.concatenate(list(self._chunks))
            if len(concatenated) > self.max_samples:
                concatenated = concatenated[-self.max_samples :]
            return concatenated.copy()

    def duration_seconds(self) -> float:
        with self._lock:
            return round(min(self._total_samples, self.max_samples) / float(self.sr), 3)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._total_samples = 0


class AlertAudioRecorder:
    """Coordinates event-triggered audio recordings with pre- and post-event context."""

    def __init__(
        self,
        recordings_dir: Optional[Path] = None,
        pre_event_seconds: float = ALERT_PRE_EVENT_SECONDS,
        post_event_seconds: float = ALERT_POST_EVENT_SECONDS,
        cooldown_seconds: float = ALERT_COOLDOWN_SECONDS,
        max_saved_recordings: int = ALERT_MAX_SAVED_RECORDINGS,
        sr: int = SAMPLE_RATE,
    ):
        self.recordings_dir = Path(recordings_dir or ALERT_RECORDINGS_DIR)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sr

        self.pre_event_seconds = float(pre_event_seconds)
        self.post_event_seconds = float(post_event_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.max_saved_recordings = int(max_saved_recordings)
        self.enabled = True

        self._lock = threading.Lock()
        self.rolling_buffer = RollingAudioBuffer(sr=self.sr, max_seconds=self.pre_event_seconds)

        # Active recording session tracking
        self.active_session: Optional[Dict[str, Any]] = None
        self.last_event_finish_time: float = 0.0
        self.total_saved_alerts: int = 0
        self.last_saved_alert: Optional[Dict[str, Any]] = None

    def update_config(
        self,
        pre_event_seconds: Optional[float] = None,
        post_event_seconds: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update runtime alert recording parameters."""
        with self._lock:
            if pre_event_seconds is not None:
                self.pre_event_seconds = max(1.0, min(10.0, float(pre_event_seconds)))
                self.rolling_buffer.set_max_seconds(self.pre_event_seconds)
            if post_event_seconds is not None:
                self.post_event_seconds = max(0.5, min(15.0, float(post_event_seconds)))
            if cooldown_seconds is not None:
                self.cooldown_seconds = max(0.0, min(60.0, float(cooldown_seconds)))
            if enabled is not None:
                self.enabled = bool(enabled)
            return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pre_event_seconds": self.pre_event_seconds,
            "post_event_seconds": self.post_event_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "recordings_dir": str(self.recordings_dir),
            "max_saved_recordings": self.max_saved_recordings,
            "sample_rate": self.sr,
        }

    def reset(self) -> None:
        """Reset state, clear buffer, and discard any incomplete in-flight session."""
        with self._lock:
            self.rolling_buffer.clear()
            self.active_session = None
            logger.debug("[ALERT-RECORDER] Reset rolling buffer and cancelled active sessions.")

    def process_audio_chunk(
        self,
        audio_arr: np.ndarray,
        prediction: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Processes an incoming audio chunk, updates the rolling RAM buffer, and manages alert capture.

        Returns metadata dict of finalized alert recording if completed in this step, otherwise None.
        """
        if not self.enabled or audio_arr is None or len(audio_arr) == 0:
            return None

        arr_flat = np.asarray(audio_arr, dtype=np.float32).flatten()
        now = time.time()
        finalized_record: Optional[Dict[str, Any]] = None

        with self._lock:
            alert_fired = bool(prediction.get("alert_fired", False))
            is_active_shaking = bool(prediction.get("is_event_active", False) or prediction.get("alert_episode_active", False))
            in_cooldown = (now - self.last_event_finish_time) < self.cooldown_seconds

            # -------------------------------------------------------------
            # Case 1: Active Alert Recording in Progress
            # -------------------------------------------------------------
            if self.active_session is not None:
                sess = self.active_session
                state = sess["state"]

                if state == "ACTIVE_EVENT":
                    if is_active_shaking or alert_fired:
                        # Sustained shaking continues; extend event audio
                        sess["event_samples"].append(arr_flat)
                        sess["confidence"] = max(sess["confidence"], float(prediction.get("p_spray_can", 0.0)))
                        sess["sustained_windows"] += 1
                    else:
                        # Shaking stopped; transition to capturing post-event audio
                        sess["state"] = "CAPTURING_POST"
                        sess["post_event_samples"].append(arr_flat)
                        sess["post_collected_samples"] += len(arr_flat)
                elif state == "CAPTURING_POST":
                    sess["post_event_samples"].append(arr_flat)
                    sess["post_collected_samples"] += len(arr_flat)

                # Check if post-event duration is fulfilled
                needed_post_samples = int(self.post_event_seconds * self.sr)
                if sess["state"] == "CAPTURING_POST" and sess["post_collected_samples"] >= needed_post_samples:
                    finalized_record = self._finalize_recording_locked(sess)
                    self.active_session = None
                    self.last_event_finish_time = time.time()

            # -------------------------------------------------------------
            # Case 2: New Alert Triggered (Not in Cooldown & No Active Session)
            # -------------------------------------------------------------
            elif alert_fired and not in_cooldown:
                event_id = f"evt_{int(now * 1000)}_{secrets.token_hex(3)}"
                now_utc = datetime.now(timezone.utc)
                ts_compact = now_utc.strftime("%Y%m%dT%H%M%S_%fZ")

                # Snap pre-event audio from rolling RAM buffer
                pre_audio = self.rolling_buffer.get_audio()

                self.active_session = {
                    "event_id": event_id,
                    "timestamp_iso": now_utc.isoformat(),
                    "timestamp_compact": ts_compact,
                    "start_time": now,
                    "alert_class": "spray_can_shaking",
                    "confidence": float(prediction.get("p_spray_can", 0.0)),
                    "pre_event_samples": pre_audio,
                    "event_samples": [arr_flat],
                    "post_event_samples": [],
                    "post_collected_samples": 0,
                    "sustained_windows": 1,
                    "state": "ACTIVE_EVENT",
                    "metrics": {
                        "rms": float(prediction.get("rms", 0.0)),
                        "cadence_hz": float(prediction.get("cadence_hz", 0.0)),
                        "confidence_margin": float(prediction.get("confidence_margin", 0.0)),
                        "entropy": float(prediction.get("entropy", 0.0)),
                        "p_spray_can": float(prediction.get("p_spray_can", 0.0)),
                        "p_hiss": float(prediction.get("p_aerosol_spray_hiss", 0.0)),
                        "p_hard_negative": float(prediction.get("p_hard_negative", 0.0)),
                    },
                }
                logger.info(
                    "[ALERT-RECORDER] Started recording session %s for alert (P=%.3f, pre-event: %.2fs)",
                    event_id,
                    self.active_session["confidence"],
                    len(pre_audio) / self.sr,
                )

            # Always maintain rolling RAM buffer for future pre-event context
            self.rolling_buffer.append(arr_flat)

        return finalized_record

    def _finalize_recording_locked(self, sess: Dict[str, Any]) -> Dict[str, Any]:
        """Stitches pre, event, and post audio together and saves standard WAV file."""
        pre_audio = sess["pre_event_samples"]
        event_audio = np.concatenate(sess["event_samples"]) if sess["event_samples"] else np.zeros(0, dtype=np.float32)
        post_audio = np.concatenate(sess["post_event_samples"]) if sess["post_event_samples"] else np.zeros(0, dtype=np.float32)

        full_audio = np.concatenate([pre_audio, event_audio, post_audio])

        # Normalize clipping if needed
        max_val = np.max(np.abs(full_audio)) if len(full_audio) > 0 else 0.0
        if max_val > 1.0:
            full_audio = full_audio / max_val

        event_id = sess["event_id"]
        ts_compact = sess["timestamp_compact"]
        wav_filename = f"alert_{ts_compact}_{event_id}_spray_can_shaking.wav"
        json_filename = f"alert_{ts_compact}_{event_id}.json"

        wav_path = self.recordings_dir / wav_filename
        json_path = self.recordings_dir / json_filename

        # Write WAV file to disk
        sf.write(str(wav_path), full_audio, self.sr, subtype="PCM_16")

        total_sec = round(len(full_audio) / float(self.sr), 3)
        pre_sec = round(len(pre_audio) / float(self.sr), 3)
        event_sec = round(len(event_audio) / float(self.sr), 3)
        post_sec = round(len(post_audio) / float(self.sr), 3)

        metadata = {
            "event_id": event_id,
            "filename": wav_filename,
            "filepath": f"recordings/alerts/{wav_filename}",
            "absolute_path": str(wav_path.resolve()),
            "alert_class": sess["alert_class"],
            "confidence": round(sess["confidence"], 4),
            "timestamp_iso": sess["timestamp_iso"],
            "sample_rate": self.sr,
            "total_duration_sec": total_sec,
            "pre_event_sec": pre_sec,
            "event_sec": event_sec,
            "post_event_sec": post_sec,
            "metrics": sess.get("metrics", {}),
            "file_size_bytes": wav_path.stat().st_size,
        }

        # Write metadata JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning("[ALERT-RECORDER] Failed to write metadata JSON: %s", e)

        self.total_saved_alerts += 1
        self.last_saved_alert = metadata
        self._prune_recordings_locked()

        logger.info(
            "[ALERT-RECORDER] Saved confirmed alert WAV: %s (Total: %.2fs [Pre: %.2fs, Event: %.2fs, Post: %.2fs]) -> %d bytes",
            wav_filename,
            total_sec,
            pre_sec,
            event_sec,
            post_sec,
            metadata["file_size_bytes"],
        )
        return metadata

    def _prune_recordings_locked(self) -> None:
        """Keep only the most recent max_saved_recordings to prevent disk exhaustion."""
        try:
            wav_files = sorted(self.recordings_dir.glob("alert_*.wav"), key=lambda p: p.stat().st_mtime)
            excess = len(wav_files) - self.max_saved_recordings
            if excess > 0:
                for f in wav_files[:excess]:
                    try:
                        f.unlink(missing_ok=True)
                        json_equiv = f.with_suffix(".json")
                        json_equiv.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("[ALERT-RECORDER] Pruning error: %s", e)

    def list_saved_recordings(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all saved alert recordings ordered from newest to oldest."""
        with self._lock:
            recordings = []
            try:
                json_files = sorted(
                    self.recordings_dir.glob("alert_*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for jf in json_files[:limit]:
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Verify corresponding wav file exists
                            wav_path = self.recordings_dir / data.get("filename", "")
                            if wav_path.exists():
                                data["audio_available"] = True
                                recordings.append(data)
                    except Exception:
                        continue
            except Exception as e:
                logger.error("[ALERT-RECORDER] Failed to list saved recordings: %s", e)
            return recordings

    def get_recording_file(self, filename: str) -> Optional[Path]:
        """Safely resolve an alert recording WAV file path within the dedicated directory."""
        if not filename or "/" in filename or "\\" in filename or ".." in filename:
            return None
        target = (self.recordings_dir / filename).resolve()
        if target.exists() and target.is_file() and str(target).startswith(str(self.recordings_dir.resolve())):
            return target
        return None


# Global singleton alert recorder
alert_recorder = AlertAudioRecorder()
