"""ESP32 Remote Microphone Controller & State Manager.

Coordinates Wi-Fi command polling, status acknowledgement, authenticated audio reception,
and real-time streaming pipeline evaluation for ESP32-WROVER-R and INMP441 hardware.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import queue
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alert_recorder import alert_recorder
from audio_utils import load_audio, pad_or_trim
from config import (
    AUDIO_SAMPLES,
    ESP32_AUDIO_SAVE_DIR,
    ESP32_DEVICE_TOKEN,
    ESP32_HEARTBEAT_TIMEOUT,
    ESP32_MAX_SAVED_CHUNKS,
    ESP32_SAVE_AUDIO_CHUNKS,
    ESP32_START_TIMEOUT,
    ESP32_STOP_TIMEOUT,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class ESP32State:
    STOPPED = "Stopped"
    STARTING = "Starting"
    LISTENING = "Listening"
    STOPPING = "Stopping"
    CONNECTION_ERROR = "Connection Error"


class ESP32Manager:
    """Thread-safe manager for ESP32 remote microphone state and audio ingest."""

    def __init__(
        self,
        device_token: Optional[str] = None,
        save_dir: Optional[Path] = None,
        start_timeout: float = ESP32_START_TIMEOUT,
        stop_timeout: float = ESP32_STOP_TIMEOUT,
        heartbeat_timeout: float = ESP32_HEARTBEAT_TIMEOUT,
    ):
        self._lock = threading.Lock()
        self.device_token = device_token or ESP32_DEVICE_TOKEN
        self.save_dir = Path(save_dir or ESP32_AUDIO_SAVE_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.start_timeout = start_timeout
        self.stop_timeout = stop_timeout
        self.heartbeat_timeout = heartbeat_timeout

        # State tracking
        self.state: str = ESP32State.STOPPED
        self.desired_state: str = "STOP"  # "START" or "STOP"
        self.transition_time: float = 0.0

        self.last_heartbeat_time: float = 0.0
        self.last_audio_time: float = 0.0
        self.last_audio_file: Optional[str] = None
        self.last_audio_bytes: int = 0
        self.audio_chunk_count: int = 0
        self.device_ip: Optional[str] = None
        self.connection_error_reason: Optional[str] = None

        # Cached latest detector analysis
        self.latest_prediction: Optional[Dict[str, Any]] = None
        self.last_confirmed_alert_time: float = 0.0

        # Real-time SSE subscribers for push telemetry (<2ms dashboard updates)
        self._subscribers: List[queue.Queue] = []

        # Logging trackers to prevent log spamming while maintaining complete visibility
        self._last_logged_state: Optional[str] = None
        self._last_logged_command: Optional[str] = None
        self._last_poll_log_time: float = 0.0

    def _check_timeouts_locked(self, now: float) -> None:
        """Internal helper to transition states on timeout (must hold self._lock)."""
        if self.state == ESP32State.STARTING:
            if now - self.transition_time > self.start_timeout:
                self.state = ESP32State.CONNECTION_ERROR
                self.connection_error_reason = (
                    f"ESP32 did not acknowledge start command within {int(self.start_timeout)}s. "
                    "Check ESP32 power and Wi-Fi connection."
                )
                logger.warning("ESP32 start timed out: %s", self.connection_error_reason)

        elif self.state == ESP32State.STOPPING:
            if now - self.transition_time > self.stop_timeout:
                # Failsafe: force stopped if stop acknowledgement took too long
                self.state = ESP32State.STOPPED
                self.connection_error_reason = None
                logger.info("[ESP32] Stop timed out -> force transitioned to Stopped")

        elif self.state == ESP32State.LISTENING:
            if self.last_heartbeat_time > 0 and (now - self.last_heartbeat_time > self.heartbeat_timeout):
                self.state = ESP32State.CONNECTION_ERROR
                self.connection_error_reason = (
                    f"ESP32 audio transmission lost (no contact for >{int(self.heartbeat_timeout)}s). "
                    "Microphone stream disconnected."
                )
                logger.warning("ESP32 stream lost: %s", self.connection_error_reason)

    def request_start(self) -> Dict[str, Any]:
        """Request the ESP32 to start recording audio."""
        with self._lock:
            now = time.time()
            if self.state == ESP32State.LISTENING:
                return {
                    "success": True,
                    "state": self.state,
                    "message": "ESP32 is already actively listening",
                }

            self.desired_state = "START"
            self.state = ESP32State.STARTING
            self.transition_time = now
            self.connection_error_reason = None
            self.audio_chunk_count = 0
            alert_recorder.reset()
            logger.info("[DASHBOARD-CONTROL] Start requested from dashboard -> state: Starting (desired: START)")
            return {
                "success": True,
                "state": self.state,
                "message": "Start command queued for ESP32",
            }

    def request_stop(self) -> Dict[str, Any]:
        """Request the ESP32 to stop recording audio."""
        with self._lock:
            now = time.time()
            alert_recorder.reset()
            if self.state == ESP32State.STOPPED:
                return {
                    "success": True,
                    "state": self.state,
                    "message": "ESP32 is already stopped",
                }

            self.desired_state = "STOP"
            self.state = ESP32State.STOPPING
            self.transition_time = now
            logger.info("[DASHBOARD-CONTROL] Stop requested from dashboard -> state: Stopping (desired: STOP)")
            return {
                "success": True,
                "state": self.state,
                "message": "Stop command queued for ESP32",
            }

    def get_status_dict(self) -> Dict[str, Any]:
        """Return public status dictionary for the dashboard (never exposes sensitive tokens)."""
        with self._lock:
            now = time.time()
            self._check_timeouts_locked(now)

            return {
                "state": self.state,
                "desired_state": self.desired_state,
                "device_ip": self.device_ip,
                "last_heartbeat_time": round(self.last_heartbeat_time, 2) if self.last_heartbeat_time else None,
                "last_audio_time": round(self.last_audio_time, 2) if self.last_audio_time else None,
                "seconds_since_heartbeat": round(now - self.last_heartbeat_time, 1) if self.last_heartbeat_time else None,
                "last_audio_file": self.last_audio_file,
                "last_audio_bytes": self.last_audio_bytes,
                "audio_chunk_count": self.audio_chunk_count,
                "connection_error_reason": self.connection_error_reason,
                "latest_prediction": self.latest_prediction,
                "alert_recorder_config": alert_recorder.get_config(),
                "total_saved_alerts": alert_recorder.total_saved_alerts,
                "last_saved_alert": alert_recorder.last_saved_alert,
            }

    def validate_device_token(self, supplied_token: str) -> bool:
        """Verify device token using constant-time comparison."""
        if not supplied_token or not self.device_token:
            return False
        return secrets.compare_digest(supplied_token.strip(), self.device_token.strip())

    def handle_command_poll(
        self,
        supplied_token: str,
        client_ip: str,
        reported_state: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Handle periodic command poll from ESP32."""
        if not self.validate_device_token(supplied_token):
            return {"error": "Unauthorized device token"}, 403

        with self._lock:
            now = time.time()
            self.last_heartbeat_time = now
            self.device_ip = client_ip

            # If ESP32 reports its current operational state
            if reported_state:
                rep_norm = reported_state.strip().upper()
                if rep_norm in ("LISTENING", "RECORDING", "A"):
                    if self.desired_state != "STOP":
                        self.state = ESP32State.LISTENING
                        self.connection_error_reason = None
                elif rep_norm in ("STOPPED", "IDLE", "B"):
                    if self.desired_state == "STOP" or self.state == ESP32State.STOPPING:
                        self.state = ESP32State.STOPPED
                        self.connection_error_reason = None

            self._check_timeouts_locked(now)

            command = "START" if self.desired_state == "START" else "STOP"

            # Clean log: output on state/command change or every 10 seconds for idle heartbeat
            state_changed = (self.state != self._last_logged_state or command != self._last_logged_command)
            time_since_log = now - self._last_poll_log_time

            if state_changed or time_since_log >= 10.0:
                self._last_logged_state = self.state
                self._last_logged_command = command
                self._last_poll_log_time = now
                logger.info(
                    "[COMMAND-POLL] Device %s polled /esp32/command (reported: %s) -> responding: %s (manager state: %s)",
                    client_ip,
                    reported_state or "N/A",
                    command,
                    self.state,
                )

            return {
                "command": command,
                "desired_state": self.desired_state,
                "server_time": now,
            }, 200

    def handle_status_update(
        self,
        supplied_token: str,
        client_ip: str,
        reported_state: str,
    ) -> Tuple[Dict[str, Any], int]:
        """Handle explicit state change confirmation (ACK) from ESP32."""
        if not self.validate_device_token(supplied_token):
            return {"error": "Unauthorized device token"}, 403

        with self._lock:
            now = time.time()
            self.last_heartbeat_time = now
            self.device_ip = client_ip

            rep_norm = reported_state.strip().upper()
            if rep_norm in ("LISTENING", "RECORDING"):
                self.state = ESP32State.LISTENING
                self.connection_error_reason = None
                logger.info("[STATE-ACK] ESP32 acknowledged state: LISTENING from %s", client_ip)
            elif rep_norm in ("STOPPED", "IDLE"):
                self.state = ESP32State.STOPPED
                self.connection_error_reason = None
                logger.info("[STATE-ACK] ESP32 acknowledged state: STOPPED from %s", client_ip)

            command = "START" if self.desired_state == "START" else "STOP"
            return {
                "status": "ok",
                "state": self.state,
                "command": command,
            }, 200

    def _prune_saved_audio_files(self) -> None:
        """Keep only the most recent ESP32_MAX_SAVED_CHUNKS files to prevent disk exhaustion."""
        try:
            files = sorted(self.save_dir.glob("esp32_*.wav"), key=lambda p: p.stat().st_mtime)
            excess = len(files) - ESP32_MAX_SAVED_CHUNKS
            if excess > 0:
                for f in files[:excess]:
                    try:
                        f.unlink()
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("Prune saved audio error: %s", e)

    def subscribe_events(self) -> queue.Queue:
        """Subscribe to real-time prediction and waveform stream events."""
        with self._lock:
            q: queue.Queue = queue.Queue(maxsize=10)
            self._subscribers.append(q)
            return q

    def unsubscribe_events(self, q: queue.Queue) -> None:
        """Unsubscribe from stream events."""
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast_prediction(self, payload: Dict[str, Any]) -> None:
        """Instantly broadcast evaluation payload to all connected SSE clients (<1ms latency)."""
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(payload)
                    except Exception:
                        dead.append(q)
            for d in dead:
                if d in self._subscribers:
                    self._subscribers.remove(d)

    def handle_edge_alert(
        self,
        supplied_token: str,
        client_ip: str,
        alert_payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int]:
        """Process instantaneous lightweight alert notification directly from ESP32 edge detection engine."""
        t_rx = time.perf_counter()
        if not self.validate_device_token(supplied_token):
            return {"error": "Unauthorized device token"}, 403

        event_id = alert_payload.get("event_id") or f"evt_edge_{int(time.time()*1000)}"
        alert_type = (alert_payload.get("alert_type") or "CONFIRMED").upper()
        verdict = alert_payload.get("verdict") or "CONFIRMED_SPRAY_CAN_SHAKING"
        p_spray = float(alert_payload.get("p_spray", 0.95))
        confidence_margin = float(alert_payload.get("confidence_margin", 0.0))
        entropy = float(alert_payload.get("entropy", 0.0))
        predicted_class = alert_payload.get("predicted_class", "spray_can_shaking")
        capture_seq = int(alert_payload.get("capture_seq", 0))
        timings = alert_payload.get("timings", {})

        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        is_confirmed = (alert_type == "CONFIRMED")
        is_preliminary = (alert_type == "PRELIMINARY")

        t_dispatch = time.perf_counter()
        dispatch_ms = round((t_dispatch - t_rx) * 1000, 2)

        timing_info = dict(timings)
        timing_info["server_dispatch_ms"] = dispatch_ms
        timing_info["server_rx_time_iso"] = now_iso

        device_total_ms = float(
            timings.get("total_device_ms", 0.0)
            or (timings.get("cnn_ms", 0) + timings.get("dsp_ms", 0) + timings.get("s1_eval_ms", 0))
        )
        net_tx_ms = float(timings.get("net_tx_ms", 0.0))
        end_to_end_ms = round(device_total_ms + net_tx_ms + dispatch_ms, 2)
        timing_info["end_to_end_server_ms"] = end_to_end_ms

        pred_event = {
            "event_id": event_id,
            "alert_type": alert_type,
            "alert_fired": is_confirmed,
            "is_preliminary": is_preliminary,
            "status": (
                "ALERT: POSSIBLE AEROSOL-CAN SHAKING DETECTED"
                if is_confirmed
                else ("PRELIMINARY: SHAKING CANDIDATE DETECTED" if is_preliminary else "REJECTED_BY_CNN")
            ),
            "p_spray_can": p_spray,
            "confidence_margin": confidence_margin,
            "entropy": entropy,
            "predicted_class": predicted_class,
            "is_stage1_candidate": is_preliminary or is_confirmed,
            "timestamp_iso": now_iso,
            "source": "ESP32_EDGE_CNN",
            "device_ip": client_ip,
            "chunk_seq": capture_seq,
            "timing": timing_info,
            "explanation": (
                f"On-device CNN verified spray-can shaking (Confidence: {p_spray*100:.1f}%, Margin: +{confidence_margin:.2f}). Direct human operator verification requested."
                if is_confirmed
                else (
                    "Preliminary shaking kinematic burst detected by Stage 1. On-device CNN evaluating window..."
                    if is_preliminary
                    else f"On-device CNN rejected candidate: classified as {predicted_class} (Confidence: {p_spray*100:.1f}%)."
                )
            ),
        }

        with self._lock:
            self.last_heartbeat_time = time.time()
            self.device_ip = client_ip
            self.latest_prediction = pred_event
            if is_confirmed:
                self.last_confirmed_alert_time = time.time()

        # Broadcast INSTANTLY to dashboard via SSE (< 1 ms push latency)
        self._broadcast_prediction(pred_event)

        logger.info(
            "[EDGE-ALERT] Instant %s alert broadcast (ID: %s, P=%.4f) in %.2fms server dispatch | Device: %.1fms, NetTx: %.1fms, Total: %.1fms",
            alert_type,
            event_id,
            p_spray,
            dispatch_ms,
            device_total_ms,
            net_tx_ms,
            end_to_end_ms,
        )

        return {
            "status": "ok",
            "event_id": event_id,
            "alert_type": alert_type,
            "server_dispatch_ms": dispatch_ms,
            "command": "CONTINUE" if self.desired_state != "STOP" else "STOP",
        }, 200

    def handle_audio_upload(
        self,
        supplied_token: str,
        audio_bytes: bytes,
        mimetype: str,
        client_ip: str,
        engine: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Validate, process in-memory, and execute the detection pipeline on an incoming ESP32 audio chunk."""
        t_receipt = time.perf_counter()
        if not self.validate_device_token(supplied_token):
            return {"error": "Unauthorized device token"}, 403

        if mimetype and "audio/wav" not in mimetype and mimetype != "application/octet-stream":
            return {"error": "Expected audio/wav"}, 415

        if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            return {"error": "Invalid WAV file"}, 400

        with self._lock:
            # Immediate stop enforcement: If system is stopped or stopping, discard chunk without evaluation
            if self.desired_state == "STOP" or self.state in (ESP32State.STOPPING, ESP32State.STOPPED):
                self.state = ESP32State.STOPPED
                self.last_heartbeat_time = time.time()
                self.device_ip = client_ip
                logger.info(
                    "[AUDIO-UPLOAD] In-flight chunk received while stopped/stopping from %s (%d bytes) -> discarded; responding command: STOP",
                    client_ip,
                    len(audio_bytes),
                )
                return {
                    "status": "stopped",
                    "command": "STOP",
                    "bytes": len(audio_bytes),
                    "file": None,
                    "alert_fired": False,
                    "p_spray": 0.0,
                    "message": "Recording is stopped. Audio chunk discarded.",
                }, 201

        now_dt = datetime.now(timezone.utc)
        timestamp = now_dt.strftime("%Y%m%dT%H%M%S_%fZ")
        virtual_filename = f"esp32_{timestamp}.wav"

        # Bounded disk save: only if explicitly enabled (defaults to False for zero disk latency)
        if ESP32_SAVE_AUDIO_CHUNKS:
            try:
                target_file = self.save_dir / virtual_filename
                with target_file.open("xb") as f:
                    f.write(audio_bytes)
                self._prune_saved_audio_files()
            except Exception as e:
                logger.warning("Failed to write ESP32 WAV file %s: %s", virtual_filename, e)

        # Parse timing & sequence headers sent from ESP32
        cap_start_ms = 0.0
        cap_end_ms = 0.0
        tx_start_ms = 0.0
        client_capture_seq = 0
        client_upload_seq = 0
        edge_telemetry = None
        if headers:
            try:
                cap_start_ms = float(headers.get("X-Capture-Start-Ms", 0) or 0)
                cap_end_ms = float(headers.get("X-Capture-End-Ms", 0) or 0)
                tx_start_ms = float(headers.get("X-Tx-Start-Ms", 0) or 0)
                client_capture_seq = int(headers.get("X-Capture-Seq", 0) or headers.get("X-Chunk-Seq", 0) or 0)
                client_upload_seq = int(headers.get("X-Upload-Seq", 0) or 0)
                if headers.get("X-Edge-Evaluated") or headers.get("X-Edge-Verdict"):
                    edge_telemetry = {
                        "evaluated": True,
                        "verdict": headers.get("X-Edge-Verdict", "UNKNOWN"),
                        "p_spray": float(headers.get("X-Edge-P-Spray", 0.0) or 0.0),
                        "predicted_class": headers.get("X-Edge-Class", "unknown"),
                        "inference_time_ms": float(headers.get("X-Edge-Infer-Ms", 0) or 0),
                        "preprocess_time_ms": float(headers.get("X-Edge-Prep-Ms", 0) or 0),
                    }
            except Exception:
                pass

        # Execute existing detector engine on the incoming audio in-memory
        prediction_result: Dict[str, Any] = {}
        t_decode_start = time.perf_counter()
        try:
            audio_arr = load_audio(audio_bytes, sr=SAMPLE_RATE)
            if len(audio_arr) != AUDIO_SAMPLES:
                audio_arr = pad_or_trim(audio_arr, AUDIO_SAMPLES)
            t_decode_end = time.perf_counter()

            t_infer_start = time.perf_counter()
            prediction_result = engine.evaluate_window(audio_arr)
            t_infer_end = time.perf_counter()

            # Downsample waveform (128 points) for smooth dashboard oscilloscope rendering
            step = max(1, len(audio_arr) // 128)
            waveform_downsampled = [round(float(x), 4) for x in audio_arr[::step][:128]]
            prediction_result["waveform_points"] = waveform_downsampled
            prediction_result["audio_file"] = virtual_filename
            prediction_result["source"] = "ESP32_INMP441"
            prediction_result["device_ip"] = client_ip

            # Latency profiling metrics
            decode_ms = round((t_decode_end - t_decode_start) * 1000, 2)
            infer_ms = round((t_infer_end - t_infer_start) * 1000, 2)
            server_total_ms = round((time.perf_counter() - t_receipt) * 1000, 2)
            timing_info = {
                "server_decode_ms": decode_ms,
                "server_inference_ms": infer_ms,
                "server_total_ms": server_total_ms,
            }
            if cap_end_ms > 0 and cap_start_ms > 0:
                timing_info["capture_dur_ms"] = round(cap_end_ms - cap_start_ms, 1)
            prediction_result["timing"] = timing_info

            if edge_telemetry:
                prediction_result["edge_device"] = edge_telemetry
                if edge_telemetry.get("verdict") == "CONFIRMED_ALERT" or edge_telemetry.get("p_spray", 0.0) >= 0.80:
                    prediction_result["alert_fired"] = True
                    prediction_result["status"] = "ALERT: POSSIBLE AEROSOL-CAN SHAKING DETECTED"
                logger.info(
                    "[EDGE-CNN] ESP32 on-device prediction: verdict=%s, p_spray=%.4f, class=%s, infer=%d ms",
                    edge_telemetry["verdict"],
                    edge_telemetry["p_spray"],
                    edge_telemetry["predicted_class"],
                    int(edge_telemetry["inference_time_ms"]),
                )

            # Event-based alert audio recording (rolling RAM buffer + WAV persistence on confirmed alert)
            saved_alert = alert_recorder.process_audio_chunk(audio_arr, prediction_result)
            if saved_alert:
                prediction_result["saved_recording"] = saved_alert
                prediction_result["recording_file"] = saved_alert["filename"]

        except Exception as e:
            logger.error("Error executing detector pipeline on ESP32 audio: %s", e, exc_info=True)
            prediction_result = {
                "error": str(e),
                "is_stage1_candidate": False,
                "alert_fired": False,
                "rms": 0.0,
            }

        with self._lock:
            now = time.time()
            self.last_heartbeat_time = now
            self.last_audio_time = now
            self.last_audio_file = virtual_filename
            self.last_audio_bytes = len(audio_bytes)
            self.audio_chunk_count += 1
            seq_to_use = client_capture_seq if client_capture_seq > 0 else self.audio_chunk_count
            prediction_result["chunk_seq"] = seq_to_use
            prediction_result["server_seq"] = self.audio_chunk_count
            prediction_result["client_capture_seq"] = client_capture_seq
            prediction_result["client_upload_seq"] = client_upload_seq
            if client_capture_seq > 0:
                prediction_result["dropped_chunks_estimate"] = max(0, client_capture_seq - self.audio_chunk_count)
            self.device_ip = client_ip
            self.latest_prediction = prediction_result

            # Any incoming valid audio chunk confirms the ESP32 is actively listening
            if self.desired_state != "STOP":
                self.state = ESP32State.LISTENING
                self.connection_error_reason = None

            # Determine response command
            command = "STOP" if self.desired_state == "STOP" else "CONTINUE"

        # Broadcast immediately to connected dashboard SSE clients (<1ms latency)
        self._broadcast_prediction(prediction_result)

        logger.info(
            "[AUDIO-UPLOAD] Processed chunk #%d (CapSeq=#%d, UpSeq=#%d, %d bytes) from %s -> P(spray)=%.3f, Alert=%s, Command=%s (server: %.2fms)",
            self.audio_chunk_count,
            client_capture_seq,
            client_upload_seq,
            len(audio_bytes),
            client_ip,
            prediction_result.get("p_spray_can", 0.0),
            prediction_result.get("alert_fired", False),
            command,
            prediction_result.get("timing", {}).get("server_total_ms", 0.0),
        )

        resp_data = {
            "status": "received",
            "command": command,
            "bytes": len(audio_bytes),
            "file": virtual_filename,
            "chunk_seq": self.audio_chunk_count,
            "server_seq": self.audio_chunk_count,
            "client_capture_seq": client_capture_seq,
            "client_upload_seq": client_upload_seq,
            "dropped_chunks_estimate": max(0, client_capture_seq - self.audio_chunk_count) if client_capture_seq > 0 else 0,
            "alert_fired": prediction_result.get("alert_fired", False),
            "p_spray": prediction_result.get("p_spray_can", 0.0),
            "timing": prediction_result.get("timing", {}),
        }
        if "saved_recording" in prediction_result:
            resp_data["saved_recording"] = prediction_result["saved_recording"]
            resp_data["recording_file"] = prediction_result["recording_file"]
        if "edge_device" in prediction_result:
            resp_data["edge_device"] = prediction_result["edge_device"]
        return resp_data, 201


# Global shared instance
esp32_manager = ESP32Manager()
