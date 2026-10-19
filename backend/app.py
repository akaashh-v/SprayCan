"""Flask Web Application for Precision Aerosol Spray Can Sound Detector.

Features:
- Live streaming analysis from browser microphone via Web Audio API.
- File upload diagnostics with timeline evaluation and Log-Mel spectrogram visualization.
- One-click testing with pre-generated authentic spray and hard-negative samples.
- Interactive visualization of Stage 1 kinematic meters, Stage 2 probabilities,
  multi-window persistence indicators, and threshold calibration curves.
"""

import base64
import io
import json
import logging
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Dict

# Ensure backend directory is in sys.path for direct execution
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory, session

from admin_auth import (
    authenticate_user,
    admin_required,
    generate_csrf_token,
    get_audit_logs,
    get_current_user,
    log_audit_action,
    login_required,
    validate_csrf,
)
from dataset_manager import (
    convert_and_validate_audio,
    create_new_category,
    delete_recording,
    list_categories,
    query_recordings,
    save_recording_sample,
    update_recording_status,
    validate_dataset_for_training,
)
from training_manager import training_manager

from audio_utils import (
    compute_log_mel_spectrogram,
    generate_ambient_room_noise,
    generate_ambient_speech,
    generate_env_rain,
    generate_env_spray_hiss,
    generate_env_train,
    generate_env_water_drops,
    generate_hard_negative_bottle_caps,
    generate_hard_negative_coins,
    generate_hard_negative_containers,
    generate_hard_negative_glassware,
    generate_hard_negative_keys,
    generate_hard_negative_liquid_bottle,
    generate_hard_negative_pill_bottle,
    generate_impact_clap,
    generate_impact_desk_bang,
    generate_impact_door_slam,
    generate_impact_dropped_object,
    generate_spray_continuous_hiss,
    generate_spray_nozzle_sputter,
    generate_spray_pulsed_bursts,
    load_audio,
    pad_or_trim,
    resample_audio_poly,
)
from config import (
    AUDIO_RETENTION_HOURS,
    AUDIO_SAMPLES,
    BASE_DIR,
    CALIBRATION_PATH,
    CLASS_NAMES,
    CONSERVATIVE_THRESHOLD,
    EVALUATION_REPORT_PATH,
    FIELD_ANNOTATIONS_DIR,
    FIELD_RECORDINGS_DIR,
    FIELD_SESSIONS_DIR,
    FRONTEND_DIR,
    MAX_REVIEW_QUEUE_ITEMS,
    RAW_SPRAY_DIR,
    REVIEW_QUEUE_DIR,
    SAMPLE_RATE,
    SAVE_MIC_AUDIO,
    SHADOW_ALERTS_LOG,
    SHADOW_MODE_ENABLED,
    STATIC_DIR,
    TEMPLATES_DIR,
    ESP32_RECEIVER_HOST,
    ESP32_RECEIVER_PORT,
)
from detector_engine import SprayCanDetectorEngine
from esp32_manager import esp32_manager, ESP32State
from alert_recorder import alert_recorder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HttpRequestCategoryFilter(logging.Filter):
    """Tags incoming HTTP requests with clear functional categories."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if "/api/esp32/status" in msg and not str(record.msg).startswith("[DASHBOARD-STATUS-POLL]"):
                record.msg = f"[DASHBOARD-STATUS-POLL] {record.msg}"
            elif "/esp32/command" in msg and not str(record.msg).startswith("[COMMAND-POLL]"):
                record.msg = f"[COMMAND-POLL] {record.msg}"
            elif "/esp32/test-audio" in msg and not str(record.msg).startswith("[AUDIO-UPLOAD]"):
                record.msg = f"[AUDIO-UPLOAD] {record.msg}"
            elif "/esp32/status" in msg and not str(record.msg).startswith("[STATE-ACK]"):
                record.msg = f"[STATE-ACK] {record.msg}"
        except Exception:
            pass
        return True


logging.getLogger("werkzeug").addFilter(HttpRequestCategoryFilter())

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "spray_can_detector_super_secret_session_key_2026")
is_production = bool(os.environ.get("RENDER") or os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("PRODUCTION"))
if is_production or os.environ.get("CROSS_ORIGIN_COOKIES", "").lower() in ("true", "1"):
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
    app.config["SESSION_COOKIE_SECURE"] = True
else:
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

try:
    from flask_cors import CORS
    CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "*"}}, allow_headers=["Content-Type", "X-CSRF-Token", "Authorization"])
except Exception as _cors_err:
    logger.warning("flask_cors not initialized: %s", _cors_err)


@app.before_request
def handle_options_preflight():
    if request.method == "OPTIONS":
        res = app.make_default_options_response()
        origin = request.headers.get("Origin")
        if origin:
            res.headers["Access-Control-Allow-Origin"] = origin
            res.headers["Access-Control-Allow-Credentials"] = "true"
            res.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            res.headers["Access-Control-Allow-Headers"] = "Content-Type, X-CSRF-Token, Authorization"
        return res


@app.after_request
def apply_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-CSRF-Token, Authorization"
    return response

engine = SprayCanDetectorEngine()

# Ensure demo samples directory exists
DEMO_DIR = BASE_DIR / "data" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def ensure_demo_samples():
    """Ensure authentic spray, loud impact, continuous environmental, and hard-negative audio samples exist."""
    # 1. Authentic Spray Samples from diverse cans & fill states
    spray_demos = {
        "authentic_spray_can.wav": "1.mp3",
        "authentic_spray_can_mini.wav": "preview_spray_can_rattle_mini_can.mp3",
        "authentic_spray_can_empty.wav": "preview_spray_can_shake_nearly_empty.mp3",
        "authentic_spray_paint.wav": "preview_reg7783_spray_paint_rattle.mp3",
    }

    for demo_name, src_name in spray_demos.items():
        dst_path = DEMO_DIR / demo_name
        src_path = RAW_SPRAY_DIR / src_name
        if not dst_path.exists():
            if src_path.exists():
                audio = load_audio(src_path, sr=SAMPLE_RATE)
                dur = min(len(audio), int(SAMPLE_RATE * 3.0))
                sf.write(str(dst_path), audio[:dur], SAMPLE_RATE)
            elif (RAW_SPRAY_DIR / "1.mp3").exists():
                audio = load_audio(RAW_SPRAY_DIR / "1.mp3", sr=SAMPLE_RATE)
                dur = min(len(audio), int(SAMPLE_RATE * 3.0))
                sf.write(str(dst_path), audio[:dur], SAMPLE_RATE)

    # 2. Loud Impacts, Continuous Environmental, Hard Negatives & Ambient
    sample_generators = {
        "impact_door_slam.wav": generate_impact_door_slam,
        "impact_desk_bang.wav": generate_impact_desk_bang,
        "impact_dropped_object.wav": generate_impact_dropped_object,
        "impact_clap.wav": generate_impact_clap,
        "env_rain.wav": generate_env_rain,
        "env_water_drops.wav": generate_env_water_drops,
        "env_train.wav": generate_env_train,
        "spray_hiss_continuous.wav": generate_spray_continuous_hiss,
        "spray_hiss_pulsed_bursts.wav": generate_spray_pulsed_bursts,
        "spray_hiss_nozzle_sputter.wav": generate_spray_nozzle_sputter,
        "keys_jangling.wav": generate_hard_negative_keys,
        "coins_rattling.wav": generate_hard_negative_coins,
        "pill_bottle.wav": generate_hard_negative_pill_bottle,
        "bottle_caps.wav": generate_hard_negative_bottle_caps,
        "container_shake.wav": generate_hard_negative_containers,
        "glassware.wav": generate_hard_negative_glassware,
        "liquid_bottle.wav": generate_hard_negative_liquid_bottle,
        "ambient_speech.wav": generate_ambient_speech,
        "ambient_silence.wav": generate_ambient_room_noise,
    }

    for fname, gen_fn in sample_generators.items():
        sample_path = DEMO_DIR / fname
        if not sample_path.exists():
            sig = gen_fn(duration=3.0, sr=SAMPLE_RATE)
            sf.write(str(sample_path), sig, SAMPLE_RATE)


@app.route("/")
def index():
    """Render main application dashboard."""
    user = session.get("user")
    is_admin = bool(user and user.get("role") == "admin")
    return render_template("index.html", is_admin=is_admin, user=user)


@app.route("/admin")
@admin_required
def admin_page():
    """Admin-only dashboard route."""
    user = session.get("user")
    return render_template("index.html", is_admin=True, user=user, active_admin=True)


@app.route("/css/<path:filename>")
def serve_css(filename):
    """Serve frontend CSS stylesheet assets."""
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    """Serve frontend JavaScript scripts."""
    return send_from_directory(str(FRONTEND_DIR / "js"), filename)


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve frontend static media & images."""
    return send_from_directory(str(FRONTEND_DIR / "assets"), filename)



@app.route("/api/status", methods=["GET"])
def get_status():
    """Get system health, model load state, and active calibration parameters."""
    engine.reload_model_if_available()
    calib = {}
    if CALIBRATION_PATH.exists():
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                calib = json.load(f)
        except Exception:
            pass

    return jsonify({
        "status": "ready" if engine.model_loaded else "standby",
        "model_loaded": engine.model_loaded,
        "enable_noise_cancellation": engine.enable_noise_cancellation,
        "conservative_threshold": engine.conservative_threshold,
        "required_consecutive_windows": engine.required_consecutive_windows,
        "cooldown_seconds": engine.cooldown_seconds,
        "sample_rate": SAMPLE_RATE,
        "classes": CLASS_NAMES,
        "calibration": calib,
        "shadow_mode": getattr(engine, "shadow_mode", True),
        "audio_retention_hours": AUDIO_RETENTION_HOURS,
        "automatic_enforcement_allowed": False,
        "esp32": esp32_manager.get_status_dict(),
    })


# =========================================================================
# ESP32 Hardware & Web Dashboard Integration Routes
# =========================================================================

_esp32_server_thread = None


def start_esp32_receiver_service(host: str = ESP32_RECEIVER_HOST, port: int = ESP32_RECEIVER_PORT):
    """Start the dedicated ESP32 hardware receiver on port 5001 in a background daemon thread."""
    global _esp32_server_thread
    if _esp32_server_thread is not None and _esp32_server_thread.is_alive():
        return

    from werkzeug.serving import make_server
    try:
        srv = make_server(host, port, app, threaded=True)
        t = threading.Thread(target=srv.serve_forever, daemon=True, name="ESP32Receiver5001")
        t.start()
        _esp32_server_thread = t
        logger.info("ESP32 hardware receiver daemon active on http://%s:%d", host, port)
    except OSError as e:
        logger.error(
            "FATAL: ESP32 receiver could not bind to %s:%d (%s). "
            "Another process is already using port %d. Please terminate existing instances.",
            host, port, e, port
        )
        raise RuntimeError(f"Port {port} is already in use. Please terminate existing processes before starting run.py.")
    except Exception as e:
        logger.error("ESP32 receiver service error: %s", e)
        raise


@app.route("/api/esp32/status", methods=["GET"])
def api_esp32_status():
    """Return live ESP32 recording state, connection health, and latest audio detection results."""
    return jsonify(esp32_manager.get_status_dict())


@app.route("/api/esp32/start", methods=["POST"])
def api_esp32_start():
    """Trigger the ESP32 to start recording audio via local Wi-Fi."""
    res = esp32_manager.request_start()
    return jsonify(res)


@app.route("/api/esp32/stop", methods=["POST"])
def api_esp32_stop():
    """Trigger the ESP32 to stop recording audio."""
    res = esp32_manager.request_stop()
    return jsonify(res)


@app.route("/api/esp32/latest_prediction", methods=["GET"])
def api_esp32_latest_prediction():
    """Return the most recent detection analysis evaluated from the ESP32 stream."""
    pred = esp32_manager.latest_prediction
    if pred is None:
        return jsonify({"status": "no_prediction_yet", "alert_fired": False})
    return jsonify(pred)


@app.route("/api/esp32/stream", methods=["GET"])
def api_esp32_stream():
    """Server-Sent Events (SSE) stream for real-time ESP32 predictions & waveforms (<1ms push latency)."""
    q = esp32_manager.subscribe_events()

    def event_stream():
        try:
            # Send initial status snapshot immediately on connect
            initial_status = esp32_manager.get_status_dict()
            yield f"event: status\ndata: {json.dumps(initial_status)}\n\n"

            while True:
                try:
                    payload = q.get(timeout=2.0)
                    yield f"event: prediction\ndata: {json.dumps(payload)}\n\n"
                except queue.Empty:
                    # Keep-alive heartbeat ping to prevent proxy/browser timeout
                    yield ": ping\n\n"
        except GeneratorExit:
            pass
        finally:
            esp32_manager.unsubscribe_events(q)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/esp32/command", methods=["GET"])
def esp32_command_poll():
    """Command polling endpoint for ESP32. Returns START or STOP command."""
    supplied_token = request.headers.get("X-Device-Token", "")
    reported_state = request.args.get("state") or request.headers.get("X-Device-State")
    client_ip = request.remote_addr or "unknown"
    resp, code = esp32_manager.handle_command_poll(supplied_token, client_ip, reported_state=reported_state)
    return jsonify(resp), code


@app.route("/esp32/status", methods=["POST"])
def esp32_status_ack():
    """State acknowledgement endpoint called by ESP32 to confirm state transitions."""
    supplied_token = request.headers.get("X-Device-Token", "")
    data = request.get_json(silent=True) or {}
    reported_state = data.get("state") or request.headers.get("X-Device-State", "")
    client_ip = request.remote_addr or "unknown"
    resp, code = esp32_manager.handle_status_update(supplied_token, client_ip, reported_state=reported_state)
    return jsonify(resp), code


@app.route("/esp32/alert", methods=["POST"])
def esp32_receive_alert():
    """Receive instant lightweight detection alert (< 300 bytes) from ESP32 edge CNN."""
    supplied_token = request.headers.get("X-Device-Token", "")
    client_ip = request.remote_addr or "unknown"
    data = request.get_json(silent=True) or {}
    resp, code = esp32_manager.handle_edge_alert(supplied_token, client_ip, data)
    response = jsonify(resp)
    response.headers["Connection"] = "close"
    return response, code


@app.route("/esp32/test-audio", methods=["POST"])
def esp32_receive_audio():
    """Receive, save, and process 1.0-second 22050 Hz WAV audio uploaded by ESP32."""
    supplied_token = request.headers.get("X-Device-Token", "")
    client_ip = request.remote_addr or "unknown"
    audio_data = request.get_data(cache=False)
    resp, code = esp32_manager.handle_audio_upload(
        supplied_token=supplied_token,
        audio_bytes=audio_data,
        mimetype=request.mimetype,
        client_ip=client_ip,
        engine=engine,
        headers=dict(request.headers),
    )
    response = jsonify(resp)
    response.headers["Connection"] = "close"
    return response, code


@app.route("/esp32/health", methods=["GET"])
def esp32_health():
    """Health check endpoint for ESP32 receiver service."""
    status = esp32_manager.get_status_dict()
    return jsonify({
        "status": "ok",
        "service": "ESP32 acoustic detector receiver",
        "state": status["state"],
        "desired_state": status["desired_state"],
    })


@app.route("/api/alerts/recordings", methods=["GET"])
def get_alert_recordings():
    """Return list of confirmed alert event recordings with metadata."""
    recordings = alert_recorder.list_saved_recordings(limit=100)
    return jsonify({
        "status": "success",
        "recordings": recordings,
        "count": len(recordings),
        "total_saved": alert_recorder.total_saved_alerts,
    })


@app.route("/api/alerts/recordings/<path:filename>", methods=["GET"])
def get_alert_recording_file(filename: str):
    """Stream a confirmed alert WAV audio recording for browser playback/download."""
    wav_path = alert_recorder.get_recording_file(filename)
    if not wav_path or not wav_path.exists():
        return jsonify({"error": "Alert recording not found"}), 404
    return send_file(str(wav_path), mimetype="audio/wav")


@app.route("/api/alerts/config", methods=["GET"])
def get_alert_config():
    """Return runtime configuration for event-based alert audio recording."""
    return jsonify({
        "status": "success",
        "config": alert_recorder.get_config(),
    })


@app.route("/api/alerts/config", methods=["POST"])
def update_alert_config():
    """Update runtime configuration for event-based alert audio recording."""
    data = request.get_json(force=True, silent=True) or {}
    updated = alert_recorder.update_config(
        pre_event_seconds=data.get("pre_event_seconds"),
        post_event_seconds=data.get("post_event_seconds"),
        cooldown_seconds=data.get("cooldown_seconds"),
        enabled=data.get("enabled"),
    )
    return jsonify({
        "status": "success",
        "config": updated,
    })




@app.route("/api/config", methods=["POST"])
@admin_required
def update_config():
    """Dynamically update threshold, persistence settings, or noise cancellation."""
    data = request.get_json(force=True, silent=True) or {}
    if "enable_noise_cancellation" in data:
        engine.enable_noise_cancellation = bool(data["enable_noise_cancellation"])
    if "conservative_threshold" in data:
        new_th = float(data["conservative_threshold"])
        new_th = max(0.50, min(0.99, new_th))
        engine.conservative_threshold = new_th
    if "required_consecutive_windows" in data:
        new_p = int(data["required_consecutive_windows"])
        new_p = max(1, min(5, new_p))
        engine.required_consecutive_windows = new_p
    if "cooldown_seconds" in data:
        new_cd = float(data["cooldown_seconds"])
        new_cd = max(0.0, min(10.0, new_cd))
        engine.cooldown_seconds = new_cd
    return jsonify({
        "status": "success",
        "enable_noise_cancellation": engine.enable_noise_cancellation,
        "conservative_threshold": engine.conservative_threshold,
        "required_consecutive_windows": engine.required_consecutive_windows,
        "cooldown_seconds": engine.cooldown_seconds,
    })


@app.route("/api/predict_window", methods=["POST"])
def predict_window():
    """Process a single 1.0-second audio window received from browser microphone."""
    try:
        data = request.get_json(force=True, silent=True) or {}

        if "audio" in data:
            # Array of floats
            audio_arr = np.array(data["audio"], dtype=np.float32)
        elif "audio_base64" in data:
            raw_bytes = base64.b64decode(data["audio_base64"])
            if raw_bytes.startswith(b"RIFF"):
                audio_arr = load_audio(raw_bytes, sr=SAMPLE_RATE)
            else:
                audio_arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            return jsonify({"error": "No audio payload provided"}), 400

        native_sr = int(data.get("native_sample_rate") or data.get("sample_rate") or SAMPLE_RATE)
        if native_sr != SAMPLE_RATE and len(audio_arr) != AUDIO_SAMPLES:
            audio_arr = resample_audio_poly(audio_arr, orig_sr=native_sr, target_sr=SAMPLE_RATE)

        # Ensure audio array matches exactly AUDIO_SAMPLES (22,050 samples)
        if len(audio_arr) != AUDIO_SAMPLES:
            audio_arr = pad_or_trim(audio_arr, AUDIO_SAMPLES)

        result = engine.evaluate_window(audio_arr)
        return jsonify(result)
    except Exception as e:
        logger.error("Error in predict_window: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate_file", methods=["POST"])
def evaluate_file():
    """Evaluate an uploaded audio file across sliding windows with full diagnostics."""
    if "audio" not in request.files and "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    uploaded = request.files.get("audio") or request.files.get("file")
    file_bytes = uploaded.read()

    try:
        audio = load_audio(file_bytes, sr=SAMPLE_RATE)
    except Exception as e:
        return jsonify({"error": f"Failed to decode audio file: {str(e)}"}), 400

    # Process sliding windows
    file_results = engine.process_audio_file(audio)

    # Generate Log-Mel Spectrogram image for visualization
    try:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")

        # Spectrogram of first 2-3 seconds
        clip_len = min(len(audio), int(SAMPLE_RATE * 3.5))
        clip_audio = audio[:clip_len]
        spec = compute_log_mel_spectrogram(clip_audio, sr=SAMPLE_RATE)

        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, clip_len / SAMPLE_RATE, 100, 11000],
        )
        ax.set_title("Log-Mel Spectrogram (Full Spectro-Temporal Pattern)", color="#E5E7EB", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (seconds)", color="#9CA3AF", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color="#9CA3AF", fontsize=9)
        ax.tick_params(colors="#9CA3AF", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#374151")

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="#9CA3AF")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#9CA3AF", fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        spec_base64 = base64.b64encode(buf.read()).decode("utf-8")
        file_results["spectrogram_image_base64"] = spec_base64
    except Exception as e:
        logger.warning("Spectrogram rendering failed: %s", e)
        file_results["spectrogram_image_base64"] = None

    return jsonify(file_results)


@app.route("/api/demo_samples", methods=["GET"])
def list_demo_samples():
    """List available pre-generated test audio files."""
    ensure_demo_samples()
    samples = []
    category_map = {
        "authentic_spray_can.wav": {"label": "Authentic Spray Can (User Sample)", "type": "positive", "badge": "TARGET SPRAY"},
        "authentic_spray_can_mini.wav": {"label": "Mini Aerosol Can Rattle", "type": "positive", "badge": "TARGET SPRAY"},
        "authentic_spray_can_empty.wav": {"label": "Nearly Empty Can Shake", "type": "positive", "badge": "TARGET SPRAY"},
        "authentic_spray_paint.wav": {"label": "Spray Paint Metal Ball Rattle", "type": "positive", "badge": "TARGET SPRAY"},
        "impact_door_slam.wav": {"label": "Loud Door Slam (Sudden Impact)", "type": "loud_impact", "badge": "LOUD IMPACT"},
        "impact_desk_bang.wav": {"label": "Heavy Desk Bang (Sudden Impact)", "type": "loud_impact", "badge": "LOUD IMPACT"},
        "impact_dropped_object.wav": {"label": "Dropped Heavy Item (Sudden Impact)", "type": "loud_impact", "badge": "LOUD IMPACT"},
        "impact_clap.wav": {"label": "Sharp Hand Clap (Sudden Impact)", "type": "loud_impact", "badge": "LOUD IMPACT"},
        "env_rain.wav": {"label": "Continuous Rain Shower", "type": "environmental", "badge": "ENVIRONMENTAL"},
        "env_water_drops.wav": {"label": "Water Drops in Basin", "type": "environmental", "badge": "ENVIRONMENTAL"},
        "env_train.wav": {"label": "Train Low Rumble Passing", "type": "environmental", "badge": "ENVIRONMENTAL"},
        "spray_hiss_continuous.wav": {"label": "Continuous Spray Hiss (Negative)", "type": "aerosol_spray_hiss", "badge": "NEGATIVE: HISS"},
        "spray_hiss_pulsed_bursts.wav": {"label": "Pulsed Spray Bursts (Negative)", "type": "aerosol_spray_hiss", "badge": "NEGATIVE: HISS"},
        "spray_hiss_nozzle_sputter.wav": {"label": "Nozzle Discharge Sputter (Negative)", "type": "aerosol_spray_hiss", "badge": "NEGATIVE: HISS"},
        "keys_jangling.wav": {"label": "Keys Jangling (Hard Negative)", "type": "hard_negative", "badge": "HARD NEGATIVE"},
        "coins_rattling.wav": {"label": "Coins in Jar (Hard Negative)", "type": "hard_negative", "badge": "HARD NEGATIVE"},
        "pill_bottle.wav": {"label": "Pill Bottle Rattle (Hard Negative)", "type": "hard_negative", "badge": "HARD NEGATIVE"},
        "bottle_caps.wav": {"label": "Bottle Caps Clinking (Hard Negative)", "type": "hard_negative", "badge": "HARD NEGATIVE"},
        "container_shake.wav": {"label": "Other Shaking Container (Hard Negative)", "type": "hard_negative", "badge": "HARD NEGATIVE"},
        "glassware.wav": {"label": "Glassware Clinking (Hard Negative)", "type": "hard_negative", "badge": "HARD NEGATIVE"},
        "liquid_bottle.wav": {"label": "Liquid Bottle Sloshing (Hard Negative)", "type": "hard_negative", "badge": "HARD NEGATIVE"},
        "ambient_speech.wav": {"label": "Speech & Background Voice", "type": "ambient", "badge": "AMBIENT"},
        "ambient_silence.wav": {"label": "Room Tone / Silence", "type": "ambient", "badge": "AMBIENT"},
    }

    for f in sorted(DEMO_DIR.glob("*.wav")):
        meta = category_map.get(f.name, {"label": f.stem.replace("_", " ").title(), "type": "other", "badge": "SAMPLE"})
        samples.append({
            "id": f.name,
            "filename": f.name,
            "label": meta["label"],
            "type": meta["type"],
            "badge": meta["badge"],
        })

    return jsonify({"samples": samples})


@app.route("/api/demo_sample/<sample_id>/evaluate", methods=["POST"])
def evaluate_demo_sample(sample_id: str):
    """Evaluate a specific demo sample without requiring file upload."""
    ensure_demo_samples()
    sample_path = DEMO_DIR / sample_id
    if not sample_path.exists():
        return jsonify({"error": f"Sample {sample_id} not found"}), 404

    audio = load_audio(sample_path, sr=SAMPLE_RATE)
    results = engine.process_audio_file(audio)

    # Render spectrogram
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    clip_len = min(len(audio), int(SAMPLE_RATE * 3.5))
    clip_audio = audio[:clip_len]
    spec = compute_log_mel_spectrogram(clip_audio, sr=SAMPLE_RATE)

    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, clip_len / SAMPLE_RATE, 100, 11000],
    )
    ax.set_title(f"Log-Mel Spectrogram: {sample_id}", color="#E5E7EB", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", color="#9CA3AF", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color="#9CA3AF", fontsize=9)
    ax.tick_params(colors="#9CA3AF", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#374151")

    fig.colorbar(im, ax=ax, pad=0.02)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    results["spectrogram_image_base64"] = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify(results)


@app.route("/api/demo_sample/<sample_id>/audio", methods=["GET"])
def get_demo_audio(sample_id: str):
    """Serve demo sample audio file for browser playback."""
    ensure_demo_samples()
    sample_path = DEMO_DIR / sample_id
    if not sample_path.exists():
        return "Not found", 404
    return send_file(sample_path, mimetype="audio/wav")


@app.route("/api/calibration_report", methods=["GET"])
def get_calibration_report():
    """Return threshold calibration sweep and evaluation report."""
    if EVALUATION_REPORT_PATH.exists():
        with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Evaluation report not found"}), 404


@app.route("/api/review_queue", methods=["GET"])
def get_review_queue():
    """Return items in the operator review queue (uncertain candidates and confirmed alerts)."""
    items = engine.get_review_queue()
    return jsonify({
        "items": items,
        "count": len(items),
        "retention_hours": AUDIO_RETENTION_HOURS,
        "privacy_notice": (
            "Ephemeral review buffer. Audio clips are retained up to "
            f"{AUDIO_RETENTION_HOURS} hours for human operator assessment, "
            "then permanently purged. No biometric or personal identification is performed."
        ),
    })


@app.route("/api/review_queue/<event_id>/audio", methods=["GET"])
def get_review_audio(event_id: str):
    """Serve ephemeral audio snippet for human operator review."""
    for item in REVIEW_QUEUE_DIR.glob(f"{event_id}_*.wav"):
        return send_file(item, mimetype="audio/wav")
    return "Review audio not found or expired", 404


@app.route("/api/review_queue/resolve", methods=["POST"])
def resolve_review():
    """Human operator records verdict on a review item."""
    data = request.get_json(force=True)
    event_id = data.get("event_id")
    human_verdict = data.get("human_verdict")  # "verified_spray", "false_alert", "inconclusive"
    notes = data.get("notes", "")
    if not event_id or not human_verdict:
        return jsonify({"error": "event_id and human_verdict required"}), 400

    success = engine.resolve_review_event(event_id, human_verdict, notes)
    if success:
        return jsonify({"status": "resolved", "event_id": event_id, "human_verdict": human_verdict})
    return jsonify({"error": "Failed to resolve event"}), 404


@app.route("/api/privacy_policy", methods=["GET"])
def get_privacy_policy():
    """Expose current audio retention and data governance settings."""
    return jsonify({
        "continuous_audio_stored": SAVE_MIC_AUDIO,
        "ephemeral_review_enabled": True,
        "audio_retention_hours": AUDIO_RETENTION_HOURS,
        "max_stored_clips": MAX_REVIEW_QUEUE_ITEMS,
        "human_tracking_disabled": True,
        "compliance": "Acoustic event detection only; no voice identification, biometrics, or tracking.",
    })


@app.route("/api/record_sample", methods=["POST"])
def record_sample():
    """Save an operator-recorded ground truth sample to field dataset with metadata."""
    try:
        class_name = request.form.get("class_name") or "spray_can_shaking"
        condition = request.form.get("condition") or "clean_indoor"
        session_id = request.form.get("session_id") or "session_local_can_01"
        distance_m = request.form.get("distance_m") or "1.0"
        microphone_name = request.form.get("microphone_name") or "Target Microphone"
        can_type = request.form.get("can_type") or "n_a_negative"
        can_fill_level = request.form.get("can_fill_level") or "n_a_negative"
        shaking_style = request.form.get("shaking_style") or "n_a_negative"
        label_confidence = request.form.get("label_confidence") or "confirmed_high"
        notes = request.form.get("notes") or ""

        # Validate class
        allowed_classes = list(CLASS_NAMES) + ["overlapping_train_announcement"]
        if class_name not in allowed_classes:
            return jsonify({"error": f"Invalid class name '{class_name}'. Must be one of {allowed_classes}"}), 400

        audio_bytes = None
        if "audio" in request.files:
            audio_bytes = request.files["audio"].read()
        elif request.is_json:
            json_data = request.get_json(silent=True) or {}
            if "audio_base64" in json_data:
                audio_bytes = base64.b64decode(json_data["audio_base64"])
                class_name = json_data.get("class_name", class_name)
                condition = json_data.get("condition", condition)
                session_id = json_data.get("session_id", session_id)
                distance_m = json_data.get("distance_m", distance_m)
                microphone_name = json_data.get("microphone_name", microphone_name)
                can_type = json_data.get("can_type", can_type)
                can_fill_level = json_data.get("can_fill_level", can_fill_level)
                shaking_style = json_data.get("shaking_style", shaking_style)
                label_confidence = json_data.get("label_confidence", label_confidence)
                notes = json_data.get("notes", notes)

        if not audio_bytes:
            return jsonify({"error": "No audio payload provided"}), 400

        if audio_bytes.startswith(b"RIFF"):
            audio_arr = load_audio(audio_bytes, sr=SAMPLE_RATE)
        else:
            try:
                audio_arr = load_audio(audio_bytes, sr=SAMPLE_RATE)
            except Exception:
                audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        duration_sec = round(len(audio_arr) / SAMPLE_RATE, 2)
        if duration_sec < 0.5:
            return jsonify({"error": f"Audio too short ({duration_sec}s < 0.5s)"}), 400

        # Target directory: data/raw/field_recordings/<class_name>
        target_dir = FIELD_RECORDINGS_DIR / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        import time, uuid
        ts = int(time.time())
        uid = str(uuid.uuid4())[:8]
        clean_sess = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_id)
        filename = f"{clean_sess}_{condition}_{ts}_{uid}.wav"
        save_path = target_dir / filename

        sf.write(str(save_path), audio_arr, SAMPLE_RATE)

        # Update manifest: data/raw/field_recordings/manifest.json
        manifest_path = FIELD_RECORDINGS_DIR / "manifest.json"
        manifest_entries = []
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_entries = json.load(f)
            except Exception:
                manifest_entries = []

        entry = {
            "id": uid,
            "filename": filename,
            "filepath": str(save_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "class_name": class_name,
            "condition": condition,
            "session_id": clean_sess,
            "microphone_name": microphone_name,
            "can_type": can_type,
            "can_fill_level": can_fill_level,
            "shaking_style": shaking_style,
            "label_confidence": label_confidence,
            "distance_m": distance_m,
            "notes": notes,
            "duration_sec": duration_sec,
            "samples": len(audio_arr),
            "sample_rate": SAMPLE_RATE,
            "timestamp": ts,
        }
        manifest_entries.append(entry)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, indent=2)

        return jsonify({
            "status": "success",
            "message": f"Saved {duration_sec}s sample to {class_name}",
            "entry": entry,
            "total_recordings": len(manifest_entries),
        })
    except Exception as e:
        logger.error("Error recording sample: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/audio_file", methods=["GET"])
def get_audio_file():
    """Safely stream a recorded audio file for playback or inspection."""
    req_path = request.args.get("path", "")
    if not req_path:
        return jsonify({"error": "No file path specified"}), 400
    file_path = (BASE_DIR / req_path).resolve()
    base_resolved = BASE_DIR.resolve()
    try:
        if not str(file_path).startswith(str(base_resolved)):
            return jsonify({"error": "Access denied"}), 403
    except Exception:
        return jsonify({"error": "Invalid path"}), 400

    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(file_path), mimetype="audio/wav")


@app.route("/api/field_recordings", methods=["GET"])
def get_field_recordings():
    """List all operator-recorded ground truth field recordings."""
    manifest_path = FIELD_RECORDINGS_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
                return jsonify({"status": "success", "recordings": entries, "count": len(entries)})
        except Exception:
            pass
    return jsonify({"status": "success", "recordings": [], "count": 0})


@app.route("/api/field_session/save_continuous", methods=["POST"])
def save_continuous_field_session():
    """Save a continuous real-site audio session from installed microphone with metadata."""
    try:
        session_id = request.form.get("session_id") or f"session_{int(time.time())}"
        site_id = request.form.get("site_id") or "railway_station_platform"
        microphone_id = request.form.get("microphone_id") or "installed_cardioid_01"
        operator_id = request.form.get("operator_id") or "operator_field_tech"
        condition = request.form.get("condition") or "station_ambient"
        distance_m = request.form.get("distance_m") or "3.0"
        notes = request.form.get("notes") or ""

        audio_bytes = None
        if "audio" in request.files:
            audio_bytes = request.files["audio"].read()
        elif request.is_json:
            json_data = request.get_json(silent=True) or {}
            audio_b64 = json_data.get("audio_base64")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
            session_id = json_data.get("session_id", session_id)
            site_id = json_data.get("site_id", site_id)
            microphone_id = json_data.get("microphone_id", microphone_id)
            operator_id = json_data.get("operator_id", operator_id)
            condition = json_data.get("condition", condition)
            distance_m = json_data.get("distance_m", distance_m)
            notes = json_data.get("notes", notes)

        if not audio_bytes:
            return jsonify({"error": "No audio payload provided"}), 400

        audio_arr = load_audio(audio_bytes, sr=SAMPLE_RATE)
        duration_sec = round(len(audio_arr) / SAMPLE_RATE, 2)

        # Sanitize session_id
        clean_sess = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_id)
        sess_dir = FIELD_SESSIONS_DIR / clean_sess
        sess_dir.mkdir(parents=True, exist_ok=True)

        audio_path = sess_dir / "audio.wav"
        sf.write(str(audio_path), audio_arr, SAMPLE_RATE)

        meta = {
            "session_id": clean_sess,
            "site_id": site_id,
            "microphone_id": microphone_id,
            "operator_id": operator_id,
            "condition": condition,
            "distance_m": distance_m,
            "notes": notes,
            "duration_sec": duration_sec,
            "duration_hours": round(duration_sec / 3600.0, 4),
            "sample_rate": SAMPLE_RATE,
            "num_samples": len(audio_arr),
            "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "created_epoch": time.time(),
            "retention_policy_hours": AUDIO_RETENTION_HOURS,
            "shadow_mode_advisory_only": True,
            "non_inference_rule": "Audio evidence measures acoustic energy only. Do not infer graffiti intent or identify individuals.",
        }

        with open(sess_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return jsonify({
            "status": "success",
            "message": f"Saved {duration_sec}s continuous session '{clean_sess}'",
            "metadata": meta,
        })
    except Exception as e:
        logger.error("Error saving continuous field session: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/field_sessions", methods=["GET"])
def list_field_sessions():
    """List all continuous field recording sessions with metadata and annotation status."""
    engine._purge_expired_review_items()
    sessions = []
    if FIELD_SESSIONS_DIR.exists():
        for sess_dir in sorted(FIELD_SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not sess_dir.is_dir():
                continue
            meta_path = sess_dir / "metadata.json"
            annot_path = sess_dir / "annotations.json"
            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            annot_count = 0
            if annot_path.exists():
                try:
                    with open(annot_path, "r", encoding="utf-8") as f:
                        ann_data = json.load(f)
                        annot_count = len(ann_data.get("events", []))
                except Exception:
                    pass
            meta["annotation_count"] = annot_count
            meta["has_annotations"] = annot_count > 0
            meta["session_folder"] = sess_dir.name
            sessions.append(meta)
    return jsonify({"sessions": sessions, "total_sessions": len(sessions)})


@app.route("/api/field_session/annotate", methods=["POST"])
def annotate_field_session():
    """Save trained human reviewer interval annotations for a continuous field session."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id")
        events = data.get("events", [])
        reviewer_id = data.get("reviewer_id") or "reviewer_trained_01"
        confidence_level = data.get("verification_confidence") or "confirmed_high"
        reviewer_notes = data.get("reviewer_notes") or ""

        if not session_id:
            return jsonify({"error": "session_id required"}), 400

        clean_sess = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_id)
        sess_dir = FIELD_SESSIONS_DIR / clean_sess
        if not sess_dir.exists():
            return jsonify({"error": f"Session '{clean_sess}' does not exist"}), 404

        # Validate events
        valid_classes = list(CLASS_NAMES)
        for ev in events:
            if "start_sec" not in ev or "end_sec" not in ev or "class_label" not in ev:
                return jsonify({"error": "Each event must have 'start_sec', 'end_sec', and 'class_label'"}), 400
            if ev["start_sec"] >= ev["end_sec"]:
                return jsonify({"error": f"Invalid interval: start_sec ({ev['start_sec']}) >= end_sec ({ev['end_sec']})"}), 400
            if ev["class_label"] not in valid_classes:
                return jsonify({"error": f"Invalid class '{ev['class_label']}'. Must be in {valid_classes}"}), 400

        annot_payload = {
            "session_id": clean_sess,
            "reviewer_id": reviewer_id,
            "verification_confidence": confidence_level,
            "reviewer_notes": reviewer_notes,
            "reviewed_timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "non_inference_statement": "Acoustic ground truth labels strictly define audio physics. No person intent or identity is inferred.",
            "total_annotated_events": len(events),
            "events": events,
        }

        annot_path = sess_dir / "annotations.json"
        with open(annot_path, "w", encoding="utf-8") as f:
            json.dump(annot_payload, f, indent=2)

        return jsonify({
            "status": "success",
            "message": f"Saved {len(events)} event annotations for session '{clean_sess}'",
            "annotations": annot_payload,
        })
    except Exception as e:
        logger.error("Error annotating session: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/field_session/<session_id>/annotations", methods=["GET"])
def get_session_annotations(session_id):
    """Retrieve event annotations for a specified field session."""
    clean_sess = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_id)
    annot_path = FIELD_SESSIONS_DIR / clean_sess / "annotations.json"
    if annot_path.exists():
        try:
            with open(annot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"session_id": clean_sess, "events": [], "total_annotated_events": 0})


@app.route("/api/field_session/purge_expired", methods=["POST"])
def purge_expired_recordings():
    """Explicitly execute data retention cleanup across review queue and field sessions."""
    try:
        engine._purge_expired_review_items()
        return jsonify({
            "status": "success",
            "message": f"Privacy retention cleanup executed (Retention limit: {AUDIO_RETENTION_HOURS} hours)",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/shadow_alerts", methods=["GET"])
def get_shadow_alerts():
    """Fetch recent advisory shadow mode alerts from append-only log."""
    alerts = []
    if SHADOW_ALERTS_LOG.exists():
        try:
            with open(SHADOW_ALERTS_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        alerts.append(json.loads(line))
        except Exception as e:
            logger.warning("Error reading shadow alerts log: %s", e)
    # Return last 100 alerts newest first
    return jsonify({"alerts": list(reversed(alerts[-100:])), "total_count": len(alerts)})


@app.route("/report.pdf")
@app.route("/api/download_report")
def download_presentation_report():
    """Serve the comprehensive presentation report PDF."""
    pdf_path = BASE_DIR / "Acoustic_Spray_Can_Detector_Presentation_Report.pdf"
    if not pdf_path.exists():
        from generate_presentation_report import build_pdf
        build_pdf(str(pdf_path))
    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=False,
        download_name="Acoustic_Spray_Can_Detector_Presentation_Report.pdf",
    )


# =========================================================================
# Authentication & Role-Based Access Control APIs
# =========================================================================

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Authenticate administrator or user credentials."""
    data = request.get_json(force=True, silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    client_ip = request.remote_addr or "127.0.0.1"

    user, err = authenticate_user(username, password, client_ip=client_ip)
    if err:
        return jsonify({"success": False, "error": err}), 401

    session["user"] = user
    csrf_token = generate_csrf_token()
    return jsonify({
        "success": True,
        "user": user,
        "csrf_token": csrf_token,
    })


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    """Log out current user and clear session."""
    user = session.get("user")
    if user:
        log_audit_action(user.get("username", "admin"), "logout", {}, user_id=user.get("id"), ip_address=request.remote_addr)
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    """Return currently authenticated session identity."""
    user = session.get("user")
    csrf_token = generate_csrf_token() if user else None
    return jsonify({
        "authenticated": user is not None,
        "user": user,
        "csrf_token": csrf_token,
    })


# =========================================================================
# Admin Audio Recording & Ingestion APIs
# =========================================================================

@app.route("/api/admin/record/validate", methods=["POST"])
@admin_required
def admin_record_validate():
    """Pre-validate an audio snippet for duration, energy, and clipping without saving."""
    try:
        audio_bytes = None
        if "audio" in request.files:
            audio_bytes = request.files["audio"].read()
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            if "audio_base64" in data:
                audio_bytes = base64.b64decode(data["audio_base64"])

        if not audio_bytes:
            return jsonify({"valid": False, "error": "No audio payload received."}), 400

        audio_arr, err = convert_and_validate_audio(audio_bytes, target_sr=SAMPLE_RATE)
        if err:
            return jsonify({"valid": False, "error": err}), 400

        duration_sec = round(len(audio_arr) / SAMPLE_RATE, 3)
        rms = round(float(np.sqrt(np.mean(audio_arr**2))), 5)
        peak = round(float(np.max(np.abs(audio_arr))), 3)

        return jsonify({
            "valid": True,
            "duration_sec": duration_sec,
            "sample_rate": SAMPLE_RATE,
            "rms_energy": rms,
            "peak_amplitude": peak,
        })
    except Exception as e:
        logger.error("Error in admin_record_validate: %s", e)
        return jsonify({"valid": False, "error": str(e)}), 500


@app.route("/api/admin/record/save", methods=["POST"])
@admin_required
def admin_record_save():
    """Convert browser recording to canonical 22,050 Hz WAV and save into dataset."""
    try:
        user = session["user"]
        audio_bytes = None
        category_id = "spray_can_shaking"
        label_type = "positive"
        session_id = "session_browser_01"
        notes = ""
        microphone_name = "Browser Microphone"
        can_type = "aerosol_paint_can"
        can_fill_level = "full"
        shaking_style = "vertical_vigorous"
        distance_m = 1.0
        condition = "indoor_quiet"

        if "audio" in request.files:
            audio_bytes = request.files["audio"].read()
            category_id = request.form.get("category_id") or category_id
            label_type = request.form.get("label_type") or label_type
            session_id = request.form.get("session_id") or session_id
            notes = request.form.get("notes") or notes
            microphone_name = request.form.get("microphone_name") or microphone_name
            can_type = request.form.get("can_type") or can_type
            can_fill_level = request.form.get("can_fill_level") or can_fill_level
            shaking_style = request.form.get("shaking_style") or shaking_style
            try:
                distance_m = float(request.form.get("distance_m", 1.0))
            except Exception:
                distance_m = 1.0
            condition = request.form.get("condition") or condition
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            if "audio_base64" in data:
                audio_bytes = base64.b64decode(data["audio_base64"])
            category_id = data.get("category_id", category_id)
            label_type = data.get("label_type", label_type)
            session_id = data.get("session_id", session_id)
            notes = data.get("notes", notes)
            microphone_name = data.get("microphone_name", microphone_name)
            can_type = data.get("can_type", can_type)
            can_fill_level = data.get("can_fill_level", can_fill_level)
            shaking_style = data.get("shaking_style", shaking_style)
            distance_m = float(data.get("distance_m", 1.0))
            condition = data.get("condition", condition)

        if not audio_bytes:
            return jsonify({"error": "No audio payload provided."}), 400

        audio_arr, err = convert_and_validate_audio(audio_bytes, target_sr=SAMPLE_RATE)
        if err:
            return jsonify({"error": err}), 400

        rec_dict, save_err = save_recording_sample(
            audio_arr=audio_arr,
            category_id=category_id,
            label_type=label_type,
            admin_user=user["username"],
            session_id=session_id,
            microphone_name=microphone_name,
            can_type=can_type,
            can_fill_level=can_fill_level,
            shaking_style=shaking_style,
            distance_m=distance_m,
            condition=condition,
            notes=notes,
        )

        if save_err:
            return jsonify({"error": save_err}), 400

        return jsonify({
            "success": True,
            "message": f"Successfully saved {rec_dict['duration_sec']}s recording to category '{category_id}'.",
            "recording": rec_dict,
        })
    except Exception as e:
        logger.error("Error in admin_record_save: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# =========================================================================
# Admin Dataset Manager APIs
# =========================================================================

@app.route("/api/admin/dataset/samples", methods=["GET"])
@admin_required
def admin_dataset_samples():
    """Query recordings metadata with filtering and pagination."""
    cat = request.args.get("category_id")
    label = request.args.get("label_type")
    status = request.args.get("status")
    search = request.args.get("search")
    limit = min(200, max(1, int(request.args.get("limit", 50))))
    offset = max(0, int(request.args.get("offset", 0)))

    samples, total = query_recordings(
        category_id=cat,
        label_type=label,
        status=status,
        search=search,
        limit=limit,
        offset=offset,
    )

    return jsonify({
        "samples": samples,
        "total_count": total,
        "limit": limit,
        "offset": offset,
    })


@app.route("/api/admin/dataset/sample/<sample_id>/audio", methods=["GET"])
@admin_required
def admin_sample_audio(sample_id: str):
    """Safely stream recorded audio sample for administrator playback."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT relative_path FROM recordings WHERE id = ?", (sample_id,))
        row = cursor.fetchone()
        if not row:
            return "Sample not found", 404

        file_path = BASE_DIR / row["relative_path"]
        if not file_path.exists():
            return "Audio file missing on disk", 404

        return send_file(str(file_path), mimetype="audio/wav")


@app.route("/api/admin/dataset/sample/<sample_id>/status", methods=["POST"])
@admin_required
def admin_sample_update_status(sample_id: str):
    """Update validation status (accepted, quarantined, rejected)."""
    user = session["user"]
    data = request.get_json(force=True, silent=True) or {}
    new_status = data.get("status")
    reason = data.get("reason", "")

    if not new_status:
        return jsonify({"error": "New status is required."}), 400

    ok, err = update_recording_status(sample_id, new_status, user["username"], reason=reason)
    if not ok:
        return jsonify({"error": err}), 400

    return jsonify({"success": True, "sample_id": sample_id, "new_status": new_status})


@app.route("/api/admin/dataset/sample/<sample_id>/delete", methods=["POST"])
@admin_required
def admin_sample_delete(sample_id: str):
    """Soft delete or permanently remove a recording."""
    user = session["user"]
    data = request.get_json(force=True, silent=True) or {}
    permanent = bool(data.get("permanent", False))

    ok, err = delete_recording(sample_id, user["username"], permanent=permanent)
    if not ok:
        return jsonify({"error": err}), 400

    return jsonify({"success": True, "sample_id": sample_id, "permanent": permanent})


@app.route("/api/admin/dataset/categories", methods=["GET"])
@admin_required
def admin_dataset_categories():
    """List all categories with sample stats and model integration status."""
    cats = list_categories()
    return jsonify({"categories": cats, "count": len(cats)})


@app.route("/api/admin/dataset/category/create", methods=["POST"])
@admin_required
def admin_category_create():
    """Create a new sound category with pending model integration safeguard."""
    user = session["user"]
    data = request.get_json(force=True, silent=True) or {}
    category_id = data.get("id") or data.get("category_id") or ""
    display_name = data.get("display_name") or category_id
    description = data.get("description", "")
    label_type = data.get("label_type", "negative")

    cat_dict, err = create_new_category(
        category_id=category_id,
        display_name=display_name,
        description=description,
        label_type=label_type,
        admin_user=user["username"],
    )

    if err:
        return jsonify({"error": err}), 400

    return jsonify({"success": True, "category": cat_dict})


@app.route("/api/admin/dataset/validate", methods=["GET"])
@admin_required
def admin_dataset_validate():
    """Run automated dataset preparation and pre-flight validation."""
    report = validate_dataset_for_training()
    return jsonify(report)


# =========================================================================
# Admin One-Click Training & Model Management APIs
# =========================================================================

@app.route("/api/admin/training/start", methods=["POST"])
@admin_required
def admin_training_start():
    """Launch one-click model training in a controlled background worker."""
    user = session["user"]
    data = request.get_json(force=True, silent=True) or {}
    epochs = int(data.get("epochs", 15))
    epochs = max(2, min(50, epochs))
    fine_tune = bool(data.get("fine_tune", True))
    rebuild = bool(data.get("rebuild_dataset", True))

    job_id, err = training_manager.start_training_job(
        admin_user=user["username"],
        epochs=epochs,
        fine_tune=fine_tune,
        rebuild_dataset=rebuild,
    )

    if err:
        return jsonify({"error": err}), 409

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": f"Training job '{job_id}' started in background ({epochs} epochs).",
    })


@app.route("/api/admin/training/status", methods=["GET"])
@admin_required
def admin_training_status():
    """Poll progress, current epoch, loss, and recent logs for training jobs."""
    job_id = request.args.get("job_id")
    status_data = training_manager.get_job_status(job_id=job_id)
    if not status_data:
        return jsonify({"error": "No training job found."}), 404
    return jsonify(status_data)


@app.route("/api/admin/training/cancel", methods=["POST"])
@admin_required
def admin_training_cancel():
    """Request clean cancellation of the active training job."""
    user = session["user"]
    data = request.get_json(force=True, silent=True) or {}
    job_id = data.get("job_id") or training_manager.get_active_job_id()

    if not job_id:
        return jsonify({"error": "No active job to cancel."}), 400

    ok, msg = training_manager.request_cancellation(job_id, user["username"])
    if not ok:
        return jsonify({"error": msg}), 400

    return jsonify({"success": True, "message": msg, "job_id": job_id})


@app.route("/api/admin/model/versions", methods=["GET"])
@admin_required
def admin_model_versions():
    """List all candidate and active model versions with evaluation metrics."""
    versions = training_manager.list_model_versions()
    return jsonify({"versions": versions, "count": len(versions)})


@app.route("/api/admin/model/activate", methods=["POST"])
@admin_required
def admin_model_activate():
    """Explicitly activate a candidate model version and reload detector engine."""
    user = session["user"]
    data = request.get_json(force=True, silent=True) or {}
    version_id = data.get("version_id")

    if not version_id:
        return jsonify({"error": "version_id is required."}), 400

    ok, err = training_manager.activate_model_version(version_id, user["username"], engine_instance=engine)
    if not ok:
        return jsonify({"error": err}), 400

    return jsonify({
        "success": True,
        "message": f"Model version '{version_id}' successfully activated. Detector engine hot-reloaded.",
        "active_version": version_id,
    })


@app.route("/api/admin/model/rollback", methods=["POST"])
@admin_required
def admin_model_rollback():
    """Roll back active model to a previous version."""
    user = session["user"]
    data = request.get_json(force=True, silent=True) or {}
    version_id = data.get("version_id")

    if not version_id:
        return jsonify({"error": "version_id is required."}), 400

    ok, err = training_manager.rollback_to_version(version_id, user["username"], engine_instance=engine)
    if not ok:
        return jsonify({"error": err}), 400

    return jsonify({
        "success": True,
        "message": f"Successfully rolled back to model version '{version_id}'.",
        "active_version": version_id,
    })


@app.route("/api/admin/audit_logs", methods=["GET"])
@admin_required
def admin_audit_logs():
    """Retrieve administrative action audit logs."""
    limit = min(200, max(1, int(request.args.get("limit", 100))))
    logs = get_audit_logs(limit=limit)
    return jsonify({"audit_logs": logs, "count": len(logs)})


if __name__ == "__main__":
    ensure_demo_samples()
    start_esp32_receiver_service()
    logger.info("Starting Flask Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
