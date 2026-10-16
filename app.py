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
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_file

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
    MAX_REVIEW_QUEUE_ITEMS,
    RAW_SPRAY_DIR,
    REVIEW_QUEUE_DIR,
    SAMPLE_RATE,
    SAVE_MIC_AUDIO,
    SHADOW_ALERTS_LOG,
    SHADOW_MODE_ENABLED,
    STATIC_DIR,
    TEMPLATES_DIR,
)
from detector_engine import SprayCanDetectorEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

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
    return render_template("index.html")


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
    })


@app.route("/api/config", methods=["POST"])
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


if __name__ == "__main__":
    ensure_demo_samples()
    logger.info("Starting Flask Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
