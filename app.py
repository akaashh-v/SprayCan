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
)
from config import (
    AUDIO_SAMPLES,
    BASE_DIR,
    CALIBRATION_PATH,
    CLASS_NAMES,
    CONSERVATIVE_THRESHOLD,
    EVALUATION_REPORT_PATH,
    RAW_SPRAY_DIR,
    SAMPLE_RATE,
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
        new_cd = max(0.1, min(10.0, new_cd))
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


if __name__ == "__main__":
    ensure_demo_samples()
    logger.info("Starting Flask Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
