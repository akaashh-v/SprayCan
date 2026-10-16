
import os
import secrets
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, request, jsonify, abort

app = Flask(__name__)

API_TOKEN = os.environ.get("ESP32_TEST_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Set ESP32_TEST_TOKEN before starting the server")

SAVE_DIR = Path("tmp_esp32_audio")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 200_000


@app.get("/esp32/health")
def health():
    return jsonify({"status": "ok", "service": "ESP32 test receiver"})


@app.post("/esp32/test-audio")
def receive_audio():
    supplied_token = request.headers.get("X-Device-Token", "")

    if not secrets.compare_digest(supplied_token, API_TOKEN):
        abort(403)

    if request.mimetype != "audio/wav":
        return jsonify({"error": "Expected audio/wav"}), 415

    audio = request.get_data(cache=False)

    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return jsonify({"error": "Invalid WAV file"}), 400

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    filename = SAVE_DIR / f"esp32_{timestamp}.wav"

    with filename.open("xb") as f:
        f.write(audio)

    print(f"Received {len(audio)} bytes -> {filename}")

    return jsonify({
        "status": "received",
        "bytes": len(audio),
        "file": filename.name
    }), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
