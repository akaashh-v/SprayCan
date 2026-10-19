"""Standalone Remote ESP32 Hardware Audio Receiver.

Can be run independently or integrated into the main application.
Coordinates with esp32_manager and detector_engine to save audio and process detections.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from flask import Flask, jsonify, request
from config import ESP32_RECEIVER_HOST, ESP32_RECEIVER_PORT
from detector_engine import SprayCanDetectorEngine
from esp32_manager import esp32_manager

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
            elif "/esp32/alert" in msg and not str(record.msg).startswith("[EDGE-ALERT]"):
                record.msg = f"[EDGE-ALERT] {record.msg}"
            elif "/esp32/test-audio" in msg and not str(record.msg).startswith("[AUDIO-UPLOAD]"):
                record.msg = f"[AUDIO-UPLOAD] {record.msg}"
            elif "/esp32/status" in msg and not str(record.msg).startswith("[STATE-ACK]"):
                record.msg = f"[STATE-ACK] {record.msg}"
        except Exception:
            pass
        return True


logging.getLogger("werkzeug").addFilter(HttpRequestCategoryFilter())

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500_000

# Standalone detector engine instance if run directly
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = SprayCanDetectorEngine()
    return _engine


@app.get("/esp32/health")
def health():
    status = esp32_manager.get_status_dict()
    return jsonify({
        "status": "ok",
        "service": "ESP32 hardware receiver",
        "state": status["state"],
        "desired_state": status["desired_state"],
    })


@app.get("/esp32/command")
def command_poll():
    supplied_token = request.headers.get("X-Device-Token", "")
    reported_state = request.args.get("state") or request.headers.get("X-Device-State")
    client_ip = request.remote_addr or "unknown"
    resp, code = esp32_manager.handle_command_poll(supplied_token, client_ip, reported_state=reported_state)
    return jsonify(resp), code


@app.post("/esp32/status")
def status_ack():
    supplied_token = request.headers.get("X-Device-Token", "")
    data = request.get_json(silent=True) or {}
    reported_state = data.get("state") or request.headers.get("X-Device-State", "")
    client_ip = request.remote_addr or "unknown"
    resp, code = esp32_manager.handle_status_update(supplied_token, client_ip, reported_state=reported_state)
    return jsonify(resp), code


@app.post("/esp32/alert")
def receive_edge_alert():
    supplied_token = request.headers.get("X-Device-Token", "")
    client_ip = request.remote_addr or "unknown"
    data = request.get_json(silent=True) or {}
    resp, code = esp32_manager.handle_edge_alert(supplied_token, client_ip, data)
    response = jsonify(resp)
    response.headers["Connection"] = "close"
    return response, code


@app.post("/esp32/test-audio")
def receive_audio():
    supplied_token = request.headers.get("X-Device-Token", "")
    client_ip = request.remote_addr or "unknown"
    audio_data = request.get_data(cache=False)
    resp, code = esp32_manager.handle_audio_upload(
        supplied_token=supplied_token,
        audio_bytes=audio_data,
        mimetype=request.mimetype,
        client_ip=client_ip,
        engine=get_engine(),
        headers=dict(request.headers),
    )
    response = jsonify(resp)
    response.headers["Connection"] = "close"
    return response, code


if __name__ == "__main__":
    host = os.environ.get("ESP32_RECEIVER_HOST", ESP32_RECEIVER_HOST)
    port = int(os.environ.get("ESP32_RECEIVER_PORT", ESP32_RECEIVER_PORT))
    logger.info("Starting standalone ESP32 Hardware Receiver on http://%s:%d ...", host, port)
    app.run(host=host, port=port, debug=False)
