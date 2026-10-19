"""Comprehensive Unit and Integration Test Suite for ESP32 Hardware Integration.

Tests:
1. Authenticated START and STOP control endpoints.
2. ESP32 command polling mechanism and token authentication.
3. ESP32 state acknowledgement (ACK) and bidirectional state transitions.
4. Audio upload reception, WAV validation, and detector pipeline execution.
5. Timeout safety and connection error handling.
6. Non-exposure of device token in public APIs.
"""

import io
import time
import unittest
import wave
import numpy as np

from app import app
from config import ESP32_DEVICE_TOKEN
from esp32_manager import esp32_manager, ESP32State


def generate_test_wav(duration_sec: float = 1.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    """Generate a valid PCM 16-bit mono WAV in-memory for testing."""
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Sine wave
    sig = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sig.tobytes())
    return buf.getvalue()


class TestESP32Integration(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        # Reset esp32_manager state
        with esp32_manager._lock:
            esp32_manager.state = ESP32State.STOPPED
            esp32_manager.desired_state = "STOP"
            esp32_manager.transition_time = 0.0
            esp32_manager.last_heartbeat_time = 0.0
            esp32_manager.last_audio_time = 0.0
            esp32_manager.last_audio_file = None
            esp32_manager.last_audio_bytes = 0
            esp32_manager.audio_chunk_count = 0
            esp32_manager.connection_error_reason = None
            esp32_manager.latest_prediction = None

    def test_unauthenticated_esp32_endpoints_rejected(self):
        """Hardware endpoints must reject missing or invalid X-Device-Token."""
        # 1. /esp32/command without token
        resp = self.client.get("/esp32/command")
        self.assertEqual(resp.status_code, 403)

        # 2. /esp32/command with invalid token
        resp = self.client.get("/esp32/command", headers={"X-Device-Token": "wrong_invalid_token"})
        self.assertEqual(resp.status_code, 403)

        # 3. /esp32/status without token
        resp = self.client.post("/esp32/status", json={"state": "LISTENING"})
        self.assertEqual(resp.status_code, 403)

        # 4. /esp32/test-audio without token
        wav_bytes = generate_test_wav()
        resp = self.client.post("/esp32/test-audio", data=wav_bytes, content_type="audio/wav")
        self.assertEqual(resp.status_code, 403)

    def test_status_endpoint_never_leaks_token(self):
        """Public dashboard status API must never expose device token or secrets."""
        resp = self.client.get("/api/esp32/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("state", data)
        self.assertIn("desired_state", data)
        self.assertNotIn("device_token", data)
        self.assertNotIn("token", data)
        self.assertNotIn("wifi_password", data)
        self.assertNotIn(ESP32_DEVICE_TOKEN, resp.get_data(as_text=True))

    def test_start_listening_flow_and_acknowledgement(self):
        """Verify full START lifecycle: Dashboard -> Backend -> ESP32 Poll -> ESP32 ACK -> Dashboard."""
        # Step 1: Initial state is Stopped
        resp = self.client.get("/api/esp32/status")
        self.assertEqual(resp.get_json()["state"], ESP32State.STOPPED)

        # Step 2: User clicks Start Listening on Dashboard
        resp = self.client.post("/api/esp32/start")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["state"], ESP32State.STARTING)

        # Dashboard status confirms Starting
        resp = self.client.get("/api/esp32/status")
        self.assertEqual(resp.get_json()["state"], ESP32State.STARTING)

        # Step 3: ESP32 polls /esp32/command
        resp = self.client.get(
            "/esp32/command",
            headers={"X-Device-Token": ESP32_DEVICE_TOKEN}
        )
        self.assertEqual(resp.status_code, 200)
        cmd_data = resp.get_json()
        self.assertEqual(cmd_data["command"], "START")
        self.assertEqual(cmd_data["desired_state"], "START")

        # Step 4: ESP32 sends state ACK (LISTENING)
        resp = self.client.post(
            "/esp32/status",
            headers={"X-Device-Token": ESP32_DEVICE_TOKEN},
            json={"state": "LISTENING"}
        )
        self.assertEqual(resp.status_code, 200)

        # Step 5: Dashboard status updates to Listening
        resp = self.client.get("/api/esp32/status")
        status_data = resp.get_json()
        self.assertEqual(status_data["state"], ESP32State.LISTENING)

    def test_stop_listening_flow_and_acknowledgement(self):
        """Verify full STOP lifecycle: Dashboard -> Backend -> ESP32 Poll -> ESP32 ACK -> Dashboard."""
        # Set to Listening
        esp32_manager.state = ESP32State.LISTENING
        esp32_manager.desired_state = "START"

        # Step 1: User clicks Stop Listening on Dashboard
        resp = self.client.post("/api/esp32/stop")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["state"], ESP32State.STOPPING)

        # Step 2: ESP32 polls /esp32/command and receives STOP
        resp = self.client.get(
            "/esp32/command",
            headers={"X-Device-Token": ESP32_DEVICE_TOKEN}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["command"], "STOP")

        # Step 3: ESP32 confirms stopped via ACK
        resp = self.client.post(
            "/esp32/status",
            headers={"X-Device-Token": ESP32_DEVICE_TOKEN},
            json={"state": "STOPPED"}
        )
        self.assertEqual(resp.status_code, 200)

        # Step 4: Dashboard status confirms Stopped
        resp = self.client.get("/api/esp32/status")
        self.assertEqual(resp.get_json()["state"], ESP32State.STOPPED)

    def test_audio_upload_processes_detection_pipeline(self):
        """Verify ESP32 audio upload runs detector engine and populates latest_prediction."""
        esp32_manager.state = ESP32State.STARTING
        esp32_manager.desired_state = "START"

        wav_bytes = generate_test_wav(duration_sec=1.0, sr=22050, freq=440.0)

        resp = self.client.post(
            "/esp32/test-audio",
            headers={
                "X-Device-Token": ESP32_DEVICE_TOKEN,
                "Content-Type": "audio/wav",
            },
            data=wav_bytes,
        )
        self.assertEqual(resp.status_code, 201)
        upload_data = resp.get_json()
        self.assertEqual(upload_data["status"], "received")
        self.assertEqual(upload_data["command"], "CONTINUE")
        self.assertIn("file", upload_data)

        # Verify state automatically moved from Starting to Listening upon first chunk
        status_resp = self.client.get("/api/esp32/status")
        status_data = status_resp.get_json()
        self.assertEqual(status_data["state"], ESP32State.LISTENING)
        self.assertEqual(status_data["audio_chunk_count"], 1)

        # Verify latest_prediction exists with pipeline fields
        pred = status_data["latest_prediction"]
        self.assertIsNotNone(pred)
        self.assertIn("rms", pred)
        self.assertIn("p_spray_can", pred)
        self.assertIn("waveform_points", pred)
        self.assertIn("alert_fired", pred)

        # Verify /api/esp32/latest_prediction endpoint
        pred_resp = self.client.get("/api/esp32/latest_prediction")
        self.assertEqual(pred_resp.status_code, 200)
        self.assertEqual(pred_resp.get_json()["audio_file"], upload_data["file"])

    def test_audio_upload_signals_stop_command_when_stopping(self):
        """When dashboard requested STOP, next audio upload response must return command STOP."""
        esp32_manager.state = ESP32State.STOPPING
        esp32_manager.desired_state = "STOP"

        wav_bytes = generate_test_wav(duration_sec=1.0, sr=22050)
        resp = self.client.post(
            "/esp32/test-audio",
            headers={
                "X-Device-Token": ESP32_DEVICE_TOKEN,
                "Content-Type": "audio/wav",
            },
            data=wav_bytes,
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.get_json()["command"], "STOP")

    def test_starting_timeout_transitions_to_connection_error(self):
        """If ESP32 does not acknowledge start within timeout, state becomes Connection Error."""
        esp32_manager.state = ESP32State.STARTING
        esp32_manager.desired_state = "START"
        esp32_manager.transition_time = time.time() - (esp32_manager.start_timeout + 5.0)

        resp = self.client.get("/api/esp32/status")
        data = resp.get_json()
        self.assertEqual(data["state"], ESP32State.CONNECTION_ERROR)
        self.assertIn("did not acknowledge", data["connection_error_reason"])

    def test_stream_loss_timeout_transitions_to_connection_error(self):
        """If actively Listening but no audio/heartbeat received for >timeout, state becomes Connection Error."""
        esp32_manager.state = ESP32State.LISTENING
        esp32_manager.desired_state = "START"
        esp32_manager.last_heartbeat_time = time.time() - (esp32_manager.heartbeat_timeout + 5.0)

        resp = self.client.get("/api/esp32/status")
        data = resp.get_json()
        self.assertEqual(data["state"], ESP32State.CONNECTION_ERROR)
        self.assertIn("stream disconnected", data["connection_error_reason"])

    def test_repeated_clicks_are_idempotent(self):
        """Multiple consecutive clicks to start or stop must not crash or cause inconsistent state."""
        # 1. Repeated start calls
        r1 = self.client.post("/api/esp32/start")
        r2 = self.client.post("/api/esp32/start")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(esp32_manager.state, ESP32State.STARTING)

        # Transition to Listening
        esp32_manager.state = ESP32State.LISTENING
        r3 = self.client.post("/api/esp32/start")
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.get_json()["state"], ESP32State.LISTENING)

        # 2. Repeated stop calls
        r4 = self.client.post("/api/esp32/stop")
        r5 = self.client.post("/api/esp32/stop")
        self.assertEqual(r4.status_code, 200)
        self.assertEqual(r5.status_code, 200)
        self.assertEqual(esp32_manager.state, ESP32State.STOPPING)

        esp32_manager.state = ESP32State.STOPPED
        r6 = self.client.post("/api/esp32/stop")
        self.assertEqual(r6.status_code, 200)
        self.assertEqual(r6.get_json()["state"], ESP32State.STOPPED)

    def test_audio_upload_discarded_immediately_when_stopped(self):
        """When system is in STOPPED state, any audio upload must be discarded and return STOP."""
        esp32_manager.state = ESP32State.STOPPED
        esp32_manager.desired_state = "STOP"
        esp32_manager.audio_chunk_count = 0

        wav_bytes = generate_test_wav(duration_sec=1.0, sr=22050)
        resp = self.client.post(
            "/esp32/test-audio",
            headers={
                "X-Device-Token": ESP32_DEVICE_TOKEN,
                "Content-Type": "audio/wav",
            },
            data=wav_bytes,
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertEqual(data["status"], "stopped")
        self.assertEqual(data["command"], "STOP")
        self.assertIsNone(data["file"])
        self.assertFalse(data["alert_fired"])
        # audio_chunk_count must NOT have incremented
        self.assertEqual(esp32_manager.audio_chunk_count, 0)
        self.assertEqual(esp32_manager.state, ESP32State.STOPPED)

    def test_start_stop_start_resumption_cycle(self):
        """Verify ESP32 can cycle from Stopped -> Starting -> Listening -> Stopping -> Stopped -> Resumed without reflash."""
        # 1. Start listening
        resp = self.client.post("/api/esp32/start")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["state"], ESP32State.STARTING)

        # 2. ESP32 polls and gets START
        resp = self.client.get("/esp32/command", headers={"X-Device-Token": ESP32_DEVICE_TOKEN})
        self.assertEqual(resp.get_json()["command"], "START")

        # 3. ESP32 sends ACK
        resp = self.client.post(
            "/esp32/status",
            headers={"X-Device-Token": ESP32_DEVICE_TOKEN},
            json={"state": "LISTENING"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(esp32_manager.state, ESP32State.LISTENING)

        # 4. Upload audio while listening -> should be received and processed
        wav_bytes = generate_test_wav(duration_sec=1.0, sr=22050)
        resp = self.client.post(
            "/esp32/test-audio",
            headers={"X-Device-Token": ESP32_DEVICE_TOKEN, "Content-Type": "audio/wav"},
            data=wav_bytes,
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.get_json()["status"], "received")
        self.assertEqual(resp.get_json()["command"], "CONTINUE")
        self.assertEqual(esp32_manager.audio_chunk_count, 1)

        # 5. Stop listening from dashboard
        resp = self.client.post("/api/esp32/stop")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["state"], ESP32State.STOPPING)

        # 6. ESP32 polls and gets STOP
        resp = self.client.get("/esp32/command", headers={"X-Device-Token": ESP32_DEVICE_TOKEN})
        self.assertEqual(resp.get_json()["command"], "STOP")

        # 7. ESP32 sends ACK STOPPED
        resp = self.client.post(
            "/esp32/status",
            headers={"X-Device-Token": ESP32_DEVICE_TOKEN},
            json={"state": "STOPPED"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(esp32_manager.state, ESP32State.STOPPED)

        # 8. Resume listening: click Start Listening again
        resp = self.client.post("/api/esp32/start")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["state"], ESP32State.STARTING)

        # 9. ESP32 polls and receives START again
        resp = self.client.get("/esp32/command", headers={"X-Device-Token": ESP32_DEVICE_TOKEN})
        self.assertEqual(resp.get_json()["command"], "START")

        # 10. ESP32 ACKs LISTENING
        resp = self.client.post(
            "/esp32/status",
            headers={"X-Device-Token": ESP32_DEVICE_TOKEN},
            json={"state": "LISTENING"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(esp32_manager.state, ESP32State.LISTENING)


if __name__ == "__main__":
    unittest.main()
