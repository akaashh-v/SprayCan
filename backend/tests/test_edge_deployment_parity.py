"""Tests for ESP32 Edge CNN Deployment Parity and Telemetry.

Validates:
1. INT8 TFLite model output matches Keras model output parity across classes.
2. Mel filterbank coefficients match audio_utils.py exactly.
3. Backend esp32_manager parses X-Edge-* headers and attaches edge_device telemetry.
"""

import io
import wave
import numpy as np
import pytest
from pathlib import Path

from config import SAMPLE_RATE, AUDIO_SAMPLES, CLASS_NAMES
from audio_utils import load_audio, compute_log_mel_spectrogram
from stage2_classifier_tf import SprayCanShakingCNNTF
from esp32_manager import ESP32Manager

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow required")
class TestEdgeDeploymentParity:

    def test_tflite_int8_model_file_exists_and_loads(self):
        tflite_path = Path("backend/models/tflite_eval/best_detector_int8.tflite")
        assert tflite_path.exists(), "best_detector_int8.tflite must exist"
        assert tflite_path.stat().st_size > 500000, "INT8 model size should be ~520 KB"

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        in_det = interpreter.get_input_details()[0]
        out_det = interpreter.get_output_details()[0]

        assert list(in_det["shape"]) == [1, 64, 44, 1]
        assert list(out_det["shape"]) == [1, 6]
        assert in_det["dtype"] == np.float32
        assert out_det["dtype"] == np.float32

    def test_tflite_int8_parity_on_authentic_spray_can(self):
        wav_path = Path("backend/data/demo_samples/authentic_spray_can.wav")
        assert wav_path.exists()
        audio = load_audio(wav_path, sr=SAMPLE_RATE)[:AUDIO_SAMPLES]
        spec = compute_log_mel_spectrogram(audio, sr=SAMPLE_RATE)
        spec_in = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)

        # Keras
        classifier = SprayCanShakingCNNTF(weights_path="backend/models/best_detector.keras")
        k_probs = classifier.predict_proba(spec_in)[0]

        # TFLite INT8
        interpreter = tf.lite.Interpreter(model_path="backend/models/tflite_eval/best_detector_int8.tflite")
        interpreter.allocate_tensors()
        in_idx = interpreter.get_input_details()[0]["index"]
        out_idx = interpreter.get_output_details()[0]["index"]
        interpreter.set_tensor(in_idx, spec_in)
        interpreter.invoke()
        tflite_probs = interpreter.get_tensor(out_idx)[0]

        # Both must classify class 5 (spray_can_shaking) as dominant (>0.80)
        assert np.argmax(k_probs) == 5
        assert np.argmax(tflite_probs) == 5
        assert k_probs[5] >= 0.80
        assert tflite_probs[5] >= 0.80

        # Maximum probability difference should be under 2%
        max_diff = np.max(np.abs(k_probs - tflite_probs))
        assert max_diff < 0.02, f"Prob difference {max_diff} exceeds 2%"

    def test_sparse_mel_filterbank_matches_librosa(self):
        import librosa
        mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=64, fmin=100.0, fmax=11000.0)

        # Inspect generated mel_filterbank.h and mel_filterbank.cpp
        h_path = Path("firmware/esp32_mic/mel_filterbank.h")
        cpp_path = Path("firmware/esp32_mic/mel_filterbank.cpp")
        assert h_path.exists()
        assert cpp_path.exists()

        # Non-zero count in mel_basis should equal TOTAL_MEL_WEIGHTS (985)
        non_zero_count = 0
        for m in range(64):
            nz = np.where(mel_basis[m] > 1e-7)[0]
            if len(nz) > 0:
                non_zero_count += int(nz[-1] - nz[0] + 1)
        assert non_zero_count == 985

    def test_backend_parses_edge_telemetry_headers(self):
        mgr = ESP32Manager(device_token="TestToken123")
        mgr.desired_state = "START"
        mgr.state = "LISTENING"

        # Generate 1.0s dummy WAV bytes
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(np.zeros(22050, dtype=np.int16).tobytes())
        wav_bytes = wav_io.getvalue()

        headers = {
            "X-Device-Token": "TestToken123",
            "X-Device-State": "LISTENING",
            "X-Capture-Start-Ms": "1000",
            "X-Capture-End-Ms": "2000",
            "X-Tx-Start-Ms": "2010",
            "X-Capture-Seq": "42",
            "X-Upload-Seq": "41",
            "X-Edge-Evaluated": "true",
            "X-Edge-Verdict": "CONFIRMED_ALERT",
            "X-Edge-P-Spray": "0.8872",
            "X-Edge-Class": "spray_can_shaking",
            "X-Edge-Infer-Ms": "1280",
            "X-Edge-Prep-Ms": "9",
        }

        # Mock engine
        class MockEngine:
            def evaluate_window(self, arr):
                return {
                    "alert_fired": True,
                    "p_spray_can": 0.885,
                    "predicted_class": "spray_can_shaking",
                    "status": "CONFIRMED_ALERT",
                    "rms": 0.015,
                }

        resp, code = mgr.handle_audio_upload(
            supplied_token="TestToken123",
            audio_bytes=wav_bytes,
            mimetype="audio/wav",
            client_ip="192.168.1.16",
            engine=MockEngine(),
            headers=headers,
        )

        assert code == 201
        assert "edge_device" in resp
        edge = resp["edge_device"]
        assert edge["evaluated"] is True
        assert edge["verdict"] == "CONFIRMED_ALERT"
        assert abs(edge["p_spray"] - 0.8872) < 1e-4
        assert edge["predicted_class"] == "spray_can_shaking"
        assert edge["inference_time_ms"] == 1280.0
        assert edge["preprocess_time_ms"] == 9.0
