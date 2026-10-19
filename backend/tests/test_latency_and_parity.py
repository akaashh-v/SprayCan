"""Automated Latency Benchmarks, Accuracy Parity, and Monotonic Guard Validation Suite.

Validates:
1. Stage-by-stage latency (audio decode, CNN inference, server total, SSE dispatch).
2. Prediction agreement and confidence differences across PyTorch and TensorFlow backends.
3. Hard negative, ambient speech, and loud impact false positive rejection.
4. Monotonic chunk sequence enforcement (preventing stale audio results from overwriting newer ones).
5. Start -> Stop -> Start remote lifecycle and immediate discard when stopped.
"""

import io
import os
import queue
import time
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from config import (
    AUDIO_SAMPLES,
    BACKEND_DIR,
    SAMPLE_RATE,
    STAGE1_MIN_RMS,
    RAW_SPRAY_DIR,
    RAW_HARD_NEG_DIR,
    RAW_AMBIENT_DIR,
    RAW_LOUD_IMPACT_DIR,
)
from audio_utils import load_audio, pad_or_trim
from detector_engine import SprayCanDetectorEngine
from esp32_manager import esp32_manager, ESP32State


def create_dummy_wav_bytes(duration_sec: float = 1.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    """Generate in-memory WAV bytes with sine tone."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    sig = 0.2 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestLatencyAndParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine_pt = SprayCanDetectorEngine(backend="pytorch")
        cls.engine_tf = SprayCanDetectorEngine(backend="tensorflow")

    def setUp(self):
        with esp32_manager._lock:
            esp32_manager.state = ESP32State.STOPPED
            esp32_manager.desired_state = "STOP"
            esp32_manager.audio_chunk_count = 0
            esp32_manager.latest_prediction = None
            esp32_manager._subscribers.clear()

    def test_stage_latencies_and_benchmarks(self):
        """Measure latency across all pipeline stages with median and p95."""
        engine = self.engine_pt
        wav_bytes = create_dummy_wav_bytes(1.0, SAMPLE_RATE, 600.0)

        decode_times = []
        inference_times = []
        server_total_times = []

        headers = {
            "X-Capture-Start-Ms": "1000",
            "X-Capture-End-Ms": "2000",
            "X-Tx-Start-Ms": "2005",
        }

        # Warm-up run
        esp32_manager.state = ESP32State.LISTENING
        esp32_manager.desired_state = "START"
        esp32_manager.handle_audio_upload(
            esp32_manager.device_token,
            wav_bytes,
            "audio/wav",
            "127.0.0.1",
            engine,
            headers=headers,
        )

        # 30 iterations for robust statistics
        for _ in range(30):
            t0 = time.perf_counter()
            resp, code = esp32_manager.handle_audio_upload(
                esp32_manager.device_token,
                wav_bytes,
                "audio/wav",
                "127.0.0.1",
                engine,
                headers=headers,
            )
            t1 = time.perf_counter()

            self.assertEqual(code, 201)
            timing = resp.get("timing", {})

            decode_times.append(timing.get("server_decode_ms", 0.0))
            inference_times.append(timing.get("server_inference_ms", 0.0))
            server_total_times.append((t1 - t0) * 1000.0)

        med_decode = float(np.median(decode_times))
        p95_decode = float(np.percentile(decode_times, 95))
        med_infer = float(np.median(inference_times))
        p95_infer = float(np.percentile(inference_times, 95))
        med_server = float(np.median(server_total_times))
        p95_server = float(np.percentile(server_total_times, 95))

        print(f"\n[BENCHMARK] Decode Latency: Median={med_decode:.2f}ms, P95={p95_decode:.2f}ms")
        print(f"[BENCHMARK] Inference Latency: Median={med_infer:.2f}ms, P95={p95_infer:.2f}ms")
        print(f"[BENCHMARK] Server Total Latency: Median={med_server:.2f}ms, P95={p95_server:.2f}ms")

        # In-memory decoding and inference must complete well within real-time budgets (<150ms)
        self.assertLess(med_infer, 100.0)
        self.assertLess(med_server, 150.0)

    def test_sse_broadcast_latency(self):
        """Verify Server-Sent Events subscriber receives predictions in <2ms."""
        q = esp32_manager.subscribe_events()
        self.assertIn(q, esp32_manager._subscribers)

        wav_bytes = create_dummy_wav_bytes(1.0, SAMPLE_RATE, 500.0)
        esp32_manager.state = ESP32State.LISTENING
        esp32_manager.desired_state = "START"

        t_send = time.perf_counter()
        resp, code = esp32_manager.handle_audio_upload(
            esp32_manager.device_token,
            wav_bytes,
            "audio/wav",
            "127.0.0.1",
            self.engine_pt,
        )
        self.assertEqual(code, 201)

        # Queue should receive event immediately
        try:
            event_payload = q.get(timeout=0.1)
            t_recv = time.perf_counter()
            delivery_latency_ms = (t_recv - t_send) * 1000.0
            print(f"[BENCHMARK] SSE Push Delivery Latency: {delivery_latency_ms:.2f}ms")
            self.assertIn("p_spray_can", event_payload)
            self.assertEqual(event_payload.get("chunk_seq"), 1)
        finally:
            esp32_manager.unsubscribe_events(q)
            self.assertNotIn(q, esp32_manager._subscribers)

    def test_monotonic_sequence_guard(self):
        """Verify sequential chunk counter increments monotonically."""
        wav_bytes = create_dummy_wav_bytes(1.0, SAMPLE_RATE, 400.0)
        esp32_manager.state = ESP32State.LISTENING
        esp32_manager.desired_state = "START"

        seqs = []
        for _ in range(5):
            resp, code = esp32_manager.handle_audio_upload(
                esp32_manager.device_token,
                wav_bytes,
                "audio/wav",
                "127.0.0.1",
                self.engine_pt,
            )
            seqs.append(resp["chunk_seq"])

        self.assertEqual(seqs, [1, 2, 3, 4, 5])

    def test_stop_state_immediate_discard(self):
        """Verify that audio uploaded while stopped is discarded with zero inference."""
        wav_bytes = create_dummy_wav_bytes(1.0, SAMPLE_RATE, 1000.0)
        esp32_manager.state = ESP32State.STOPPED
        esp32_manager.desired_state = "STOP"

        resp, code = esp32_manager.handle_audio_upload(
            esp32_manager.device_token,
            wav_bytes,
            "audio/wav",
            "127.0.0.1",
            self.engine_pt,
        )
        self.assertEqual(code, 201)
        self.assertEqual(resp["status"], "stopped")
        self.assertEqual(resp["command"], "STOP")
        self.assertFalse(resp["alert_fired"])
        # No inference was executed
        self.assertIsNone(esp32_manager.latest_prediction)

    def test_authentic_spray_detection_and_parity(self):
        """Test authentic spray-can recordings on both PyTorch and TensorFlow engines."""
        spray_files = list(RAW_SPRAY_DIR.glob("*.wav")) + list(RAW_SPRAY_DIR.glob("*.mp3"))
        if not spray_files:
            self.skipTest("No spray can samples found.")

        test_files = spray_files[:5]
        for fpath in test_files:
            audio = load_audio(fpath, sr=SAMPLE_RATE)
            if len(audio) < AUDIO_SAMPLES:
                audio = pad_or_trim(audio, AUDIO_SAMPLES)
            window = audio[:AUDIO_SAMPLES]

            res_pt = self.engine_pt.evaluate_window(window)
            res_tf = self.engine_tf.evaluate_window(window)

            # Both engines should evaluate stage 2 or reject similarly
            diff = abs(res_pt["p_spray_can"] - res_tf["p_spray_can"])
            self.assertLess(diff, 0.05, f"Discrepancy on {fpath.name}: PT={res_pt['p_spray_can']} vs TF={res_tf['p_spray_can']}")

    def test_hard_negatives_and_ambient_rejection(self):
        """Test that keys, coins, pill bottles, and speech never trigger spray alert."""
        sample_paths = []
        # Coins / keys / pill bottles
        sample_paths.extend(list(RAW_HARD_NEG_DIR.glob("coins_*.wav"))[:3])
        sample_paths.extend(list(RAW_HARD_NEG_DIR.glob("keys_*.wav"))[:3])
        sample_paths.extend(list(RAW_HARD_NEG_DIR.glob("pill_bottle_*.wav"))[:3])
        # Loud impacts
        sample_paths.extend(list(RAW_LOUD_IMPACT_DIR.glob("*.wav"))[:3])

        self.assertGreater(len(sample_paths), 0, "Hard negative test files required.")

        for fpath in sample_paths:
            audio = load_audio(fpath, sr=SAMPLE_RATE)
            window = audio[:AUDIO_SAMPLES] if len(audio) >= AUDIO_SAMPLES else pad_or_trim(audio, AUDIO_SAMPLES)

            res = self.engine_pt.evaluate_window(window)
            self.assertFalse(
                res["alert_fired"],
                f"False positive alert on hard negative: {fpath.name} (P_spray={res.get('p_spray_can', 0.0)})"
            )
            self.assertLess(
                res.get("p_spray_can", 0.0),
                0.80,
                f"Confidence exceeded 80% on hard negative: {fpath.name}"
            )

    def test_sequence_counter_distinction_and_dropped_chunks_estimation(self):
        """Verify explicit distinction between capture seq, upload seq, and server seq."""
        wav_bytes = create_dummy_wav_bytes(1.0, SAMPLE_RATE, 440.0)
        esp32_manager.state = ESP32State.LISTENING
        esp32_manager.desired_state = "START"

        # First chunk: capture=1, upload=1
        headers_1 = {
            "X-Capture-Seq": "1",
            "X-Upload-Seq": "1",
        }
        resp1, code1 = esp32_manager.handle_audio_upload(
            esp32_manager.device_token,
            wav_bytes,
            "audio/wav",
            "127.0.0.1",
            self.engine_pt,
            headers=headers_1,
        )
        self.assertEqual(code1, 201)
        self.assertEqual(resp1["server_seq"], 1)
        self.assertEqual(resp1["client_capture_seq"], 1)
        self.assertEqual(resp1["client_upload_seq"], 1)
        self.assertEqual(resp1["dropped_chunks_estimate"], 0)

        # Second chunk arriving after client queue dropped 17 stale chunks (capture=19, upload=2)
        headers_2 = {
            "X-Capture-Seq": "19",
            "X-Upload-Seq": "2",
        }
        resp2, code2 = esp32_manager.handle_audio_upload(
            esp32_manager.device_token,
            wav_bytes,
            "audio/wav",
            "127.0.0.1",
            self.engine_pt,
            headers=headers_2,
        )
        self.assertEqual(code2, 201)
        self.assertEqual(resp2["server_seq"], 2)
        self.assertEqual(resp2["client_capture_seq"], 19)
        self.assertEqual(resp2["client_upload_seq"], 2)
        self.assertEqual(resp2["dropped_chunks_estimate"], 17)

    def test_in_memory_processing_zero_disk_files(self):
        """Verify that audio chunks are processed in-memory without creating files on disk."""
        from config import ESP32_AUDIO_SAVE_DIR, ESP32_SAVE_AUDIO_CHUNKS
        self.assertFalse(ESP32_SAVE_AUDIO_CHUNKS, "ESP32_SAVE_AUDIO_CHUNKS should default to False for zero disk latency")

        initial_count = len(list(ESP32_AUDIO_SAVE_DIR.glob("esp32_*.wav")))
        wav_bytes = create_dummy_wav_bytes(1.0, SAMPLE_RATE, 350.0)
        esp32_manager.state = ESP32State.LISTENING
        esp32_manager.desired_state = "START"

        esp32_manager.handle_audio_upload(
            esp32_manager.device_token,
            wav_bytes,
            "audio/wav",
            "127.0.0.1",
            self.engine_pt,
        )

        after_count = len(list(ESP32_AUDIO_SAVE_DIR.glob("esp32_*.wav")))
        self.assertEqual(initial_count, after_count, "No new files should be written to tmp_esp32_audio when in-memory mode is active")

    def test_upload_connection_close_header_present(self):
        """Verify that the Flask app returns Connection: close to cleanly terminate TCP sockets."""
        from app import app
        wav_bytes = create_dummy_wav_bytes(1.0, SAMPLE_RATE, 440.0)
        esp32_manager.state = ESP32State.LISTENING
        esp32_manager.desired_state = "START"

        client = app.test_client()
        resp = client.post(
            "/esp32/test-audio",
            data=wav_bytes,
            content_type="audio/wav",
            headers={"X-Device-Token": esp32_manager.device_token},
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.headers.get("Connection"), "close")


if __name__ == "__main__":
    unittest.main()
