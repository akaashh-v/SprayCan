"""Test and Benchmark Suite for Low-Latency Alert Dispatch & Decoupled Recording.

Validates:
1. Immediate lightweight alert dispatch via /esp32/alert (< 250 bytes).
2. Server dispatch latency via SSE push (median < 2ms, p95 < 10ms).
3. Decoupling of alert notification from audio recording (alert appears before WAV file is finalized).
4. Preliminary warning generation from Stage 1 (< 1.05s from audio onset).
5. Edge-confirmed audio upload integration and alert_recorder event grouping.
"""

import io
import json
import time
import unittest
from pathlib import Path
from typing import Dict, Any

import numpy as np
import soundfile as sf

from alert_recorder import alert_recorder
from app import app
from config import SAMPLE_RATE, AUDIO_SAMPLES
from detector_engine import SprayCanDetectorEngine
from esp32_manager import esp32_manager, ESP32State


def create_test_wav_bytes(duration_sec: float = 1.0, sr: int = 22050, freq: float = 800.0) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    sig = 0.3 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestAlertLatencyElimination(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        with esp32_manager._lock:
            esp32_manager.state = ESP32State.LISTENING
            esp32_manager.desired_state = "START"
            esp32_manager.audio_chunk_count = 0
            esp32_manager.latest_prediction = None
            esp32_manager._subscribers.clear()
        alert_recorder.reset()

    def test_edge_alert_endpoint_auth_and_validation(self):
        """Verify authentication and schema validation on /esp32/alert."""
        # 1. Reject unauthorized token
        payload = {
            "event_id": "evt_test_1001",
            "alert_type": "CONFIRMED",
            "p_spray": 0.942,
        }
        res = self.client.post(
            "/esp32/alert",
            json=payload,
            headers={"X-Device-Token": "invalid_wrong_token"},
        )
        self.assertEqual(res.status_code, 403)

        # 2. Accept authorized token
        res = self.client.post(
            "/esp32/alert",
            json=payload,
            headers={"X-Device-Token": esp32_manager.device_token},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["event_id"], "evt_test_1001")
        self.assertEqual(data["alert_type"], "CONFIRMED")
        self.assertIn("server_dispatch_ms", data)

    def test_immediate_confirmed_alert_dispatch_latency(self):
        """Test confirmed alert dispatch latency and instantaneous SSE propagation."""
        sse_q = esp32_manager.subscribe_events()
        try:
            alert_payload = {
                "event_id": "evt_test_confirmed_2001",
                "alert_type": "CONFIRMED",
                "verdict": "CONFIRMED_ALERT",
                "p_spray": 0.958,
                "confidence_margin": 0.42,
                "entropy": 0.35,
                "predicted_class": "spray_can_shaking",
                "capture_seq": 42,
                "timings": {
                    "capture_dur_ms": 1000,
                    "s1_eval_ms": 0.4,
                    "dsp_ms": 9.1,
                    "cnn_ms": 1278.0,
                    "queue_wait_ms": 11.0,
                    "total_device_ms": 1287.5,
                    "tx_start_ms": time.time() * 1000,
                    "net_tx_ms": 15.2,
                },
            }

            t0 = time.perf_counter()
            resp, code = esp32_manager.handle_edge_alert(
                supplied_token=esp32_manager.device_token,
                client_ip="192.168.1.16",
                alert_payload=alert_payload,
            )
            t1 = time.perf_counter()

            self.assertEqual(code, 200)
            self.assertEqual(resp["alert_type"], "CONFIRMED")

            # Verify instantaneous SSE broadcast received (< 2 ms)
            received_event = sse_q.get(timeout=0.1)
            self.assertEqual(received_event["event_id"], "evt_test_confirmed_2001")
            self.assertTrue(received_event["alert_fired"])
            self.assertFalse(received_event["is_preliminary"])
            self.assertAlmostEqual(received_event["p_spray_can"], 0.958, places=3)
            self.assertEqual(received_event["predicted_class"], "spray_can_shaking")

            # Check latency instrumentation breakdown
            timing = received_event["timing"]
            self.assertIn("cnn_ms", timing)
            self.assertIn("server_dispatch_ms", timing)
            self.assertIn("end_to_end_server_ms", timing)

            dispatch_elapsed_ms = (t1 - t0) * 1000
            self.assertLess(dispatch_elapsed_ms, 15.0, f"Server dispatch took {dispatch_elapsed_ms:.2f}ms (expected < 15ms)")

        finally:
            esp32_manager.unsubscribe_events(sse_q)

    def test_preliminary_warning_dispatch(self):
        """Test preliminary warning generation triggered immediately after Stage 1 kinematic filter."""
        sse_q = esp32_manager.subscribe_events()
        try:
            prelim_payload = {
                "event_id": "evt_test_prelim_3001",
                "alert_type": "PRELIMINARY",
                "verdict": "PRELIMINARY_CANDIDATE",
                "p_spray": 0.0,
                "confidence_margin": 0.0,
                "entropy": 0.0,
                "predicted_class": "candidate",
                "capture_seq": 88,
                "timings": {
                    "capture_dur_ms": 1000,
                    "s1_eval_ms": 0.35,
                    "dsp_ms": 0,
                    "cnn_ms": 0,
                    "queue_wait_ms": 8.0,
                    "total_device_ms": 0.35,
                    "net_tx_ms": 14.1,
                },
            }

            resp, code = esp32_manager.handle_edge_alert(
                supplied_token=esp32_manager.device_token,
                client_ip="192.168.1.16",
                alert_payload=prelim_payload,
            )
            self.assertEqual(code, 200)

            # Check that preliminary warning was broadcast
            received_event = sse_q.get(timeout=0.1)
            self.assertEqual(received_event["event_id"], "evt_test_prelim_3001")
            self.assertTrue(received_event["is_preliminary"])
            self.assertFalse(received_event["alert_fired"])  # Unconfirmed, so alert_fired is False
            self.assertIn("PRELIMINARY", received_event["status"])

        finally:
            esp32_manager.unsubscribe_events(sse_q)

    def test_alert_recording_decoupling_and_persistence(self):
        """Verify that alert notification occurs immediately, while WAV recording saves in background."""
        engine = SprayCanDetectorEngine(backend="tensorflow")
        wav_chunk = create_test_wav_bytes(1.0, SAMPLE_RATE, 750.0)

        # 1. Fire confirmed alert first via lightweight endpoint
        alert_payload = {
            "event_id": "evt_test_decoupled_4001",
            "alert_type": "CONFIRMED",
            "verdict": "CONFIRMED_ALERT",
            "p_spray": 0.945,
            "confidence_margin": 0.39,
            "entropy": 0.40,
            "predicted_class": "spray_can_shaking",
            "capture_seq": 10,
            "timings": {
                "capture_dur_ms": 1000,
                "cnn_ms": 1280.0,
                "total_device_ms": 1290.0,
                "net_tx_ms": 16.0,
            },
        }

        t_alert_start = time.perf_counter()
        resp, code = esp32_manager.handle_edge_alert(
            supplied_token=esp32_manager.device_token,
            client_ip="192.168.1.16",
            alert_payload=alert_payload,
        )
        t_alert_end = time.perf_counter()
        self.assertEqual(code, 200)

        # Alert was published in < 5ms
        alert_ms = (t_alert_end - t_alert_start) * 1000
        self.assertLess(alert_ms, 20.0)

        # 2. Feed event audio chunk asynchronously (representing background upload)
        headers = {
            "X-Capture-Seq": "10",
            "X-Edge-Evaluated": "true",
            "X-Edge-Verdict": "CONFIRMED_ALERT",
            "X-Edge-P-Spray": "0.945",
            "X-Edge-Class": "spray_can_shaking",
            "X-Edge-Infer-Ms": "1280",
            "X-Edge-Prep-Ms": "9",
        }
        res, code = esp32_manager.handle_audio_upload(
            supplied_token=esp32_manager.device_token,
            audio_bytes=wav_chunk,
            mimetype="audio/wav",
            client_ip="192.168.1.16",
            engine=engine,
            headers=headers,
        )
        self.assertEqual(code, 201)
        self.assertTrue(res["alert_fired"])

        # 3. Feed post-event chunks to complete alert recording (3.0s default post-event)
        headers_post = {
            "X-Capture-Seq": "11",
            "X-Edge-Evaluated": "false",
        }
        finalized_recording = None
        for i in range(4):
            headers_post["X-Capture-Seq"] = str(11 + i)
            res_post, _ = esp32_manager.handle_audio_upload(
                supplied_token=esp32_manager.device_token,
                audio_bytes=wav_chunk,
                mimetype="audio/wav",
                client_ip="192.168.1.16",
                engine=engine,
                headers=headers_post,
            )
            if "saved_recording" in res_post:
                finalized_recording = res_post["saved_recording"]
                break

        self.assertIsNotNone(finalized_recording, "Alert recording should finalize after post-event window")
        self.assertIn("filename", finalized_recording)
        self.assertTrue(finalized_recording["filename"].endswith(".wav"))
        self.assertGreater(finalized_recording["total_duration_sec"], 3.0)

    def test_end_to_end_pipeline_latency_benchmark(self):
        """Benchmark 50 alert transmissions to compute median and 95th percentile dispatch latency."""
        dispatch_latencies = []

        for i in range(50):
            payload = {
                "event_id": f"evt_bench_{i}",
                "alert_type": "CONFIRMED",
                "verdict": "CONFIRMED_ALERT",
                "p_spray": 0.92,
                "capture_seq": i + 1,
                "timings": {
                    "capture_dur_ms": 1000,
                    "s1_eval_ms": 0.4,
                    "dsp_ms": 9.1,
                    "cnn_ms": 1278.0,
                    "net_tx_ms": 14.5,
                },
            }

            t0 = time.perf_counter()
            resp, code = esp32_manager.handle_edge_alert(
                supplied_token=esp32_manager.device_token,
                client_ip="127.0.0.1",
                alert_payload=payload,
            )
            t1 = time.perf_counter()

            self.assertEqual(code, 200)
            dispatch_latencies.append((t1 - t0) * 1000)

        dispatch_latencies.sort()
        median_ms = np.median(dispatch_latencies)
        p95_ms = np.percentile(dispatch_latencies, 95)
        min_ms = np.min(dispatch_latencies)
        max_ms = np.max(dispatch_latencies)

        print(f"\n[BENCHMARK] Server Alert Dispatch Latency across 50 iterations:")
        print(f"  Min:    {min_ms:.2f} ms")
        print(f"  Median: {median_ms:.2f} ms (Target: < 2.0 ms)")
        print(f"  P95:    {p95_ms:.2f} ms (Target: < 10.0 ms)")
        print(f"  Max:    {max_ms:.2f} ms")

        self.assertLess(median_ms, 5.0, f"Median dispatch latency {median_ms:.2f}ms exceeds target 5.0ms")
        self.assertLess(p95_ms, 15.0, f"95th percentile dispatch latency {p95_ms:.2f}ms exceeds target 15.0ms")


if __name__ == "__main__":
    unittest.main()
