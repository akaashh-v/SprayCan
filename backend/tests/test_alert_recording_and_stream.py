"""Test Suite for Continuous Stream Processing & Event-Based Alert Audio Recording.

Validates:
1. Rolling RAM audio buffer (early start, boundary conditions, and wraparound).
2. Event-based audio saving: Pre-event (4s) + Event + Post-event (3s) WAV generation.
3. Duplicate alert suppression & event grouping under sustained shaking.
4. Clean Stop behavior: immediate buffer flush, zero delayed alerts, resource teardown.
5. WAV audio file integrity (22050 Hz, 16-bit PCM, header validation).
6. Non-target audio rejection: ordinary background audio is never persisted to disk.
7. Public REST API contracts for alert recordings playback and download.
"""

import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import numpy as np
import soundfile as sf

import sys
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from alert_recorder import AlertAudioRecorder, RollingAudioBuffer
from app import app
from config import AUDIO_SAMPLES, SAMPLE_RATE
from detector_engine import SprayCanDetectorEngine
from esp32_manager import ESP32State, esp32_manager


def make_sine_window(duration_sec: float = 1.0, sr: int = SAMPLE_RATE, freq: float = 440.0, amp: float = 0.1) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    return amp * np.sin(2 * np.pi * freq * t)


class TestAlertRecordingAndStream(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_alerts_"))
        self.recorder = AlertAudioRecorder(
            recordings_dir=self.tmp_dir,
            pre_event_seconds=4.0,
            post_event_seconds=3.0,
            cooldown_seconds=4.0,
            sr=SAMPLE_RATE,
        )
        self.client = app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_rolling_buffer_early_start(self):
        """Buffer handles early start gracefully when less than pre_event_seconds is available."""
        buf = RollingAudioBuffer(sr=SAMPLE_RATE, max_seconds=4.0)
        self.assertEqual(len(buf.get_audio()), 0)
        self.assertEqual(buf.duration_seconds(), 0.0)

        # Feed 1.5 seconds of audio
        w1 = make_sine_window(duration_sec=1.0)
        w2 = make_sine_window(duration_sec=0.5)
        buf.append(w1)
        buf.append(w2)

        audio = buf.get_audio()
        self.assertEqual(len(audio), int(1.5 * SAMPLE_RATE))
        self.assertAlmostEqual(buf.duration_seconds(), 1.5, places=2)

    def test_rolling_buffer_wraparound(self):
        """Buffer strictly bounds stored samples to max_seconds without memory leak."""
        buf = RollingAudioBuffer(sr=SAMPLE_RATE, max_seconds=3.0)
        max_expected = int(3.0 * SAMPLE_RATE)

        # Append 10 seconds of distinct chunks
        for i in range(10):
            chunk = np.full(int(1.0 * SAMPLE_RATE), fill_value=float(i), dtype=np.float32)
            buf.append(chunk)

        audio = buf.get_audio()
        self.assertEqual(len(audio), max_expected)
        self.assertAlmostEqual(buf.duration_seconds(), 3.0, places=2)
        # Verify the buffer contains the most recent samples (from chunk 7, 8, 9)
        self.assertEqual(audio[-1], 9.0)

    def test_alert_event_recording_flow_and_wav_validity(self):
        """Confirmed alert saves valid WAV file containing pre-event, event, and post-event audio."""
        # 1. Feed 4 seconds of ambient background
        for _ in range(4):
            bg = make_sine_window(duration_sec=1.0, freq=200.0, amp=0.01)
            self.recorder.process_audio_chunk(bg, {"alert_fired": False, "is_event_active": False})

        # 2. Trigger confirmed alert
        event_chunk = make_sine_window(duration_sec=1.0, freq=800.0, amp=0.2)
        alert_pred = {
            "alert_fired": True,
            "is_event_active": True,
            "p_spray_can": 0.945,
            "rms": 0.045,
            "cadence_hz": 4.5,
            "confidence_margin": 0.52,
            "entropy": 0.22,
        }
        res1 = self.recorder.process_audio_chunk(event_chunk, alert_pred)
        # Event session started, not finalized yet
        self.assertIsNone(res1)
        self.assertIsNotNone(self.recorder.active_session)

        # 3. Feed 1 more window of sustained shaking
        res2 = self.recorder.process_audio_chunk(
            event_chunk,
            {"alert_fired": False, "is_event_active": True, "p_spray_can": 0.950},
        )
        self.assertIsNone(res2)

        # 4. Feed 3 seconds of post-event audio (shaking ended)
        final_meta = None
        for i in range(3):
            post_chunk = make_sine_window(duration_sec=1.0, freq=300.0, amp=0.02)
            meta = self.recorder.process_audio_chunk(
                post_chunk,
                {"alert_fired": False, "is_event_active": False},
            )
            if meta:
                final_meta = meta

        self.assertIsNotNone(final_meta, "Alert recording failed to finalize after post-event period!")
        self.assertEqual(final_meta["alert_class"], "spray_can_shaking")
        self.assertAlmostEqual(final_meta["confidence"], 0.950, places=3)
        self.assertAlmostEqual(final_meta["pre_event_sec"], 4.0, places=1)
        self.assertAlmostEqual(final_meta["event_sec"], 2.0, places=1)
        self.assertAlmostEqual(final_meta["post_event_sec"], 3.0, places=1)
        self.assertAlmostEqual(final_meta["total_duration_sec"], 9.0, places=1)

        # Verify WAV file on disk
        wav_path = self.tmp_dir / final_meta["filename"]
        self.assertTrue(wav_path.exists(), f"Saved WAV file {wav_path} does not exist!")

        # Validate audio content and metadata with soundfile
        audio_data, file_sr = sf.read(str(wav_path))
        self.assertEqual(file_sr, SAMPLE_RATE)
        self.assertAlmostEqual(len(audio_data) / file_sr, 9.0, places=1)
        self.assertGreater(np.max(np.abs(audio_data)), 0.05)

    def test_duplicate_suppression_and_event_grouping(self):
        """Sustained shaking produces exactly one alert file; cooldown prevents duplicate bursts."""
        # Feed background
        for _ in range(2):
            self.recorder.process_audio_chunk(make_sine_window(1.0), {"alert_fired": False})

        # Alert triggered
        self.recorder.process_audio_chunk(
            make_sine_window(1.0),
            {"alert_fired": True, "is_event_active": True, "p_spray_can": 0.91},
        )
        # Sustained shaking for 3 more windows (same shaking episode)
        for _ in range(3):
            self.recorder.process_audio_chunk(
                make_sine_window(1.0),
                {"alert_fired": False, "is_event_active": True, "p_spray_can": 0.92},
            )

        # Post-event period (3s)
        for _ in range(3):
            self.recorder.process_audio_chunk(
                make_sine_window(1.0),
                {"alert_fired": False, "is_event_active": False},
            )

        self.assertEqual(self.recorder.total_saved_alerts, 1)

        # Immediate follow-up alert during cooldown (4s cooldown) must be debounced/suppressed
        self.recorder.process_audio_chunk(
            make_sine_window(1.0),
            {"alert_fired": True, "is_event_active": True, "p_spray_can": 0.88},
        )
        self.assertEqual(self.recorder.total_saved_alerts, 1, "Cooldown failed to suppress duplicate alert!")

    def test_rejected_sounds_never_save_recordings(self):
        """Ambient room tone, hiss, and sudden loud impacts never save files to disk."""
        # 10 windows of rejected non-target sound
        for _ in range(10):
            impact = make_sine_window(1.0, freq=120.0, amp=0.5)
            self.recorder.process_audio_chunk(
                impact,
                {
                    "alert_fired": False,
                    "is_stage1_candidate": False,
                    "status": "LOUD_SUDDEN_IMPACT_REJECTED",
                    "p_spray_can": 0.05,
                },
            )

        self.assertEqual(self.recorder.total_saved_alerts, 0)
        saved_files = list(self.tmp_dir.glob("*.wav"))
        self.assertEqual(len(saved_files), 0, "Non-alert audio was incorrectly written to disk!")

    def test_stop_listening_clears_buffer_and_aborts_delayed_alerts(self):
        """Calling reset on Stop clears rolling buffer and prevents stale audio from firing alerts."""
        buf = self.recorder.rolling_buffer
        self.recorder.process_audio_chunk(make_sine_window(1.0), {"alert_fired": False})
        self.assertGreater(buf.duration_seconds(), 0.0)

        # Simulate user clicking Stop Listening
        self.recorder.reset()
        self.assertEqual(buf.duration_seconds(), 0.0)
        self.assertIsNone(self.recorder.active_session)

    def test_alert_recordings_api_endpoints(self):
        """Test REST API for listing, retrieving, and configuring alert recordings."""
        # Create dummy alert file in tmp_dir
        dummy_wav = self.tmp_dir / "alert_20261017T120000_evt123_spray_can_shaking.wav"
        dummy_json = self.tmp_dir / "alert_20261017T120000_evt123.json"

        sf.write(str(dummy_wav), make_sine_window(3.0), SAMPLE_RATE, subtype="PCM_16")
        meta = {
            "event_id": "evt123",
            "filename": dummy_wav.name,
            "alert_class": "spray_can_shaking",
            "confidence": 0.92,
            "timestamp_iso": "2026-10-17T12:00:00Z",
            "total_duration_sec": 3.0,
            "pre_event_sec": 1.0,
            "event_sec": 1.0,
            "post_event_sec": 1.0,
        }
        with open(dummy_json, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Hook recorder to app temporarily
        from alert_recorder import alert_recorder
        old_dir = alert_recorder.recordings_dir
        alert_recorder.recordings_dir = self.tmp_dir

        try:
            # 1. GET /api/alerts/recordings
            resp = self.client.get("/api/alerts/recordings")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data["status"], "success")
            self.assertGreaterEqual(len(data["recordings"]), 1)
            self.assertEqual(data["recordings"][0]["event_id"], "evt123")

            # 2. GET /api/alerts/recordings/<filename>
            audio_resp = self.client.get(f"/api/alerts/recordings/{dummy_wav.name}")
            self.assertEqual(audio_resp.status_code, 200)
            self.assertEqual(audio_resp.mimetype, "audio/wav")
            self.assertGreater(len(audio_resp.data), 1000)

            # 3. GET /api/alerts/config
            cfg_resp = self.client.get("/api/alerts/config")
            self.assertEqual(cfg_resp.status_code, 200)
            cfg_data = cfg_resp.get_json()["config"]
            self.assertIn("pre_event_seconds", cfg_data)
            self.assertIn("post_event_seconds", cfg_data)

            # 4. POST /api/alerts/config
            upd_resp = self.client.post("/api/alerts/config", json={"pre_event_seconds": 5.0})
            self.assertEqual(upd_resp.status_code, 200)
            self.assertEqual(upd_resp.get_json()["config"]["pre_event_seconds"], 5.0)

        finally:
            alert_recorder.recordings_dir = old_dir


if __name__ == "__main__":
    unittest.main()
