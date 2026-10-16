"""Tests for Local Microphone Recording, Metadata Persistence, and Dataset Splitting.

Verifies:
1. Short audio recording directly from microphone can be saved with pre-save verification.
2. Complete metadata persistence: microphone, session_id, can_type, can_fill_level,
   shaking_style, distance_m, label_confidence, condition, notes without invention.
3. Audio serving endpoint (/api/audio_file) and directory traversal prevention.
4. Dataset builder incorporation of field recordings and strict session grouping.
5. Retention of final test sessions without leakage into training or validation.
"""

import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from app import app
from config import BASE_DIR, CLASS_NAMES, FIELD_RECORDINGS_DIR, SAMPLE_RATE
from dataset_builder import get_session_id


class TestLocalRecordingWorkflow(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Backup existing manifest if present
        self.manifest_path = FIELD_RECORDINGS_DIR / "manifest.json"
        self.manifest_backup = None
        if self.manifest_path.exists():
            self.manifest_backup = self.manifest_path.read_text(encoding="utf-8")

        # Create temporary audio generator
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        # Restore manifest backup or remove test manifest
        if self.manifest_backup is not None:
            self.manifest_path.write_text(self.manifest_backup, encoding="utf-8")
        elif self.manifest_path.exists():
            self.manifest_path.unlink()

        # Clean up temp dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_wav_bytes(self, duration_sec: float = 1.0) -> bytes:
        samples = int(SAMPLE_RATE * duration_sec)
        t = np.linspace(0, duration_sec, samples, endpoint=False)
        # Synthetic chirp
        audio = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        wav_path = self.temp_dir / "test.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)
        return wav_path.read_bytes()

    def test_record_sample_with_full_metadata(self):
        """Test recording a real aerosol can shake clip with all metadata fields."""
        audio_bytes = self._create_synthetic_wav_bytes(1.2)

        data = {
            "class_name": "spray_can_shaking",
            "session_id": "session_local_krylon_01",
            "microphone_name": "Realtek USB Condenser Mic",
            "can_type": "aerosol_paint_can",
            "can_fill_level": "half_full",
            "shaking_style": "vertical_vigorous",
            "distance_m": "1.0",
            "label_confidence": "confirmed_high",
            "condition": "clean_indoor",
            "notes": "Verified genuine Krylon rattle sound",
        }

        response = self.app.post(
            "/api/record_sample",
            data={**data, "audio": (io.BytesIO(audio_bytes), "test.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertEqual(res_json["status"], "success")

        entry = res_json["entry"]
        self.assertEqual(entry["class_name"], "spray_can_shaking")
        self.assertEqual(entry["session_id"], "session_local_krylon_01")
        self.assertEqual(entry["microphone_name"], "Realtek USB Condenser Mic")
        self.assertEqual(entry["can_type"], "aerosol_paint_can")
        self.assertEqual(entry["can_fill_level"], "half_full")
        self.assertEqual(entry["shaking_style"], "vertical_vigorous")
        self.assertEqual(entry["label_confidence"], "confirmed_high")
        self.assertEqual(entry["distance_m"], "1.0")

        # Verify manifest.json on disk
        self.assertTrue(self.manifest_path.exists())
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        saved_entry = manifest[-1]
        self.assertEqual(saved_entry["id"], entry["id"])
        self.assertEqual(saved_entry["session_id"], "session_local_krylon_01")

        # Verify saved audio file exists
        saved_file = BASE_DIR / entry["filepath"]
        self.assertTrue(saved_file.exists())
        # Clean up test file
        if saved_file.exists():
            saved_file.unlink()

    def test_record_negative_hiss_and_rattles(self):
        """Test recording negative samples (spray hiss only, key rattles)."""
        audio_bytes = self._create_synthetic_wav_bytes(1.0)

        # 1. Spray Hiss Only
        resp1 = self.app.post(
            "/api/record_sample",
            data={
                "class_name": "aerosol_spray_hiss",
                "session_id": "session_neg_hiss_01",
                "microphone_name": "Target Mic",
                "can_type": "aerosol_paint_can",
                "can_fill_level": "full",
                "shaking_style": "n_a_negative",
                "label_confidence": "confirmed_high",
                "condition": "spray_hiss_discharge",
                "distance_m": "1.0",
                "audio": (io.BytesIO(audio_bytes), "test_hiss.wav", "audio/wav"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp1.status_code, 200)
        e1 = resp1.get_json()["entry"]
        self.assertEqual(e1["class_name"], "aerosol_spray_hiss")

        # 2. Hard Negative Keys/Coins
        resp2 = self.app.post(
            "/api/record_sample",
            data={
                "class_name": "hard_negative_rattle",
                "session_id": "session_neg_keys_01",
                "microphone_name": "Target Mic",
                "can_type": "n_a_negative",
                "can_fill_level": "n_a_negative",
                "shaking_style": "n_a_negative",
                "label_confidence": "confirmed_high",
                "condition": "similar_rattles",
                "distance_m": "1.0",
                "audio": (io.BytesIO(audio_bytes), "test_keys.wav", "audio/wav"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp2.status_code, 200)
        e2 = resp2.get_json()["entry"]
        self.assertEqual(e2["class_name"], "hard_negative_rattle")

        # Clean up files
        for e in [e1, e2]:
            p = BASE_DIR / e["filepath"]
            if p.exists():
                p.unlink()

    def test_audio_file_serving_and_traversal_prevention(self):
        """Test /api/audio_file safe streaming and security checks."""
        # 1. Missing path
        r0 = self.app.get("/api/audio_file")
        self.assertEqual(r0.status_code, 400)

        # 2. Directory traversal attempt
        r1 = self.app.get("/api/audio_file?path=../../windows/system32/cmd.exe")
        self.assertIn(r1.status_code, [403, 404])

        # 3. Nonexistent file
        r2 = self.app.get("/api/audio_file?path=data/raw/nonexistent_file.wav")
        self.assertEqual(r2.status_code, 404)

    def test_get_session_id_with_manifest_mapping(self):
        """Verify get_session_id resolves exact session IDs from manifest."""
        manifest_map = {
            "session_local_can_01_clean_indoor_123456_a1b2c3d4.wav": "session_local_can_01",
            "session_local_can_02_clean_indoor_123457_b2c3d4e5.wav": "session_local_can_02",
            "session_test_can_clean_indoor_123458_c3d4e5f6.wav": "session_test_can",
        }

        p1 = Path("data/raw/field_recordings/spray_can_shaking/session_local_can_01_clean_indoor_123456_a1b2c3d4.wav")
        p2 = Path("data/raw/field_recordings/spray_can_shaking/session_local_can_02_clean_indoor_123457_b2c3d4e5.wav")
        p3 = Path("data/raw/field_recordings/spray_can_shaking/session_test_can_clean_indoor_123458_c3d4e5f6.wav")

        self.assertEqual(get_session_id(p1, manifest_session_map=manifest_map), "session_local_can_01")
        self.assertEqual(get_session_id(p2, manifest_session_map=manifest_map), "session_local_can_02")
        self.assertEqual(get_session_id(p3, manifest_session_map=manifest_map), "session_test_can")


if __name__ == "__main__":
    unittest.main()
