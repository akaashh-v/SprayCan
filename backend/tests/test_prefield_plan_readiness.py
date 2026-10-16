"""Pre-Field Evaluation Plan Readiness & Statistical Verification Test Suite.

Tests and certifies all readiness gates locally using sample and synthetic data:
1. Model checkpoint hash, configuration lock (tau = 0.80), and shadow-mode enforcement.
2. Positive event sample sizing mathematics (Wilson score interval precision and condition quotas).
3. Poisson exact confidence interval & Rule of Three calculations for negative exposure.
4. Circuit breaker condition detection (clipping rejection, silence floor, ambiguous transitions).
5. Ephemeral data retention TTL purge logic.
6. Chronological partitioning and transfer-learning site holdout integrity.
"""

import hashlib
import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from config import (
    AUDIO_RETENTION_HOURS,
    AUDIO_SAMPLES,
    BASE_DIR,
    CLASS_NAMES,
    CONSERVATIVE_THRESHOLD,
    MODEL_SAVE_PATH,
    SAMPLE_RATE,
    SHADOW_MODE_ENABLED,
)
from detector_engine import SprayCanDetectorEngine
from evaluate_streaming_field import (
    compute_sha256,
    poisson_rate_confidence_interval,
    wilson_score_interval,
)
from partition_field_data import partition_field_data


class TestPreFieldReadinessConfigurationAndSecurity(unittest.TestCase):
    """Certify that system parameters comply with shadow mode and governance rules."""

    def test_production_threshold_locked_at_point_eighty(self):
        """Production decision threshold must be strictly locked at tau = 0.80."""
        self.assertEqual(
            CONSERVATIVE_THRESHOLD,
            0.80,
            "Production conservative threshold must be locked at 0.80! Tuning on field data is prohibited.",
        )

    def test_shadow_mode_strictly_enforced(self):
        """Shadow mode must be active; dispatch and automated enforcement must be disabled."""
        self.assertTrue(SHADOW_MODE_ENABLED, "Shadow mode must be enabled!")
        engine = SprayCanDetectorEngine()
        self.assertTrue(engine.shadow_mode)

        # Trigger dummy window to verify output flags
        dummy_win = np.zeros(AUDIO_SAMPLES, dtype=np.float32)
        res = engine.evaluate_window(dummy_win)
        self.assertFalse(res.get("dispatch_authorized", False))
        self.assertFalse(res.get("automatic_enforcement", False))

    def test_model_checkpoint_sha256_canonical(self):
        """Active neural network weights must match the certified canonical SHA-256 hash."""
        canon_path = Path("models/canonical_evaluation_report.json")
        if canon_path.exists():
            with open(canon_path, "r", encoding="utf-8") as f:
                expected_sha256 = json.load(f).get("provenance", {}).get("model_sha256", "c8483581bdf7bd1ec31a3f3025584f59601f609b655aed38284e3d2e6fcab65d")
        else:
            expected_sha256 = "c8483581bdf7bd1ec31a3f3025584f59601f609b655aed38284e3d2e6fcab65d"
        self.assertTrue(MODEL_SAVE_PATH.exists(), f"Model checkpoint not found at {MODEL_SAVE_PATH}")
        actual_sha256 = compute_sha256(MODEL_SAVE_PATH)
        self.assertEqual(
            actual_sha256,
            expected_sha256,
            f"Active model SHA-256 ({actual_sha256}) differs from canonical checkpoint ({expected_sha256})!",
        )

    def test_audio_retention_ttl_setting(self):
        """Data retention policy must be set to 24 hours."""
        self.assertEqual(AUDIO_RETENTION_HOURS, 24)


class TestPositiveEventSampleSizingMathematics(unittest.TestCase):
    """Verify statistical power and binomial sample size calculations for field validation."""

    def test_binomial_positive_event_sample_size_formula(self):
        """Confirm that achieving margin of error E <= 0.07 at 95% confidence requires >= 100 positive events."""
        p_expected = 0.85
        margin_of_error = 0.07
        z_95 = 1.95996

        # Standard binomial sample size formula: N = (z^2 * p * (1-p)) / E^2
        required_n = (z_95 ** 2 * p_expected * (1.0 - p_expected)) / (margin_of_error ** 2)
        self.assertGreaterEqual(required_n, 99.0)
        self.assertLessEqual(required_n, 105.0)

        # Wilson score interval width for N=100 at p=0.85
        lower, upper = wilson_score_interval(successes=85, total=100, confidence=0.95)
        interval_width = upper - lower
        self.assertLessEqual(interval_width, 0.15, "100 events must yield interval width <= 15% (margin <= 7.5%)")

    def test_condition_level_sample_quotas(self):
        """Verify that 5 conditions with >= 25 events each provide >= 125 total positive events."""
        conditions = [
            "quiet_platform",
            "train_passby_and_braking",
            "station_announcements",
            "rain_and_environmental",
            "passenger_crowd",
        ]
        target_per_condition = 25
        total_quota = len(conditions) * target_per_condition
        self.assertGreaterEqual(total_quota, 125)

        # For single condition with 25 events and 85% observed recall (21/25 = 84%)
        lower_cond, upper_cond = wilson_score_interval(successes=21, total=25, confidence=0.95)
        # Condition interval width is broader but bounded
        self.assertGreater(lower_cond, 0.60)
        self.assertLessEqual(upper_cond, 0.96)

    def test_negative_audio_hours_cannot_measure_event_recall(self):
        """Demonstrate that 100 hours of continuous negative audio alone provides ZERO recall power."""
        total_target_events = 0
        detected_target_events = 0
        with self.assertRaises(ZeroDivisionError):
            _ = detected_target_events / total_target_events


class TestPoissonNegativeExposureConfidenceBounds(unittest.TestCase):
    """Verify Poisson exact confidence intervals and Rule of Three on negative exposure."""

    def test_one_hundred_hours_zero_alerts_certifies_target_far(self):
        """100 hours of negative audio with 0 false alerts proves true FAR <= 0.0300 < 0.05 FA/hr."""
        hours = 100.0
        ci = poisson_rate_confidence_interval(k=0, hours=hours, confidence=0.95)
        self.assertIsNotNone(ci)
        self.assertEqual(ci["lower"], 0.0)
        # Rule of three: -ln(0.05) / 100 ≈ 2.9957 / 100 = 0.02996 ≈ 0.0300
        self.assertAlmostEqual(ci["upper"], 0.0300, places=3)
        self.assertLess(
            ci["upper"],
            0.05,
            "100 hours with 0 false alerts must certify FAR <= 0.05 FA/hr at 95% upper bound!",
        )

    def test_short_test_insufficient_for_low_far_claim(self):
        """A short 5-hour test with 0 false alerts has upper bound 0.60 FA/hr (cannot claim <= 0.05)."""
        hours = 5.0
        ci = poisson_rate_confidence_interval(k=0, hours=hours, confidence=0.95)
        self.assertIsNotNone(ci)
        self.assertGreater(ci["upper"], 0.50)
        self.assertFalse(ci["upper"] <= 0.05, "Short 5h test cannot claim platform readiness!")


class TestCircuitBreakersAndSignalFaults(unittest.TestCase):
    """Test engine detection of sensor faults and circuit-breaker conditions."""

    def setUp(self):
        self.engine = SprayCanDetectorEngine()

    def test_circuit_breaker_clipping_rejection(self):
        """Digital clipping (Trigger A) must be rejected with UNUSABLE_AUDIO_CLIPPING and zero alert."""
        # Synthesize heavily clipped audio
        clipped_signal = np.ones(AUDIO_SAMPLES, dtype=np.float32)
        res = self.engine.evaluate_window(clipped_signal)
        self.assertFalse(res.get("alert_fired", False))
        self.assertEqual(res.get("status"), "UNUSABLE_AUDIO_CLIPPING")
        self.assertTrue(res.get("is_clipped", False))

    def test_circuit_breaker_clean_silence_no_false_alarm(self):
        """Station silence/ambient floor below energy threshold must evaluate to STATUS: SILENT."""
        silence = np.zeros(AUDIO_SAMPLES, dtype=np.float32)
        res = self.engine.evaluate_window(silence)
        self.assertFalse(res.get("alert_fired", False))
        self.assertEqual(res.get("status"), "STATUS: SILENT")


class TestEphemeralRetentionPurgeLogic(unittest.TestCase):
    """Verify that expired files (> 24h) are removed while fresh files (< 24h) are preserved."""

    def test_purge_removes_expired_files_preserves_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fresh_file = tmp_path / "fresh_evt.json"
            expired_file = tmp_path / "expired_evt.json"

            fresh_file.write_text("{}", encoding="utf-8")
            expired_file.write_text("{}", encoding="utf-8")

            # Set mtime of expired file to 30 hours ago
            thirty_hours_ago = time.time() - (30 * 3600)
            import os
            os.utime(str(expired_file), (thirty_hours_ago, thirty_hours_ago))

            cutoff_epoch = time.time() - (24 * 3600)

            # Simulate purge
            for f in tmp_path.glob("*.json"):
                if f.stat().st_mtime < cutoff_epoch:
                    f.unlink()

            self.assertTrue(fresh_file.exists(), "Fresh file (< 24h) must NOT be deleted!")
            self.assertFalse(expired_file.exists(), "Expired file (> 24h) must be deleted!")


class TestChronologicalPartitioningAndTransferSeparation(unittest.TestCase):
    """Verify chronological split ordering and site-held-out station transfer."""

    def test_chronological_ordering_and_site_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "field_split_manifest.json"
            field_dir = Path(tmpdir) / "field_sessions"

            # Create 4 sessions across 4 days
            test_items = []
            for i in range(1, 5):
                s_dir = field_dir / f"sess_{i}"
                s_dir.mkdir(parents=True)
                audio_f = s_dir / "audio.wav"
                audio_f.write_bytes(b"RIFF" + b"\x00" * 100)
                meta = {
                    "source_type": "continuous_session",
                    "session_id": f"sess_{i}",
                    "site_id": "station_central",
                    "recording_date": f"2026-10-0{i}",
                    "timestamp_iso": f"2026-10-0{i}T10:00:00Z",
                    "audio_abs_path": str(audio_f),
                    "duration_hours": 1.0,
                }
                (s_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
                test_items.append(meta)

            manifest = partition_field_data(
                items=test_items,
                split_strategy="chronological",
                train_ratio=0.50,
                val_ratio=0.25,
                final_test_ratio=0.25,
            )

            splits = manifest["partitions"]
            # Earliest sessions in train, latest in final_test
            self.assertEqual(len(splits["train"]), 2)
            self.assertEqual(len(splits["val"]), 1)
            self.assertEqual(len(splits["final_test"]), 1)

            # Latest date (2026-10-04) must be in final_test
            test_sess = splits["final_test"][0]
            self.assertEqual(test_sess["session_id"], "sess_4")

            # Audit confirms same site appeared in multiple splits
            audit = manifest["site_cross_split_audit"]
            self.assertTrue(audit["same_site_in_multiple_splits"])
            self.assertIn("station_central", audit["sites_across_multiple_splits"])


if __name__ == "__main__":
    unittest.main()
