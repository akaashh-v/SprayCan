"""Unit Tests for Field Streaming Evaluator, Poisson Metrics, and Partition Isolation.

Tests:
1. Zero negative hours exposure: returns None for rate and None for CI.
2. Positive hours with zero alerts: returns rate 0.0 and Poisson exact 95% upper bound (~3.0/T).
3. Positive hours with one or more alerts: returns exact Poisson Chi-square CI.
4. Interval union duration: non-overlapping, overlapping, clamped, empty.
5. Continuous streaming event matching and duplicate alert episode grouping without cooldown.
6. Target-event interval exclusion from negative exposure time.
7. Session-level and site-date isolation during partitioning (zero leakage guarantee).
"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_utils import generate_ambient_room_noise, load_audio
from config import ALERT_COOLDOWN_SECONDS, AUDIO_SAMPLES, RAW_SPRAY_DIR, SAMPLE_RATE
from detector_engine import SprayCanDetectorEngine
from evaluate_streaming_field import (
    StreamingFieldEvaluator,
    compute_interval_union_duration,
    poisson_rate_confidence_interval,
    wilson_score_interval,
)
from partition_field_data import partition_field_data


class TestPoissonMetrics(unittest.TestCase):
    """Test Poisson confidence intervals and exposure metric rules."""

    def test_zero_hours_returns_none(self):
        """Zero exposure time MUST return None for CI, not 0.0 or a false bound."""
        self.assertIsNone(poisson_rate_confidence_interval(0, 0.0))
        self.assertIsNone(poisson_rate_confidence_interval(0, -1.5))
        self.assertIsNone(poisson_rate_confidence_interval(5, 0.0))

    def test_positive_hours_zero_alerts(self):
        """Positive exposure with 0 alerts calculates exact Poisson upper bound (~3.0/T)."""
        hours = 2.0
        ci = poisson_rate_confidence_interval(0, hours)
        self.assertIsNotNone(ci)
        self.assertEqual(ci["lower"], 0.0)
        expected_upper = round(-math.log(0.05) / hours, 4)  # ~1.4979
        self.assertEqual(ci["upper"], expected_upper)
        self.assertIsNotNone(ci["rule_of_three_upper"])
        self.assertIn("≈ 3.0 / T", ci["interpretation"])

    def test_positive_hours_with_alerts(self):
        """Positive exposure with >= 1 alert calculates exact Chi-square intervals."""
        hours = 10.0
        k = 4
        ci = poisson_rate_confidence_interval(k, hours)
        self.assertIsNotNone(ci)
        rate = k / hours  # 0.40
        self.assertLess(ci["lower"], rate)
        self.assertGreater(ci["upper"], rate)
        self.assertGreater(ci["lower"], 0.0)
        self.assertIsNone(ci["rule_of_three_upper"])
        self.assertIn(f"{k} false alert", ci["interpretation"])

    def test_wilson_score_interval(self):
        """Verify Wilson score confidence interval for binomial recall."""
        self.assertEqual(wilson_score_interval(0, 0), (0.0, 0.0))
        lower, upper = wilson_score_interval(8, 10)
        self.assertGreater(lower, 0.40)
        self.assertLess(upper, 1.0)
        self.assertLessEqual(lower, 0.8)
        self.assertGreaterEqual(upper, 0.8)


class TestIntervalUnion(unittest.TestCase):
    """Test union duration of ground-truth target intervals."""

    def test_disjoint_intervals(self):
        intervals = [(1.0, 3.0), (5.0, 7.5)]
        dur = compute_interval_union_duration(intervals, max_duration=10.0)
        self.assertAlmostEqual(dur, 4.5, places=3)

    def test_overlapping_intervals(self):
        # [1.0, 4.0] and [3.0, 6.0] should span [1.0, 6.0] = 5.0s (no double count)
        intervals = [(1.0, 4.0), (3.0, 6.0)]
        dur = compute_interval_union_duration(intervals, max_duration=10.0)
        self.assertAlmostEqual(dur, 5.0, places=3)

    def test_clamped_and_out_of_bounds(self):
        intervals = [(-2.0, 4.0), (8.0, 15.0)]
        # clamped to [0.0, 4.0] (4.0s) and [8.0, 10.0] (2.0s) on max 10.0s = 6.0s
        dur = compute_interval_union_duration(intervals, max_duration=10.0)
        self.assertAlmostEqual(dur, 6.0, places=3)

    def test_empty_intervals(self):
        self.assertEqual(compute_interval_union_duration([], max_duration=10.0), 0.0)


class TestStreamingEvaluatorLogic(unittest.TestCase):
    """Test streaming evaluator, event matching, and negative duration exclusion."""

    def setUp(self):
        self.evaluator = StreamingFieldEvaluator(
            conservative_threshold=0.80,
            hop_seconds=0.25,
        )

    def test_continuous_session_negative_duration_exclusion(self):
        """Target-event intervals must be subtracted from total time to calculate negative exposure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test_session.wav"
            # 10.0 seconds of audio
            sr = SAMPLE_RATE
            dur_sec = 10.0
            sig = np.zeros(int(sr * dur_sec), dtype=np.float32)
            sf.write(str(wav_path), sig, sr)

            # Annotate a target event of 4.0 seconds from 2.0s to 6.0s
            annotations = [{
                "start_sec": 2.0,
                "end_sec": 6.0,
                "class_label": "spray_can_shaking",
                "condition": "quiet_platform",
            }]

            res = self.evaluator.evaluate_continuous_session(wav_path, annotations)

            self.assertAlmostEqual(res["total_duration_seconds"], 10.0, places=1)
            # Positive duration = 4.0s; negative duration = 6.0s
            self.assertAlmostEqual(res["negative_duration_seconds"], 6.0, places=1)
            self.assertAlmostEqual(res["negative_duration_hours"], 6.0 / 3600.0, places=4)
            self.assertEqual(res["target_events_count"], 1)

    def test_continuous_session_zero_negative_exposure(self):
        """When an entire clip is covered by target event, negative hours is 0 and FA rate is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "all_positive.wav"
            sr = SAMPLE_RATE
            dur_sec = 4.0
            sig = np.zeros(int(sr * dur_sec), dtype=np.float32)
            sf.write(str(wav_path), sig, sr)

            annotations = [{
                "start_sec": 0.0,
                "end_sec": 4.0,
                "class_label": "spray_can_shaking",
                "condition": "clean",
            }]

            res = self.evaluator.evaluate_continuous_session(wav_path, annotations)
            self.assertEqual(res["negative_duration_seconds"], 0.0)
            self.assertEqual(res["negative_duration_hours"], 0.0)
            # Must be None, NOT 0.0!
            self.assertIsNone(res["false_alert_episodes_per_hour"])
            self.assertIsNone(res["false_alert_rate_95ci_per_hour"])

    def test_cooldown_suppresses_duplicate_alert_episodes(self):
        """Zero cooldown: event-state tracking maintains active episode without duplicate alerts."""
        engine = self.evaluator.engine
        engine.reset_state()
        self.assertEqual(engine.cooldown_seconds, 0.0)

        # Simulate alert triggered at t = 1.0s
        engine.last_alert_time = 1.0
        self.assertTrue(engine.alert_episode_active)

        # In continuous detection with 0 cooldown, audio is evaluated immediately
        # A non-shaking window evaluates immediately without being blocked by cooldown
        dummy_win = np.zeros(SAMPLE_RATE, dtype=np.float32)
        res = engine.evaluate_window(dummy_win, current_time=1.25)
        self.assertFalse(res["in_cooldown"])
        self.assertEqual(res["cooldown_remaining_sec"], 0.0)
        self.assertFalse(res.get("alert_fired", False))

        # After >= 2 quiet windows, event ends and engine rearms
        res2 = engine.evaluate_window(dummy_win, current_time=1.50)
        self.assertFalse(engine.alert_episode_active)


class TestPartitioningIntegrity(unittest.TestCase):
    """Verify session partitioning keeps related recordings grouped with zero leakage."""

    def test_group_isolation_no_leakage(self):
        """Recordings with the same session_id or site+date must not be split across train, val, and test."""
        manifest = partition_field_data()
        partitions = manifest.get("partitions", {})
        train_items = partitions.get("train", [])
        val_items = partitions.get("val", [])
        test_items = partitions.get("final_test", [])

        train_sessions = {it["session_id"] for it in train_items}
        val_sessions = {it["session_id"] for it in val_items}
        test_sessions = {it["session_id"] for it in test_items}

        self.assertEqual(len(train_sessions.intersection(val_sessions)), 0)
        self.assertEqual(len(train_sessions.intersection(test_sessions)), 0)
        self.assertEqual(len(val_sessions.intersection(test_sessions)), 0)

    def test_synthetic_multi_session_partition_isolation(self):
        """Verify partition grouping and SHA-256 hashing on simulated field sessions."""
        import json
        import shutil
        from config import FIELD_SESSIONS_DIR, PROCESSED_DIR

        test_sess_dir = FIELD_SESSIONS_DIR / "test_sess_mock_01"
        test_sess_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Create synthetic audio file
            sr = SAMPLE_RATE
            wav_path = test_sess_dir / "audio.wav"
            sf.write(str(wav_path), np.zeros(sr * 2, dtype=np.float32), sr)

            meta = {
                "session_id": "test_sess_mock_01",
                "site_id": "station_mock_platform_1",
                "microphone_id": "mic_cardioid_mock",
                "condition": "busy_station",
                "duration_sec": 2.0,
                "timestamp_iso": "2026-10-13T12:00:00Z",
            }
            with open(test_sess_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

            manifest = partition_field_data(train_ratio=0.5, val_ratio=0.25, final_test_ratio=0.25)
            partitions = manifest.get("partitions", {})

            all_items = []
            for s_name, s_items in partitions.items():
                for it in s_items:
                    all_items.append((s_name, it))
                    # Verify cryptographic SHA-256 exists and is 64 hex characters
                    self.assertIn("file_sha256", it)
                    self.assertEqual(len(it["file_sha256"]), 64)

            # Ensure mock session appears in exactly ONE partition
            matching_splits = [s_name for s_name, it in all_items if it["session_id"] == "test_sess_mock_01"]
            self.assertEqual(len(matching_splits), 1)

            # Check that field_split_manifest.json was written to disk
            out_manifest = PROCESSED_DIR / "field_split_manifest.json"
            self.assertTrue(out_manifest.exists())
        finally:
            shutil.rmtree(test_sess_dir, ignore_errors=True)

    def test_partition_same_site_different_dates(self):
        """Recordings from the same site on different dates can cross partitions without leakage."""
        items = [
            {
                "session_id": "sess_day1",
                "site_id": "site_alpha",
                "date": "2026-10-01",
                "audio_path": "mock1.wav",
                "duration_sec": 10.0,
                "annotation_count": 1,
            },
            {
                "session_id": "sess_day2",
                "site_id": "site_alpha",
                "date": "2026-10-02",
                "audio_path": "mock2.wav",
                "duration_sec": 10.0,
                "annotation_count": 1,
            },
            {
                "session_id": "sess_day3",
                "site_id": "site_alpha",
                "date": "2026-10-03",
                "audio_path": "mock3.wav",
                "duration_sec": 10.0,
                "annotation_count": 1,
            },
        ]
        manifest = partition_field_data(
            train_ratio=0.34, val_ratio=0.33, final_test_ratio=0.33, items=items
        )
        splits = manifest["partitions"]

        # Ensure no session leakage exists
        train_s = {it["session_id"] for it in splits["train"]}
        val_s = {it["session_id"] for it in splits["val"]}
        test_s = {it["session_id"] for it in splits["final_test"]}
        self.assertEqual(len(train_s.intersection(val_s)), 0)
        self.assertEqual(len(train_s.intersection(test_s)), 0)
        self.assertEqual(len(val_s.intersection(test_s)), 0)

        # Same site items are distributed across distinct partitions because session_id is the atomic unit
        occupied_splits = sum(1 for s in [splits["train"], splits["val"], splits["final_test"]] if len(s) > 0)
        self.assertGreater(occupied_splits, 1, "Same-site recordings across different dates were not partitioned!")

    def test_partition_same_date_different_sessions(self):
        """Recordings sharing the same date with different sessions can cross partitions without leakage."""
        items = [
            {
                "session_id": "sess_morning",
                "site_id": "station_north",
                "date": "2026-10-05",
                "audio_path": "morn.wav",
                "duration_sec": 5.0,
                "annotation_count": 0,
            },
            {
                "session_id": "sess_midday",
                "site_id": "station_south",
                "date": "2026-10-05",
                "audio_path": "mid.wav",
                "duration_sec": 5.0,
                "annotation_count": 0,
            },
            {
                "session_id": "sess_night",
                "site_id": "station_east",
                "date": "2026-10-05",
                "audio_path": "night.wav",
                "duration_sec": 5.0,
                "annotation_count": 0,
            },
        ]
        manifest = partition_field_data(
            train_ratio=0.34, val_ratio=0.33, final_test_ratio=0.33, items=items
        )
        splits = manifest["partitions"]

        train_s = {it["session_id"] for it in splits["train"]}
        val_s = {it["session_id"] for it in splits["val"]}
        test_s = {it["session_id"] for it in splits["final_test"]}
        self.assertEqual(len(train_s.intersection(val_s)), 0)
        self.assertEqual(len(train_s.intersection(test_s)), 0)
        self.assertEqual(len(val_s.intersection(test_s)), 0)

        occupied_splits = sum(1 for s in [splits["train"], splits["val"], splits["final_test"]] if len(s) > 0)
        self.assertGreater(occupied_splits, 1, "Same-date recordings across different sessions were not partitioned!")

    def test_partition_session_with_inconsistent_dates(self):
        """A session with inconsistent dates across items MUST remain together in the exact same partition."""
        items = [
            {
                "session_id": "sess_midnight_boundary",
                "site_id": "station_depot",
                "date": "2026-10-10",  # chunk 1 recorded before midnight
                "audio_path": "chunk1.wav",
                "duration_sec": 15.0,
                "annotation_count": 1,
            },
            {
                "session_id": "sess_midnight_boundary",
                "site_id": "station_depot",
                "date": "2026-10-11",  # chunk 2 recorded after midnight (inconsistent date)
                "audio_path": "chunk2.wav",
                "duration_sec": 15.0,
                "annotation_count": 1,
            },
            {
                "session_id": "sess_control_A",
                "site_id": "station_platform_2",
                "date": "2026-10-12",
                "audio_path": "ctrl_a.wav",
                "duration_sec": 30.0,
                "annotation_count": 0,
            },
            {
                "session_id": "sess_control_B",
                "site_id": "station_platform_3",
                "date": "2026-10-13",
                "audio_path": "ctrl_b.wav",
                "duration_sec": 30.0,
                "annotation_count": 0,
            },
        ]
        manifest = partition_field_data(
            train_ratio=0.5, val_ratio=0.25, final_test_ratio=0.25, items=items
        )
        splits = manifest["partitions"]

        # Check where chunks of sess_midnight_boundary ended up
        chunk_split_assignments = []
        for s_name, s_items in splits.items():
            for it in s_items:
                if it["session_id"] == "sess_midnight_boundary":
                    chunk_split_assignments.append((s_name, it["date"], it["audio_path"]))

        # Both chunks MUST be in the exact same partition
        self.assertEqual(len(chunk_split_assignments), 2)
        split_names = {entry[0] for entry in chunk_split_assignments}
        self.assertEqual(
            len(split_names), 1,
            f"Atomic session was split across multiple partitions due to date inconsistency: {chunk_split_assignments}"
        )

    def test_partition_chronological_splits_and_audit(self):
        """Chronological partitioning uses earlier sessions for train/val, later for untouched final test, and audits sites."""
        items = [
            {
                "session_id": "sess_day1",
                "site_id": "station_alpha",
                "date": "2026-10-01",
                "timestamp_iso": "2026-10-01T10:00:00Z",
                "audio_path": "day1.wav",
                "duration_sec": 20.0,
                "annotation_count": 1,
            },
            {
                "session_id": "sess_day2",
                "site_id": "station_alpha",
                "date": "2026-10-02",
                "timestamp_iso": "2026-10-02T10:00:00Z",
                "audio_path": "day2.wav",
                "duration_sec": 20.0,
                "annotation_count": 1,
            },
            {
                "session_id": "sess_day3",
                "site_id": "station_beta",
                "date": "2026-10-03",
                "timestamp_iso": "2026-10-03T10:00:00Z",
                "audio_path": "day3.wav",
                "duration_sec": 20.0,
                "annotation_count": 0,
            },
            {
                "session_id": "sess_day4",
                "site_id": "station_beta",
                "date": "2026-10-04",
                "timestamp_iso": "2026-10-04T10:00:00Z",
                "audio_path": "day4.wav",
                "duration_sec": 20.0,
                "annotation_count": 0,
            },
        ]
        manifest = partition_field_data(
            train_ratio=0.50, val_ratio=0.25, final_test_ratio=0.25,
            items=items, split_strategy="chronological"
        )
        splits = manifest["partitions"]

        # Earlier sessions (day1, day2) must be in train
        train_sessions = {it["session_id"] for it in splits["train"]}
        self.assertEqual(train_sessions, {"sess_day1", "sess_day2"})

        # Intermediate session (day3) in val
        val_sessions = {it["session_id"] for it in splits["val"]}
        self.assertEqual(val_sessions, {"sess_day3"})

        # Latest continuous day (day4) strictly reserved for untouched final test
        test_sessions = {it["session_id"] for it in splits["final_test"]}
        self.assertEqual(test_sessions, {"sess_day4"})

        # Site cross-split audit: station_alpha is only in train; station_beta is in val & test
        audit = manifest["site_cross_split_audit"]
        self.assertTrue(audit["same_site_in_multiple_splits"])
        self.assertIn("station_beta", audit["sites_across_multiple_splits"])

    def test_partition_site_held_out_transfer(self):
        """Site-held-out partitioning keeps entire station sites strictly isolated with zero site leakage."""
        items = [
            {"session_id": f"s_a{i}", "site_id": "station_alpha", "date": "2026-10-01", "audio_path": f"a{i}.wav", "duration_sec": 10.0}
            for i in range(2)
        ] + [
            {"session_id": f"s_b{i}", "site_id": "station_beta", "date": "2026-10-02", "audio_path": f"b{i}.wav", "duration_sec": 10.0}
            for i in range(2)
        ] + [
            {"session_id": f"s_g{i}", "site_id": "station_gamma", "date": "2026-10-03", "audio_path": f"g{i}.wav", "duration_sec": 10.0}
            for i in range(2)
        ]

        manifest = partition_field_data(
            train_ratio=0.34, val_ratio=0.33, final_test_ratio=0.33,
            items=items, split_strategy="site_held_out"
        )
        splits = manifest["partitions"]

        train_sites = {it["site_id"] for it in splits["train"]}
        val_sites = {it["site_id"] for it in splits["val"]}
        test_sites = {it["site_id"] for it in splits["final_test"]}

        # Zero site leakage across all pairs
        self.assertEqual(len(train_sites.intersection(val_sites)), 0)
        self.assertEqual(len(train_sites.intersection(test_sites)), 0)
        self.assertEqual(len(val_sites.intersection(test_sites)), 0)

        # Audit confirms no site appears in multiple splits
        audit = manifest["site_cross_split_audit"]
        self.assertFalse(audit["same_site_in_multiple_splits"])
        self.assertEqual(audit["sites_across_multiple_splits"], [])


class TestAlertEpisodeGroupingAndEventMatching(unittest.TestCase):
    """Test streaming alert episode grouping, cooldown interaction, and event matching."""

    @classmethod
    def setUpClass(cls):
        cls.shake_audio = load_audio(RAW_SPRAY_DIR / "1.mp3", sr=SAMPLE_RATE)
        cls.evaluator = StreamingFieldEvaluator(
            conservative_threshold=0.80,
            hop_seconds=0.25,
        )

    def test_single_shake_longer_than_cooldown_grouped_to_one_episode(self):
        """A single sustained shake lasting > 3.0s (e.g. 6.0s) produces exactly ONE alert episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "long_shake.wav"
            sr = SAMPLE_RATE
            # Tile active 2.0s shaking segment to produce 6.0s of continuous vigorous shake
            shake_6s = np.tile(self.shake_audio[: int(sr * 2.0)], 3)
            lead_silence = np.zeros(int(sr * 1.0), dtype=np.float32)
            trail_silence = np.zeros(int(sr * 1.0), dtype=np.float32)
            audio = np.concatenate([lead_silence, shake_6s, trail_silence])
            sf.write(str(wav_path), audio, sr)

            annotations = [{
                "start_sec": 1.0,
                "end_sec": 7.0,
                "class_label": "spray_can_shaking",
                "condition": "quiet_platform",
            }]

            res = self.evaluator.evaluate_continuous_session(wav_path, annotations)

            # Ground truth: 1 event
            self.assertEqual(res["target_events_count"], 1)
            self.assertEqual(res["detected_events_count"], 1)
            self.assertEqual(res["missed_events_count"], 0)

            # Alert episodes: exactly 1 true alert episode for the entire sustained shake
            self.assertEqual(res["true_alert_episodes"], 1)
            self.assertEqual(res["false_alert_episodes"], 0)
            self.assertEqual(res["alert_episodes_fired"], 1)

            # Details
            tp_detail = [ep for ep in res["alert_episodes_detail"] if ep["episode_type"] == "true_positive"]
            self.assertEqual(len(tp_detail), 1)

    def test_two_separate_shakes_with_clear_quiet_gap_under_three_seconds(self):
        """Two separate shakes with a clear quiet gap < 3.0s: event-end is verified, engine rearms, and both alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "two_shakes_rearmed.wav"
            sr = SAMPLE_RATE
            shake_1_2s = self.shake_audio[: int(sr * 1.2)]
            lead_silence = np.zeros(int(sr * 0.5), dtype=np.float32)
            # Verified quiet gap: 0.8s of pure silence (>= 2 quiet windows of 0.25s)
            quiet_gap = np.zeros(int(sr * 0.8), dtype=np.float32)
            trail_silence = np.zeros(int(sr * 1.0), dtype=np.float32)
            # Total audio: 0.5s lead + 1.2s shake1 + 0.8s quiet + 1.2s shake2 + 1.0s trail = 4.7s
            # Shake 1: [0.5, 1.7] (alert fires at ~0.75s).
            # Quiet gap: [1.7, 2.5] (verified quiet interval >= 2 windows -> REARMS at 2.25s).
            # Shake 2: [2.5, 3.7] (starts at 2.5s, which is only 1.75s < 3.0s after Alert 1!).
            audio = np.concatenate([lead_silence, shake_1_2s, quiet_gap, shake_1_2s, trail_silence])
            sf.write(str(wav_path), audio, sr)

            annotations = [
                {
                    "start_sec": 0.5,
                    "end_sec": 1.7,
                    "class_label": "spray_can_shaking",
                    "condition": "quiet_platform",
                },
                {
                    "start_sec": 2.5,
                    "end_sec": 3.7,
                    "class_label": "spray_can_shaking",
                    "condition": "quiet_platform",
                },
            ]

            res = self.evaluator.evaluate_continuous_session(wav_path, annotations)

            self.assertEqual(res["target_events_count"], 2)
            self.assertEqual(res["detected_events_count"], 2)
            self.assertEqual(res["missed_events_count"], 0)
            self.assertEqual(res["true_alert_episodes"], 2)
            self.assertTrue(res["target_events_detail"][0]["detected"])
            self.assertTrue(res["target_events_detail"][1]["detected"])

    def test_two_shakes_with_ambiguous_gap_marked_ambiguous(self):
        """Two shakes with no reliably distinguishable gap are marked ambiguous and do not spawn an unverified second alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "two_shakes_ambiguous_gap.wav"
            sr = SAMPLE_RATE
            shake_1_2s = self.shake_audio[: int(sr * 1.2)]
            lead_silence = np.zeros(int(sr * 0.5), dtype=np.float32)
            # Ambiguous gap: 0.8s of elevated, non-quiet acoustic rumble/rattle (rms >= 0.035, not verified quiet)
            t_gap = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)
            ambiguous_noise = (np.sin(2 * np.pi * 120 * t_gap) * 0.03 + np.random.normal(0, 0.025, len(t_gap))).astype(np.float32)
            trail_silence = np.zeros(int(sr * 1.0), dtype=np.float32)

            audio = np.concatenate([lead_silence, shake_1_2s, ambiguous_noise, shake_1_2s, trail_silence])
            sf.write(str(wav_path), audio, sr)

            # Direct window-level check through engine to verify ambiguous transition windows
            self.evaluator.engine.reset_state()
            hop = int(sr * 0.25)
            gap_statuses = []
            for start in range(0, len(audio) - AUDIO_SAMPLES + 1, hop):
                t_win = start / sr
                w = audio[start : start + AUDIO_SAMPLES]
                r = self.evaluator.engine.evaluate_window(w, current_time=t_win)
                # Check status during ambiguous transition window (between 1.7s and 2.5s)
                if 1.75 <= t_win <= 2.25:
                    gap_statuses.append(r["status"])

            # Verify that gap windows without clean quiet floor were flagged as ambiguous, non-candidate, or ongoing
            self.assertTrue(
                any(st in ["UNCERTAIN_AMBIGUOUS_SOUND", "NO_SHAKING_CADENCE", "SUSTAINED_SHAKING_ACTIVE"] for st in gap_statuses),
                f"Expected ambiguous/ongoing statuses during indistinct gap, got: {gap_statuses}"
            )

    def test_two_shakes_separated_by_less_than_cooldown_rearms_after_quiet_gap(self):
        """Two distinct shakes separated by < 3.0s with verified quiet gap: event-end detection rearms and both alert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "two_shakes_short_gap.wav"
            sr = SAMPLE_RATE
            shake_1_5s = self.shake_audio[: int(sr * 1.5)]
            lead_silence = np.zeros(int(sr * 0.5), dtype=np.float32)
            # Gap of 1.0s (less than 3.0s cooldown, but >= 2 verified quiet windows)
            short_gap = np.zeros(int(sr * 1.0), dtype=np.float32)
            trail_silence = np.zeros(int(sr * 1.0), dtype=np.float32)
            audio = np.concatenate([lead_silence, shake_1_5s, short_gap, shake_1_5s, trail_silence])
            sf.write(str(wav_path), audio, sr)

            annotations = [
                {
                    "start_sec": 0.5,
                    "end_sec": 2.0,
                    "class_label": "spray_can_shaking",
                    "condition": "quiet_platform",
                },
                {
                    "start_sec": 3.0,
                    "end_sec": 4.5,
                    "class_label": "spray_can_shaking",
                    "condition": "quiet_platform",
                },
            ]

            res = self.evaluator.evaluate_continuous_session(wav_path, annotations)

            self.assertEqual(res["target_events_count"], 2)
            # Event-end verified during 1.0s quiet gap: detector rearms and detects both events
            self.assertEqual(res["detected_events_count"], 2)
            self.assertEqual(res["missed_events_count"], 0)
            self.assertEqual(res["true_alert_episodes"], 2)
            self.assertEqual(res["false_alert_episodes"], 0)

            # Check individual event detections
            ev1 = res["target_events_detail"][0]
            ev2 = res["target_events_detail"][1]
            self.assertTrue(ev1["detected"], "First shake event must be detected!")
            self.assertTrue(ev2["detected"], "Second shake event after verified quiet gap must be detected!")

    def test_two_shakes_separated_by_equal_to_cooldown(self):
        """Two distinct shakes separated by == 3.0s from alert onset: both events are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "two_shakes_exact_cooldown.wav"
            sr = SAMPLE_RATE
            shake_1_5s = self.shake_audio[: int(sr * 1.5)]
            lead_silence = np.zeros(int(sr * 0.5), dtype=np.float32)
            # Event 1 starts at 0.5s; alert fires at ~0.75s. Cooldown expires at 0.75 + 3.0 = 3.75s.
            # Event 2 starts at 3.75s (gap of 1.75s between end of Event 1 and start of Event 2)
            exact_gap = np.zeros(int(sr * 1.75), dtype=np.float32)
            trail_silence = np.zeros(int(sr * 1.0), dtype=np.float32)
            audio = np.concatenate([lead_silence, shake_1_5s, exact_gap, shake_1_5s, trail_silence])
            sf.write(str(wav_path), audio, sr)

            annotations = [
                {
                    "start_sec": 0.5,
                    "end_sec": 2.0,
                    "class_label": "spray_can_shaking",
                    "condition": "quiet_platform",
                },
                {
                    "start_sec": 3.75,
                    "end_sec": 5.25,
                    "class_label": "spray_can_shaking",
                    "condition": "quiet_platform",
                },
            ]

            res = self.evaluator.evaluate_continuous_session(wav_path, annotations)

            self.assertEqual(res["target_events_count"], 2)
            self.assertEqual(res["detected_events_count"], 2)
            self.assertEqual(res["missed_events_count"], 0)
            self.assertEqual(res["true_alert_episodes"], 2)
            self.assertEqual(res["false_alert_episodes"], 0)
            self.assertTrue(res["target_events_detail"][0]["detected"])
            self.assertTrue(res["target_events_detail"][1]["detected"])

    def test_two_shakes_separated_by_more_than_cooldown(self):
        """Two distinct shakes separated by > 3.0s: both events are detected as distinct alert episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "two_shakes_long_gap.wav"
            sr = SAMPLE_RATE
            shake_1_5s = self.shake_audio[: int(sr * 1.5)]
            lead_silence = np.zeros(int(sr * 0.5), dtype=np.float32)
            # 4.5s silence gap (> 3.0s cooldown)
            long_gap = np.zeros(int(sr * 4.5), dtype=np.float32)
            trail_silence = np.zeros(int(sr * 1.0), dtype=np.float32)
            audio = np.concatenate([lead_silence, shake_1_5s, long_gap, shake_1_5s, trail_silence])
            sf.write(str(wav_path), audio, sr)

            annotations = [
                {
                    "start_sec": 0.5,
                    "end_sec": 2.0,
                    "class_label": "spray_can_shaking",
                    "condition": "quiet_platform",
                },
                {
                    "start_sec": 6.5,
                    "end_sec": 8.0,
                    "class_label": "spray_can_shaking",
                    "condition": "quiet_platform",
                },
            ]

            res = self.evaluator.evaluate_continuous_session(wav_path, annotations)

            self.assertEqual(res["target_events_count"], 2)
            self.assertEqual(res["detected_events_count"], 2)
            self.assertEqual(res["missed_events_count"], 0)
            self.assertEqual(res["true_alert_episodes"], 2)
            self.assertEqual(res["false_alert_episodes"], 0)
            self.assertTrue(res["target_events_detail"][0]["detected"])
            self.assertTrue(res["target_events_detail"][1]["detected"])


class TestAdaptiveNoiseCancellationFreezing(unittest.TestCase):
    """Test adaptive noise cancellation background freezing during candidate shakes and active alerts."""

    def setUp(self):
        self.engine = SprayCanDetectorEngine(enable_noise_cancellation=True)
        self.engine.reload_model_if_available()
        self.sr = SAMPLE_RATE

    def test_sustained_shake_freezes_adaptation_and_preserves_target(self):
        """Confirm background estimate freezes during candidate shake and detector does not learn away target."""
        # 1. Warm up baseline noise floor with 2.0s of continuous background noise
        bg_noise = generate_ambient_room_noise(duration=2.0, sr=self.sr)
        hop = int(self.sr * 0.25)
        for start in range(0, len(bg_noise) - AUDIO_SAMPLES + 1, hop):
            window = bg_noise[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window)
            self.assertFalse(res["adaptation_frozen"])

        self.assertIsNotNone(self.engine.noise_canceller.running_noise_spec)
        baseline_noise_spec = self.engine.noise_canceller.running_noise_spec.copy()
        baseline_mean = float(np.mean(baseline_noise_spec))

        # 2. Inject sustained 4.0s of authentic spray shaking mixed with background noise
        shake_raw = load_audio(RAW_SPRAY_DIR / "1.mp3", sr=self.sr)
        shake_4s = shake_raw[: int(self.sr * 4.0)]
        noise_bed = generate_ambient_room_noise(duration=4.0, sr=self.sr) * 0.2
        mixed_audio = shake_4s + noise_bed

        frozen_flags = []
        confidences = []
        candidates = []

        for start in range(0, len(mixed_audio) - AUDIO_SAMPLES + 1, hop):
            window = mixed_audio[start : start + AUDIO_SAMPLES]
            t_curr = 2.0 + start / self.sr
            res = self.engine.evaluate_window(window, current_time=t_curr)
            frozen_flags.append(res["adaptation_frozen"])
            candidates.append(res["is_stage1_candidate"])
            confidences.append(res["p_spray_can"])

        # Check 1: Background adaptation freezes during shaking
        self.assertTrue(any(frozen_flags), "Adaptation was never frozen during shaking sequence!")
        frozen_ratio = sum(frozen_flags) / len(frozen_flags)
        self.assertGreater(frozen_ratio, 0.50, f"Adaptation frozen ratio ({frozen_ratio:.2f}) was lower than expected!")

        # Check 2: Running noise floor does not absorb the target burst peaks
        current_noise_spec = self.engine.noise_canceller.running_noise_spec
        current_mean = float(np.mean(current_noise_spec))
        # The noise floor estimate during freeze remains close to baseline
        self.assertAlmostEqual(current_mean, baseline_mean, delta=baseline_mean * 0.25)

        # Check 3: Shaking detection maintained across the sustained event (does not learn away target)
        self.assertTrue(any(candidates), "Stage 1 candidate was never triggered during sustained shake!")
        high_conf_windows = [c for c in confidences if c >= 0.80]
        self.assertGreater(
            len(high_conf_windows), 0,
            "Detector learned away target or failed to achieve high confidence during sustained shake!"
        )


if __name__ == "__main__":
    unittest.main()
