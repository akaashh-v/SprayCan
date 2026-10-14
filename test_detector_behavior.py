"""Targeted Behavioral and Unit Test Suite for Spray-Can Shaking Detector.

Proves the required behavior:
1. Continuous aerosol spray hiss alone NEVER alerts.
2. Pulsed spray hiss bursts ("pssht... pssht...") NEVER alert.
3. Loud impacts (door slam / desk bang / high peak energy) NEVER alert.
4. Isolated transient rattles (keys / coins < 0.7s) NEVER alert.
5. Ambiguous audio / white noise / digital clipping NEVER alert (safe no-alert / uncertain).
6. Reset / clear rule: brief hiss or transient immediately resets accumulation to 0.
7. Cooldown rule: 3.0s cooldown strictly prevents repeated alerts for one episode.
8. Authentic sustained spray can shaking sequence DOES alert.
"""

from pathlib import Path
import unittest
import numpy as np

from audio_utils import (
    generate_ambient_room_noise,
    generate_env_rain,
    generate_env_water_drops,
    generate_hard_negative_coins,
    generate_hard_negative_keys,
    generate_impact_door_slam,
    generate_spray_continuous_hiss,
    generate_spray_pulsed_bursts,
    load_audio,
)
from config import (
    ALERT_COOLDOWN_SECONDS,
    AUDIO_SAMPLES,
    RAW_SPRAY_DIR,
    SAMPLE_RATE,
)
from detector_engine import SprayCanDetectorEngine


class TestSprayCanDetectorBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SprayCanDetectorEngine()
        cls.engine.reload_model_if_available()
        cls.sr = SAMPLE_RATE

    def setUp(self):
        self.engine.reset_state()

    def test_continuous_spray_hiss_never_alerts(self):
        """Continuous aerosol spray hiss alone must never trigger an alert."""
        hiss_audio = generate_spray_continuous_hiss(duration=3.0, sr=self.sr)
        hop = int(self.sr * 0.25)
        alerts_count = 0
        hiss_verdicts = []

        for start in range(0, len(hiss_audio) - AUDIO_SAMPLES + 1, hop):
            window = hiss_audio[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window)
            if res["alert_fired"]:
                alerts_count += 1
            hiss_verdicts.append(res["status"])

        self.assertEqual(alerts_count, 0, f"Continuous hiss triggered {alerts_count} false alerts!")
        for v in hiss_verdicts:
            self.assertNotEqual(v, "ALERT: SPRAY CAN SOUND CONFIRMED")

    def test_pulsed_spray_bursts_never_alerts(self):
        """Pulsed aerosol spray bursts ('pssht... pssht...') must never trigger an alert."""
        pulsed_audio = generate_spray_pulsed_bursts(duration=3.0, sr=self.sr)
        hop = int(self.sr * 0.25)
        alerts_count = 0

        for start in range(0, len(pulsed_audio) - AUDIO_SAMPLES + 1, hop):
            window = pulsed_audio[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window)
            if res["alert_fired"]:
                alerts_count += 1

        self.assertEqual(alerts_count, 0, f"Pulsed spray hiss triggered {alerts_count} false alerts!")

    def test_loud_impact_never_alerts(self):
        """A sudden loud impact (door slam / desk bang) must NEVER trigger an alert."""
        impact_audio = generate_impact_door_slam(duration=2.0, sr=self.sr)
        hop = int(self.sr * 0.25)
        alerts_count = 0

        for start in range(0, len(impact_audio) - AUDIO_SAMPLES + 1, hop):
            window = impact_audio[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window)
            if res["alert_fired"]:
                alerts_count += 1

        self.assertEqual(alerts_count, 0, "Loud impact triggered a false alert! Loudness must not alert.")

    def test_isolated_transient_rattle_never_alerts(self):
        """An isolated transient rattle (coins or keys < 0.6s) must not trigger an alert."""
        # Short rattle followed by silence
        short_rattle = generate_hard_negative_coins(duration=0.5, sr=self.sr)
        silence = np.zeros(int(self.sr * 2.0), dtype=np.float32)
        audio = np.concatenate([short_rattle, silence])

        hop = int(self.sr * 0.25)
        alerts_count = 0

        for start in range(0, len(audio) - AUDIO_SAMPLES + 1, hop):
            window = audio[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window)
            if res["alert_fired"]:
                alerts_count += 1

        self.assertEqual(alerts_count, 0, "Isolated transient rattle triggered a false alert!")

    def test_ambiguous_and_poor_quality_audio_never_alerts(self):
        """Ambiguous white noise or digital clipping must output safe no-alert or clipping."""
        # Case 1: Digital clipping (clipped audio)
        clipped_noise = np.random.uniform(-1.5, 1.5, AUDIO_SAMPLES).astype(np.float32)
        res_clip = self.engine.evaluate_window(clipped_noise)
        self.assertFalse(res_clip["alert_fired"])
        self.assertEqual(res_clip["status"], "UNUSABLE_AUDIO_CLIPPING")

        # Case 2: Ambiguous unstructured white noise
        white_noise = np.random.normal(0, 0.05, AUDIO_SAMPLES).astype(np.float32)
        res_noise = self.engine.evaluate_window(white_noise)
        self.assertFalse(res_noise["alert_fired"])
        self.assertIn(
            res_noise["status"],
            [
                "NO_SHAKING_CADENCE",
                "UNCERTAIN_AMBIGUOUS_SOUND",
                "CONTINUOUS_ENVIRONMENTAL_REJECTED",
                "AEROSOL_SPRAY_HISS_REJECTED",
                "UNCERTAIN_LOW_MARGIN",
            ]
        )

    def test_reset_clear_rule_prevents_accumulation(self):
        """Interleaving brief hiss or negative sounds immediately resets accumulated score to 0."""
        # Verified shaking window and hiss window
        shake_file = RAW_SPRAY_DIR / "1.mp3"
        raw_shake = load_audio(shake_file, sr=self.sr)
        shake_win = raw_shake[int(self.sr * 1.0) : int(self.sr * 2.0)]
        hiss_win = generate_spray_continuous_hiss(duration=1.0, sr=self.sr)

        # Evaluate shaking window
        res1 = self.engine.evaluate_window(shake_win, current_time=1.0)
        self.assertFalse(res1["alert_fired"])

        # Next evaluate hiss window -> MUST reset consecutive count immediately to 0
        res2 = self.engine.evaluate_window(hiss_win, current_time=2.0)
        self.assertFalse(res2["alert_fired"])
        self.assertEqual(self.engine.consecutive_positive_windows, 0, "Hiss failed to reset persistence counter!")

        # Evaluate shaking window again -> must restart from 1, NOT 2
        res3 = self.engine.evaluate_window(shake_win, current_time=3.0)
        self.assertFalse(res3["alert_fired"])
        self.assertLessEqual(self.engine.consecutive_positive_windows, 1)

    def test_cooldown_enforcement(self):
        """After a confirmed event fires, subsequent windows within cooldown are suppressed."""
        # Simulate firing an alert
        self.engine.last_alert_time = 10.0

        # Evaluate at t = 10.25s (0.25s into cooldown)
        shake_file = RAW_SPRAY_DIR / "1.mp3"
        raw_shake = load_audio(shake_file, sr=self.sr)
        shake_win = raw_shake[:AUDIO_SAMPLES]
        res = self.engine.evaluate_window(shake_win, current_time=10.25)

        self.assertFalse(res["alert_fired"])
        self.assertTrue(res["in_cooldown"])
        self.assertEqual(res["status"], "IN_COOLDOWN")
        expected_remaining = round(self.engine.cooldown_seconds - 0.25, 2)
        self.assertAlmostEqual(res["cooldown_remaining_sec"], expected_remaining, places=2)

        # After cooldown passes (t = 10.0 + cooldown + 0.1s)
        t_after = 10.0 + self.engine.cooldown_seconds + 0.1
        res_post = self.engine.evaluate_window(shake_win, current_time=t_after)
        self.assertFalse(res_post["in_cooldown"])

    def test_light_rain_never_alerts(self):
        """Continuous light rain audio across multiple windows must NEVER trigger an alert."""
        rain_audio = generate_env_rain(duration=3.5, sr=self.sr)
        hop = int(self.sr * 0.25)
        alerts_count = 0
        for start in range(0, len(rain_audio) - AUDIO_SAMPLES + 1, hop):
            window = rain_audio[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window)
            if res["alert_fired"]:
                alerts_count += 1
            self.assertNotEqual(res["status"], "ALERT: SPRAY CAN SOUND CONFIRMED")

        self.assertEqual(alerts_count, 0, f"Light rain triggered {alerts_count} false alerts!")

    def test_water_drops_falling_never_alerts(self):
        """Water drops falling / plinking into a sink across multiple windows must NEVER trigger an alert."""
        wd_audio = generate_env_water_drops(duration=3.5, sr=self.sr)
        hop = int(self.sr * 0.25)
        alerts_count = 0
        for start in range(0, len(wd_audio) - AUDIO_SAMPLES + 1, hop):
            window = wd_audio[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window)
            if res["alert_fired"]:
                alerts_count += 1
            self.assertNotEqual(res["status"], "ALERT: SPRAY CAN SOUND CONFIRMED")

        self.assertEqual(alerts_count, 0, f"Water drops falling triggered {alerts_count} false alerts!")

    def test_raw_rain_files_never_alert(self):
        """Real raw rain audio files from continuous_environmental dataset must NEVER trigger an alert."""
        rain_files = sorted(list(Path("data/raw/continuous_environmental").glob("rain_*.wav")))[:10]
        hop = int(self.sr * 0.25)
        alerts_count = 0
        for rf in rain_files:
            self.engine.reset_state()
            audio = load_audio(rf, sr=self.sr)
            for start in range(0, len(audio) - AUDIO_SAMPLES + 1, hop):
                window = audio[start : start + AUDIO_SAMPLES]
                res = self.engine.evaluate_window(window)
                if res["alert_fired"]:
                    alerts_count += 1
                self.assertNotEqual(res["status"], "ALERT: SPRAY CAN SOUND CONFIRMED")

        self.assertEqual(alerts_count, 0, f"Raw rain files triggered {alerts_count} false alerts!")

    def test_raw_water_drops_files_never_alert(self):
        """Real raw water drop audio files from continuous_environmental dataset must NEVER trigger an alert."""
        wd_files = sorted(list(Path("data/raw/continuous_environmental").glob("water_drops_*.wav")))[:10]
        hop = int(self.sr * 0.25)
        alerts_count = 0
        for wf in wd_files:
            self.engine.reset_state()
            audio = load_audio(wf, sr=self.sr)
            for start in range(0, len(audio) - AUDIO_SAMPLES + 1, hop):
                window = audio[start : start + AUDIO_SAMPLES]
                res = self.engine.evaluate_window(window)
                if res["alert_fired"]:
                    alerts_count += 1
                self.assertNotEqual(res["status"], "ALERT: SPRAY CAN SOUND CONFIRMED")

        self.assertEqual(alerts_count, 0, f"Raw water drop files triggered {alerts_count} false alerts!")

    def test_authentic_shaking_sequence_alerts(self):
        """An authentic sustained spray can shaking sequence across multiple windows triggers alert."""
        shake_file = RAW_SPRAY_DIR / "1.mp3"
        sustained_audio = load_audio(shake_file, sr=self.sr)
        hop = int(self.sr * 0.25)
        alert_fired = False

        for start in range(0, min(len(sustained_audio) - AUDIO_SAMPLES + 1, int(self.sr * 4.0)), hop):
            window = sustained_audio[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window, current_time=start / self.sr)
            if res["alert_fired"]:
                alert_fired = True
                break

        self.assertTrue(alert_fired, "Authentic spray can shaking sequence failed to trigger alert!")

    def test_noise_cancellation_suppresses_continuous_background_and_measures_snr(self):
        """Noise cancellation must suppress stationary noise floor and output valid SNR telemetry."""
        # 3 seconds of ambient room tone + fan noise
        ambient_noise = generate_ambient_room_noise(duration=3.0, sr=self.sr)
        self.engine.enable_noise_cancellation = True

        window = ambient_noise[:AUDIO_SAMPLES]
        res = self.engine.evaluate_window(window)

        self.assertTrue(res["noise_cancellation_active"])
        self.assertIn("snr_db", res)
        self.assertIn("noise_reduction_db", res)
        self.assertGreaterEqual(res["noise_reduction_db"], 0.0)
        self.assertFalse(res["alert_fired"])
        self.assertNotEqual(res["status"], "ALERT: SPRAY CAN SOUND CONFIRMED")

    def test_noise_cancellation_preserves_authentic_spray_detection(self):
        """Authentic spray can shaking mixed with background room noise must still be confirmed with noise cancellation active."""
        shake_file = RAW_SPRAY_DIR / "1.mp3"
        shake_audio = load_audio(shake_file, sr=self.sr)
        room_noise = generate_ambient_room_noise(duration=len(shake_audio) / self.sr, sr=self.sr) * 0.3

        mixed_audio = shake_audio + room_noise[:len(shake_audio)]
        self.engine.enable_noise_cancellation = True
        self.engine.reset_state()

        hop = int(self.sr * 0.25)
        alert_fired = False

        for start in range(0, min(len(mixed_audio) - AUDIO_SAMPLES + 1, int(self.sr * 4.0)), hop):
            window = mixed_audio[start : start + AUDIO_SAMPLES]
            res = self.engine.evaluate_window(window, current_time=start / self.sr)
            if res["alert_fired"]:
                alert_fired = True
                break

        self.assertTrue(alert_fired, "Authentic spray mixed with room noise should still alert under noise cancellation!")


if __name__ == "__main__":
    unittest.main()
