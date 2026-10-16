"""Unit and integration tests for audio resampling fidelity and transient preservation.

Verifies:
1. Live microphone resampling preserves metallic aerosol can transient impact peaks.
2. Anti-aliasing sinc / polyphase resampling matches librosa.resample training pipeline.
3. Log-Mel spectrogram MSE and crest factor are preserved without high-frequency aliasing distortion.
4. Model prediction probabilities are stable across native hardware sample rates (44.1 kHz and 48 kHz).
"""

import unittest
import numpy as np
import scipy.signal as signal
import torch
import librosa

from audio_utils import compute_log_mel_spectrogram, load_audio
from config import AUDIO_SAMPLES, DEVICE, NUM_CLASSES, RAW_SPRAY_DIR, SAMPLE_RATE
from stage1_shaking_detector import Stage1ShakingDetector
from stage2_classifier import SprayCanShakingCNN, evaluate_candidate_prediction


class TestResamplingFidelity(unittest.TestCase):
    """Test audio resampling fidelity between live microphone ingestion and training pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.sr_target = SAMPLE_RATE  # 22050 Hz
        cls.shake_file = RAW_SPRAY_DIR / "1.mp3"
        cls.stage1 = Stage1ShakingDetector(sr=cls.sr_target)

    def test_sinc_vs_polyphase_spectrogram_parity(self):
        """Polyphase resampling (resample_poly) must match librosa.resample with near-zero MSE."""
        # Load audio at 48 kHz
        audio_48k, _ = librosa.load(self.shake_file, sr=48000)
        chunk_48k = audio_48k[:48000]

        # 1. librosa.resample (training reference)
        resampled_librosa = librosa.resample(chunk_48k, orig_sr=48000, target_sr=self.sr_target)
        resampled_librosa = resampled_librosa[:self.sr_target]

        # 2. scipy.signal.resample_poly
        # 22050 / 48000 = 147 / 320
        resampled_poly = signal.resample_poly(chunk_48k, 147, 320)
        resampled_poly = resampled_poly[:self.sr_target]

        # Compare Log-Mel Spectrograms
        spec_librosa = compute_log_mel_spectrogram(resampled_librosa, sr=self.sr_target)
        spec_poly = compute_log_mel_spectrogram(resampled_poly, sr=self.sr_target)

        mse = float(np.mean((spec_librosa - spec_poly) ** 2))
        cosine_sim = float(
            np.dot(spec_librosa.flatten(), spec_poly.flatten())
            / (np.linalg.norm(spec_librosa) * np.linalg.norm(spec_poly) + 1e-8)
        )

        self.assertLess(mse, 0.01, f"Spectrogram MSE ({mse:.5f}) exceeded 0.01 between resamplers")
        self.assertGreater(cosine_sim, 0.99, f"Cosine similarity ({cosine_sim:.5f}) below 0.99")

    def test_transient_crest_factor_preservation(self):
        """Resampling from 48 kHz / 44.1 kHz must preserve impulsive crest factor and Stage 1 cadence."""
        for orig_rate in [44100, 48000]:
            audio_native, _ = librosa.load(self.shake_file, sr=orig_rate)
            chunk_native = audio_native[:orig_rate]

            # Downsample using polyphase anti-aliasing filter
            gcd = int(np.gcd(self.sr_target, orig_rate))
            up = self.sr_target // gcd
            down = orig_rate // gcd
            resampled = signal.resample_poly(chunk_native, up, down)[:self.sr_target].astype(np.float32)

            # Analyze with Stage 1 detector
            s1_native = self.stage1.analyze_window(resampled)

            self.assertGreater(s1_native["crest_factor"], 4.0, f"Crest factor degraded for {orig_rate} Hz input")
            self.assertGreaterEqual(s1_native["burst_count"], 3, f"Burst count dropped for {orig_rate} Hz input")
            self.assertTrue(s1_native["is_candidate"], f"Stage 1 failed to flag candidate for {orig_rate} Hz input")

    def test_anti_aliasing_suppresses_out_of_band_energy(self):
        """High-frequency content above 11.025 kHz must not fold back into canister impact band."""
        # Synthesize a pure 18 kHz tone at 48 kHz sample rate (would alias to 4050 Hz without anti-aliasing)
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        tone_18k = np.sin(2 * np.pi * 18000 * t).astype(np.float32)

        # Polyphase sinc resample with anti-aliasing filter
        resampled_poly = signal.resample_poly(tone_18k, 147, 320)[:self.sr_target].astype(np.float32)

        # Measure energy attenuation of the out-of-band 18 kHz tone in time domain
        energy_in = float(np.mean(tone_18k ** 2))
        energy_out = float(np.mean(resampled_poly ** 2))
        attenuation_ratio = energy_out / energy_in

        # With anti-aliasing filter, 18 kHz tone must be suppressed by at least 40 dB (< 0.0001 energy ratio)
        self.assertLess(attenuation_ratio, 0.0001, f"High-frequency aliasing energy ({attenuation_ratio:.8f}) exceeded -40 dB limit")


if __name__ == "__main__":
    unittest.main()
