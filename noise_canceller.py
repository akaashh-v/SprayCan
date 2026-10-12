"""Real-Time Adaptive Spectral & Harmonic-Percussive Noise Cancellation for Spray Can Detector.

Implements high-precision multi-band noise cancellation and transient isolation:
1. Subband Stationary & Babble Tracking:
   - Low band (< 2200 Hz): Suppresses speech vowel formants, crowded chatter/babble, HVAC hum, and room rumble.
   - High band (>= 2200 Hz): Preserves metallic canister acoustic modal resonances and agitator ball impact peaks.
2. Fast 2D Harmonic-Percussive Transient Separation (HPS):
   - Separates sustained tonal/harmonic sounds (speech vowels, drone, hum) from vertical impulsive impact strikes.
   - Ensures spray can shaking is detected even in crowded rooms or when overlapping voices/noises are louder than shaking.
3. Over-subtraction with soft spectral floor bounding to prevent musical chirps.
4. Provides real-time SNR (dB), Noise Reduction (dB), and Noise Floor telemetry.
"""

from typing import Dict, Tuple
import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter

from config import (
    AUDIO_SAMPLES,
    SAMPLE_RATE,
)


class AdaptiveSpectralNoiseCanceller:
    """High-precision real-time multi-band spectral & transient noise cancellation engine."""

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        n_fft: int = 512,
        hop_length: int = 128,
        alpha: float = None,
        beta: float = None,
        alpha_low: float = 2.4,        # Strong over-subtraction for low-frequency speech/crowd band
        alpha_high: float = 1.3,       # Gentle over-subtraction for high-frequency metallic impact band
        beta_low: float = 0.04,        # Spectral floor for low band
        beta_high: float = 0.08,       # Spectral floor for high band
        noise_percentile_low: float = 35.0,   # Quantile for speech babble / room tone tracking
        noise_percentile_high: float = 20.0,  # Quantile for quiet inter-impact valleys
        **kwargs,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noverlap = n_fft - hop_length  # 384
        self.alpha_low = float(alpha * 1.4) if alpha is not None else alpha_low
        self.alpha_high = float(alpha * 0.8) if alpha is not None else alpha_high
        self.beta_low = float(beta * 0.6) if beta is not None else beta_low
        self.beta_high = float(beta * 1.3) if beta is not None else beta_high
        self.noise_percentile_low = noise_percentile_low
        self.noise_percentile_high = noise_percentile_high
        self.window = "hann"

        # Frequency partition
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        self.low_mask = freqs < 2200
        self.high_mask = freqs >= 2200

        # Running estimate of background noise spectrum across streaming frames
        self.running_noise_spec: np.ndarray = None
        self.history_alpha: float = 0.85

    def reset(self) -> None:
        """Reset running noise floor estimate."""
        self.running_noise_spec = None

    def cancel_noise(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply adaptive multi-band spectral subtraction and transient enhancement.

        Returns:
            cleaned_audio: Noise-suppressed audio array of identical length.
            metrics: Telemetry containing SNR (dB), noise reduction (dB), and noise floor RMS.
        """
        orig_len = len(audio)
        if orig_len < self.n_fft:
            return audio.copy(), {
                "snr_db": 0.0,
                "noise_reduction_db": 0.0,
                "noise_floor_rms": 0.0,
                "is_active": False,
            }

        orig_rms = float(np.sqrt(np.mean(audio**2)))
        if orig_rms < 1e-5:
            return audio.copy(), {
                "snr_db": 0.0,
                "noise_reduction_db": 0.0,
                "noise_floor_rms": 0.0,
                "is_active": True,
            }

        # 1. Short-Time Fourier Transform (STFT) via scipy.signal
        _, _, Zxx = signal.stft(
            audio,
            fs=self.sr,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.noverlap,
            boundary="zeros",
            padded=True,
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # 2. Multi-Band Stationary & Babble Noise Floor Estimation
        noise_floor = np.zeros((mag.shape[0], 1), dtype=np.float32)
        if np.any(self.low_mask):
            noise_floor[self.low_mask] = np.percentile(mag[self.low_mask], self.noise_percentile_low, axis=1, keepdims=True)
        if np.any(self.high_mask):
            noise_floor[self.high_mask] = np.percentile(mag[self.high_mask], self.noise_percentile_high, axis=1, keepdims=True)

        if self.running_noise_spec is None or self.running_noise_spec.shape != noise_floor.shape:
            self.running_noise_spec = noise_floor
        else:
            self.running_noise_spec = (
                self.history_alpha * self.running_noise_spec + (1.0 - self.history_alpha) * noise_floor
            )

        smoothed_noise = self.running_noise_spec

        # 3. Multi-Band Over-Subtraction Gain Mask
        gain = np.ones_like(mag)
        snr_ratio = mag / (smoothed_noise + 1e-8)

        if np.any(self.low_mask):
            gain[self.low_mask] = np.maximum(
                1.0 - self.alpha_low / (snr_ratio[self.low_mask] + 1e-8), self.beta_low
            )
        if np.any(self.high_mask):
            gain[self.high_mask] = np.maximum(
                1.0 - self.alpha_high / (snr_ratio[self.high_mask] + 1e-8), self.beta_high
            )

        # 4. Fast 2D Harmonic-Percussive Transient Separation (HPS)
        # Suppress sustained horizontal harmonics (speech vowels, hums)
        # and isolate vertical broadband metallic collision transients
        H = median_filter(mag, size=(1, 11))
        P = median_filter(mag, size=(11, 1))
        t_weight = (P ** 2) / (P ** 2 + H ** 2 + 1e-6)

        # Combined magnitude reconstruction
        enhanced_mag = mag * gain * (0.25 + 0.75 * t_weight)

        # 5. Inverse STFT
        clean_Zxx = enhanced_mag * np.exp(1j * phase)
        _, cleaned_audio = signal.istft(
            clean_Zxx,
            fs=self.sr,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.noverlap,
            boundary="zeros",
        )
        cleaned_audio = cleaned_audio[:orig_len].astype(np.float32)

        # 6. Compute Telemetry Metrics
        clean_rms = float(np.sqrt(np.mean(cleaned_audio**2)))
        noise_diff = audio - cleaned_audio
        est_noise_rms = float(np.sqrt(np.mean(noise_diff**2)))

        noise_reduction_db = float(20.0 * np.log10(max(1e-5, orig_rms) / max(1e-5, clean_rms)))
        snr_db = float(20.0 * np.log10(max(1e-5, clean_rms) / max(1e-5, est_noise_rms)))

        return cleaned_audio, {
            "snr_db": round(max(0.0, snr_db), 1),
            "noise_reduction_db": round(max(0.0, noise_reduction_db), 1),
            "noise_floor_rms": round(est_noise_rms, 5),
            "orig_rms": round(orig_rms, 5),
            "clean_rms": round(clean_rms, 5),
            "is_active": True,
        }
