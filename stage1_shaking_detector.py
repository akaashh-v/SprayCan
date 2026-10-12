from typing import Dict, Tuple
import numpy as np
import scipy.signal as signal

from config import (
    CLIPPING_PEAK_THRESHOLD,
    CLIPPING_RATIO_THRESHOLD,
    SAMPLE_RATE,
    STAGE1_MAX_IMPACT_RATE,
    STAGE1_MAX_INTERVAL_CV,
    STAGE1_MIN_BURSTS,
    STAGE1_MIN_CANISTER_SPECTRAL_RATIO,
    STAGE1_MIN_CREST_FACTOR,
    STAGE1_MIN_IMPACT_RATE,
    STAGE1_MIN_RMS,
    STAGE1_MIN_SPAN_SECONDS,
    STAGE1_MIN_VALLEY_CONTRAST_DB,
)


class Stage1ShakingDetector:
    """Fast kinematic pre-filter for acoustic back-and-forth shaking burst candidate detection."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        # 5ms smoothing window for envelope extraction
        self.env_kernel_size = int(sr * 0.005)
        self.env_kernel = np.hanning(self.env_kernel_size) / (self.env_kernel_size / 2)

        # Bandpass filter (1.8 kHz - 8.5 kHz) targeting the metal canister modal impact frequencies
        # Attenuates speech fundamentals, vowel formants, traffic, and room rumbles below 1.8 kHz
        nyq = sr / 2.0
        self.impact_sos = signal.butter(
            2,
            [1800.0 / nyq, min(8500.0, nyq - 100.0) / nyq],
            btype="bandpass",
            output="sos",
        )

    def analyze_window(self, audio: np.ndarray) -> Dict:
        """Evaluate if an audio window possesses kinematic back-and-forth shaking properties."""
        if len(audio) == 0:
            return {
                "is_candidate": False,
                "is_clipped": False,
                "rms": 0.0,
                "peak": 0.0,
                "crest_factor": 0.0,
                "burst_count": 0,
                "cadence_hz": 0.0,
                "temporal_span_sec": 0.0,
                "valley_contrast_db": 0.0,
                "interval_cv": 0.0,
                "canister_spectral_ratio": 0.0,
                "reason": "Empty audio buffer",
            }

        peak_amp = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        crest_factor = float(peak_amp / (rms + 1e-8))

        # 1. Rule 1: Loudness may be used ONLY to identify clipping or unusable audio
        clip_ratio = float(np.mean(np.abs(audio) >= CLIPPING_PEAK_THRESHOLD))
        if clip_ratio >= CLIPPING_RATIO_THRESHOLD or peak_amp >= 1.05:
            return {
                "is_candidate": False,
                "is_clipped": True,
                "rms": round(rms, 4),
                "peak": round(peak_amp, 4),
                "crest_factor": round(crest_factor, 1),
                "burst_count": 0,
                "cadence_hz": 0.0,
                "temporal_span_sec": 0.0,
                "valley_contrast_db": 0.0,
                "interval_cv": 0.0,
                "canister_spectral_ratio": 0.0,
                "reason": f"Digital clipping / saturated unusable audio (Peak={peak_amp:.2f}, Clipped={clip_ratio*100:.1f}%)",
            }

        # 2. Silence / Below recording noise floor
        if rms < STAGE1_MIN_RMS:
            return {
                "is_candidate": False,
                "is_clipped": False,
                "rms": round(rms, 5),
                "peak": round(peak_amp, 4),
                "crest_factor": round(crest_factor, 1),
                "burst_count": 0,
                "cadence_hz": 0.0,
                "temporal_span_sec": 0.0,
                "valley_contrast_db": 0.0,
                "interval_cv": 0.0,
                "canister_spectral_ratio": 0.0,
                "reason": f"Signal below energy floor (RMS={rms:.4f} < {STAGE1_MIN_RMS})",
            }

        # 3. Extract Short-Time Amplitude Envelope on Canister Modal Impact Band (1.8 - 8.5 kHz)
        # Filters out low-frequency speech vowel formants and crowd babble (< 1.8 kHz)
        impact_audio = signal.sosfilt(self.impact_sos, audio)
        impact_rms = float(np.sqrt(np.mean(impact_audio**2)))
        impact_peak = float(np.max(np.abs(impact_audio))) if len(impact_audio) > 0 else 0.0
        impact_crest = float(impact_peak / (impact_rms + 1e-8))

        rectified = np.abs(impact_audio)
        envelope = signal.convolve(rectified, self.env_kernel, mode="same")

        # 4. Adaptive Threshold for Transient Point-Impact Peaks
        noise_floor = np.median(envelope) + 1e-6
        peak_threshold = max(0.0006, noise_floor * 1.35, impact_rms * 0.18)
        prominence_thresh = max(0.0003, noise_floor * 1.10, impact_rms * 0.10)

        # Min distance between peaks: 50 ms (handles impacts up to 20/sec; spray can cadence is 2.5-15 Hz)
        min_peak_distance = int(self.sr * 0.050)
        peaks, props = signal.find_peaks(
            envelope,
            height=peak_threshold,
            distance=min_peak_distance,
            prominence=prominence_thresh,
        )

        burst_count = len(peaks)
        duration_sec = len(audio) / self.sr
        cadence_hz = burst_count / duration_sec if duration_sec > 0 else 0.0

        # Check burst cadence constraints (>= 3 bursts in 1.0s, rate 2.5-15.0 Hz)
        has_cadence = (
            burst_count >= STAGE1_MIN_BURSTS and
            STAGE1_MIN_IMPACT_RATE <= cadence_hz <= STAGE1_MAX_IMPACT_RATE
        )

        # 5. Temporal Span Verification (Rules out isolated single/double impacts / drops)
        temporal_span_sec = 0.0
        if burst_count >= 2:
            first_peak_time = peaks[0] / self.sr
            last_peak_time = peaks[-1] / self.sr
            temporal_span_sec = float(last_peak_time - first_peak_time)
        has_span = temporal_span_sec >= STAGE1_MIN_SPAN_SECONDS

        # 6. Envelope Valley Contrast (Transit intervals between alternating back-and-forth strikes)
        # Continuous sounds (rain, hissing, fans) lack deep valleys between peaks
        valley_contrast_db = 0.0
        if burst_count >= 2:
            valleys = []
            for i in range(burst_count - 1):
                p1, p2 = peaks[i], peaks[i + 1]
                if p2 > p1:
                    valleys.append(np.min(envelope[p1:p2]))
            if valleys:
                med_peak = np.median(envelope[peaks])
                med_valley = np.median(valleys) + 1e-6
                valley_contrast_db = float(20.0 * np.log10(max(1e-5, med_peak) / med_valley))
        has_contrast = valley_contrast_db >= STAGE1_MIN_VALLEY_CONTRAST_DB

        # 7. Kinematic Interval Regularity (Rules out random Poisson rain drops & dripping water)
        interval_cv = 0.0
        has_regularity = True
        if burst_count >= 3:
            diffs = np.diff(peaks) / self.sr
            interval_cv = float(np.std(diffs) / (np.mean(diffs) + 1e-8))
            has_regularity = interval_cv <= STAGE1_MAX_INTERVAL_CV
        elif burst_count < STAGE1_MIN_BURSTS:
            has_regularity = False

        # 8. Canister Metallic Spectral Ratio (High canister resonance 2.2-7.5 kHz vs liquid bubbles 200-1800 Hz)
        canister_spectral_ratio = 0.0
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sr)
        e_low = np.sum(spec[(freqs >= 200) & (freqs < 1800)] ** 2) + 1e-8
        e_metal = np.sum(spec[(freqs >= 2200) & (freqs < 7500)] ** 2) + 1e-8
        canister_spectral_ratio = float(e_metal / e_low)
        has_canister_resonance = (canister_spectral_ratio >= STAGE1_MIN_CANISTER_SPECTRAL_RATIO) or (impact_rms >= 0.002 and has_cadence)

        # 9. Crest Impulsiveness Verification (Metallic marble impacts vs turbulent aerodynamic hiss)
        has_impulsiveness = (crest_factor >= STAGE1_MIN_CREST_FACTOR) or (impact_crest >= STAGE1_MIN_CREST_FACTOR)

        # Strict candidate decision: must match repeated back-and-forth physical shaking
        is_candidate = bool(
            has_cadence and
            has_span and
            has_contrast and
            has_regularity and
            has_canister_resonance and
            has_impulsiveness
        )

        if is_candidate:
            reason = (
                f"Candidate shaking sequence detected ({burst_count} bursts @ {cadence_hz:.1f} Hz, "
                f"Span={temporal_span_sec:.2f}s, Valley Contrast={valley_contrast_db:.1f} dB, "
                f"CV={interval_cv:.2f}, Metal/Low={canister_spectral_ratio:.2f}, Crest={crest_factor:.1f})"
            )
        elif not has_cadence:
            reason = f"No shaking cadence ({burst_count} bursts < {STAGE1_MIN_BURSTS} or rate={cadence_hz:.1f} Hz outside [{STAGE1_MIN_IMPACT_RATE}, {STAGE1_MAX_IMPACT_RATE}])"
        elif not has_span:
            reason = f"Isolated transient / insufficient temporal span ({temporal_span_sec:.2f}s < {STAGE1_MIN_SPAN_SECONDS:.2f}s)"
        elif not has_contrast:
            reason = f"Unrelated continuous sound / flat envelope (Valley Contrast={valley_contrast_db:.1f} dB < {STAGE1_MIN_VALLEY_CONTRAST_DB:.1f} dB)"
        elif not has_regularity:
            reason = f"Irregular stochastic timing (CV={interval_cv:.2f} > {STAGE1_MAX_INTERVAL_CV:.2f}: random rain drops / sporadic dripping)"
        elif not has_canister_resonance:
            reason = f"Low canister spectral ratio ({canister_spectral_ratio:.2f} < {STAGE1_MIN_CANISTER_SPECTRAL_RATIO:.2f}: low-frequency bubble plop)"
        else:
            reason = f"Aerodynamic turbulent hiss / non-impulsive friction noise (Crest Factor={crest_factor:.1f} < {STAGE1_MIN_CREST_FACTOR:.1f})"

        return {
            "is_candidate": is_candidate,
            "is_clipped": False,
            "rms": round(rms, 4),
            "peak": round(peak_amp, 4),
            "crest_factor": round(crest_factor, 1),
            "burst_count": burst_count,
            "cadence_hz": round(cadence_hz, 1),
            "temporal_span_sec": round(temporal_span_sec, 2),
            "valley_contrast_db": round(valley_contrast_db, 1),
            "interval_cv": round(interval_cv, 2),
            "canister_spectral_ratio": round(canister_spectral_ratio, 2),
            "reason": reason,
        }

