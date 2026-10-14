"""Audio Utilities for Two-Stage Spray Can Sound Detector.

Provides:
- Robust audio loading & resampling
- Full 2D Log-Mel spectrogram feature extraction
- SpecAugment and time-domain augmentations
- Accurate acoustic synthesizers for hard negatives (coins, keys, pill bottles, liquid bottles, glassware)
  and ambient environmental sounds for balanced multi-class training and hard-negative evaluation.
"""

import base64
import io
import random
from pathlib import Path
from typing import Optional, Union

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch

from config import (
    AUDIO_SAMPLES,
    EXPECTED_TIME_FRAMES,
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)


def load_audio(source: Union[str, Path, bytes, np.ndarray], sr: int = SAMPLE_RATE, mono: bool = True) -> np.ndarray:
    """Load an audio source into a 1D float32 numpy array normalized to [-1.0, 1.0]."""
    if isinstance(source, np.ndarray):
        audio = source.astype(np.float32)
        if mono and audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio

    if isinstance(source, (str, Path)):
        source = str(source)
    elif isinstance(source, bytes):
        source = io.BytesIO(source)

    try:
        audio, orig_sr = sf.read(source, dtype="float32", always_2d=False)
    except Exception:
        if isinstance(source, io.BytesIO):
            source.seek(0)
        audio, orig_sr = librosa.load(source, sr=sr, mono=mono)
        return audio.astype(np.float32)

    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 1.0:
        audio = audio / peak

    return audio.astype(np.float32)


def resample_audio_poly(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Downsample or upsample audio using polyphase sinc filter with anti-aliasing.

    Equivalent to librosa.resample but over 1000x faster, ensuring live microphone
    transients match training data without high-frequency aliasing foldback.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    gcd = int(np.gcd(orig_sr, target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def pad_or_trim(audio: np.ndarray, target_length: int = AUDIO_SAMPLES) -> np.ndarray:
    """Pad with zeros or trim audio array to exactly target_length."""
    if len(audio) >= target_length:
        return audio[:target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[:len(audio)] = audio
    return padded


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """Extract Log-Mel Spectrogram matrix of shape (n_mels, expected_time_frames).

    Captures the full 2D spectro-temporal time pattern over the 1.0-second window.
    """
    audio_fixed = pad_or_trim(audio, AUDIO_SAMPLES)

    # Compute short-time Fourier transform power spectrogram
    stft = librosa.stft(
        y=audio_fixed,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
    )
    power_spec = np.abs(stft) ** 2

    # Mel filter bank
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    mel_spec = np.dot(mel_basis, power_spec)

    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # Ensure shape is exactly (N_MELS, EXPECTED_TIME_FRAMES)
    if log_mel.shape[1] < EXPECTED_TIME_FRAMES:
        pad_width = EXPECTED_TIME_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=log_mel.min())
    elif log_mel.shape[1] > EXPECTED_TIME_FRAMES:
        log_mel = log_mel[:, :EXPECTED_TIME_FRAMES]

    # Normalize to zero-mean, unit-variance for neural network input
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-6
    normalized_spec = (log_mel - mean) / std

    return normalized_spec.astype(np.float32)


def apply_spec_augment(
    spec: np.ndarray,
    freq_mask_max: int = 8,
    time_mask_max: int = 6,
) -> np.ndarray:
    """Apply SpecAugment frequency and time masking to Log-Mel Spectrogram."""
    augmented = spec.copy()
    n_mels, n_frames = augmented.shape

    # Frequency masking
    f_width = random.randint(1, freq_mask_max)
    f_start = random.randint(0, n_mels - f_width)
    augmented[f_start : f_start + f_width, :] = 0.0

    # Time masking
    t_width = random.randint(1, time_mask_max)
    t_start = random.randint(0, n_frames - t_width)
    augmented[:, t_start : t_start + t_width] = 0.0

    return augmented


def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply time-domain acoustic augmentations."""
    aug = audio.copy()
    choice = random.choice(["shift", "noise", "gain", "pitch"])

    if choice == "shift":
        shift_amt = random.randint(-int(sr * 0.15), int(sr * 0.15))
        aug = np.roll(aug, shift_amt)
    elif choice == "noise":
        noise_level = random.uniform(0.002, 0.015)
        aug = aug + noise_level * np.random.randn(len(aug)).astype(np.float32)
    elif choice == "gain":
        gain = random.uniform(0.7, 1.3)
        aug = np.clip(aug * gain, -1.0, 1.0)
    elif choice == "pitch":
        steps = random.uniform(-1.5, 1.5)
        try:
            aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=steps)
        except Exception:
            pass

    return aug.astype(np.float32)


def spectrogram_to_base64_image(spec: np.ndarray, title: str = "Log-Mel Spectrogram") -> str:
    """Render Log-Mel Spectrogram to a Base64-encoded PNG image."""
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    fig.patch.set_facecolor("#0d121f")
    ax.set_facecolor("#0d121f")

    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title(title, color="#e2e8f0", fontsize=10, pad=8, fontweight="bold")
    ax.set_xlabel("Time Frames (1.0s window)", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Mel Frequency Bands", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


# ==============================================================================
# Hard Negative Acoustic Synthesizers (Coins, Keys, Pill Bottles, Glass, Liquid)
# ==============================================================================

def generate_hard_negative_coins(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate coins shaking in a jar/pocket: metallic multi-contact clatter with high dispersion."""
    n = int(duration * sr)
    audio = 0.003 * np.random.randn(n)
    num_clatters = random.randint(6, 14)
    for _ in range(num_clatters):
        pos = random.randint(0, n - int(sr * 0.1))
        c_len = random.randint(int(sr * 0.015), int(sr * 0.06))
        decay = np.exp(-np.linspace(0, 10, c_len))
        # Coins produce multiple non-cavity metallic ringing frequencies (3.5k - 7.5k)
        t = np.arange(c_len) / sr
        clink = np.zeros(c_len)
        for _ in range(3):
            freq = random.uniform(3200, 7800)
            clink += np.sin(2 * np.pi * freq * t) * decay
        audio[pos : pos + c_len] += 0.3 * clink
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.7)).astype(np.float32)


def generate_hard_negative_pill_bottle(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate pill bottle or Tic-Tac shaking: muted plastic resonance with rapid internal damping."""
    n = int(duration * sr)
    audio = 0.002 * np.random.randn(n)
    num_shakes = random.randint(5, 12)
    for _ in range(num_shakes):
        pos = random.randint(0, n - int(sr * 0.08))
        c_len = random.randint(int(sr * 0.01), int(sr * 0.04))
        decay = np.exp(-np.linspace(0, 18, c_len))  # Heavy plastic damping
        t = np.arange(c_len) / sr
        # Plastic wall resonance centered between 800 Hz and 2.2 kHz
        freq = random.uniform(850, 2100)
        audio[pos : pos + c_len] += 0.4 * np.sin(2 * np.pi * freq * t) * decay
        audio[pos : pos + c_len] += 0.1 * np.random.randn(c_len) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.65)).astype(np.float32)


def generate_hard_negative_keys(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate key jangling: loose metallic rings with sliding friction and high-frequency chime."""
    n = int(duration * sr)
    audio = 0.002 * np.random.randn(n)
    num_jangles = random.randint(8, 16)
    for _ in range(num_jangles):
        pos = random.randint(0, n - int(sr * 0.08))
        c_len = random.randint(int(sr * 0.02), int(sr * 0.07))
        decay = np.exp(-np.linspace(0, 7, c_len))
        t = np.arange(c_len) / sr
        jingle = np.zeros(c_len)
        for _ in range(4):
            f = random.uniform(4000, 9500)
            jingle += np.sin(2 * np.pi * f * t) * decay
        audio[pos : pos + c_len] += 0.25 * jingle
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.35, 0.75)).astype(np.float32)


def generate_hard_negative_screws_tin(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate loose metal screws/nuts shaking in a box."""
    n = int(duration * sr)
    audio = 0.003 * np.random.randn(n)
    num_clicks = random.randint(10, 22)
    for _ in range(num_clicks):
        pos = random.randint(0, n - int(sr * 0.05))
        c_len = random.randint(int(sr * 0.01), int(sr * 0.035))
        decay = np.exp(-np.linspace(0, 12, c_len))
        t = np.arange(c_len) / sr
        f = random.uniform(2200, 5800)
        audio[pos : pos + c_len] += 0.35 * np.sin(2 * np.pi * f * t) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.7)).astype(np.float32)


def generate_hard_negative_liquid_bottle(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate liquid water bottle / shampoo sloshing and container compression."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    audio = 0.002 * np.random.randn(n)
    # Low sloshing surges (100 - 450 Hz)
    num_sloshes = random.randint(2, 5)
    for _ in range(num_sloshes):
        pos = random.randint(0, n - int(sr * 0.2))
        s_len = int(sr * 0.2)
        win = np.hanning(s_len)
        f = random.uniform(120, 380)
        audio[pos : pos + s_len] += 0.4 * np.sin(2 * np.pi * f * np.arange(s_len) / sr) * win
    # Add plastic crinkle taps
    for _ in range(random.randint(2, 6)):
        pos = random.randint(0, n - int(sr * 0.04))
        c_len = int(sr * 0.02)
        decay = np.exp(-np.linspace(0, 14, c_len))
        audio[pos : pos + c_len] += 0.2 * np.random.randn(c_len) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.65)).astype(np.float32)


def generate_hard_negative_bottle_caps(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate metal crown bottle caps rattling and clinking together."""
    n = int(duration * sr)
    audio = 0.002 * np.random.randn(n)
    num_clicks = random.randint(8, 18)
    for _ in range(num_clicks):
        pos = random.randint(0, n - int(sr * 0.06))
        c_len = random.randint(int(sr * 0.015), int(sr * 0.045))
        decay = np.exp(-np.linspace(0, 11, c_len))
        t = np.arange(c_len) / sr
        f = random.uniform(2800, 7200)
        audio[pos : pos + c_len] += 0.35 * np.sin(2 * np.pi * f * t) * decay
        audio[pos : pos + c_len] += 0.15 * np.random.randn(c_len) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.7)).astype(np.float32)


def generate_hard_negative_containers(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate shaking other containers (plastic tub, tin, box with beads/dice/paperclips)."""
    n = int(duration * sr)
    audio = 0.002 * np.random.randn(n)
    num_bursts = random.randint(6, 14)
    for _ in range(num_bursts):
        pos = random.randint(0, n - int(sr * 0.08))
        c_len = random.randint(int(sr * 0.02), int(sr * 0.06))
        decay = np.exp(-np.linspace(0, 9, c_len))
        t = np.arange(c_len) / sr
        # Muted container resonance between 700 Hz and 2400 Hz
        f = random.uniform(700, 2400)
        audio[pos : pos + c_len] += 0.3 * np.sin(2 * np.pi * f * t) * decay
        audio[pos : pos + c_len] += 0.2 * np.random.randn(c_len) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.65)).astype(np.float32)


def generate_hard_negative_glassware(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate glassware clinking / vibrating."""
    n = int(duration * sr)
    audio = 0.001 * np.random.randn(n)
    for _ in range(random.randint(2, 6)):
        pos = random.randint(0, n - int(sr * 0.15))
        c_len = int(sr * 0.12)
        decay = np.exp(-np.linspace(0, 4, c_len))  # Long undamped glass ringing
        t = np.arange(c_len) / sr
        f = random.uniform(1800, 3200)
        audio[pos : pos + c_len] += 0.4 * np.sin(2 * np.pi * f * t) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.7)).astype(np.float32)


# ==============================================================================
# Ambient Negative Synthesizers (Room noise, HVAC, Speech, Keyboard)
# ==============================================================================

def generate_ambient_room_noise(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate ambient room tone and microphone noise floor."""
    n = int(duration * sr)
    amp = random.uniform(0.001, 0.015)
    b, a = signal.butter(1, 1200 / (sr / 2), btype="low")
    noise = signal.lfilter(b, a, np.random.randn(n))
    audio = amp * (noise / (np.max(np.abs(noise)) + 1e-8))
    return audio.astype(np.float32)


def generate_ambient_speech(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate human voice formants (120 Hz fundamental with formants 500 Hz, 1500 Hz, 2500 Hz)."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    f0 = random.uniform(110, 220)
    audio = np.zeros(n)
    for formant in [random.uniform(400, 800), random.uniform(1100, 1800), random.uniform(2200, 2800)]:
        audio += 0.03 * np.sin(2 * np.pi * formant * t) * (1 + 0.5 * np.sin(2 * np.pi * f0 * t))
    audio += 0.004 * np.random.randn(n)
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.7)).astype(np.float32)


def generate_ambient_keyboard(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate mechanical keyboard typing clicks."""
    n = int(duration * sr)
    audio = 0.002 * np.random.randn(n)
    for _ in range(random.randint(2, 6)):
        pos = random.randint(0, n - 2000)
        c_len = random.randint(400, 1600)
        decay = np.exp(-np.linspace(0, 10, c_len))
        click = np.random.uniform(-0.15, 0.15, c_len) * decay
        audio[pos : pos + c_len] += click
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.2, 0.6)).astype(np.float32)


def generate_ambient_fan_hum(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate electrical / fan low hum (50/60/120 Hz)."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    f0 = random.choice([50.0, 60.0, 100.0, 120.0])
    audio = 0.04 * np.sin(2 * np.pi * f0 * t) + 0.02 * np.sin(2 * np.pi * 2 * f0 * t) + 0.008 * np.random.randn(n)
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.2, 0.5)).astype(np.float32)


# ==============================================================================
# Loud Sudden Impact Synthesizers (Door slams, desk bangs, dropped books, claps)
# ==============================================================================

def generate_impact_door_slam(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate a single loud door slam: deep acoustic boom with fast decay, no repeated cadence."""
    n = int(duration * sr)
    audio = 0.001 * np.random.randn(n)
    pos = random.randint(int(sr * 0.1), int(sr * 0.3))
    hit_len = int(sr * 0.18)
    t = np.arange(hit_len) / sr
    decay = np.exp(-np.linspace(0, 14, hit_len))
    f0 = random.uniform(85, 160)
    audio[pos : pos + hit_len] += 0.8 * np.sin(2 * np.pi * f0 * t) * decay
    audio[pos : pos + hit_len] += 0.3 * np.random.randn(hit_len) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.7, 0.95)).astype(np.float32)


def generate_impact_desk_bang(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate a single sharp desk bang / heavy object impact."""
    n = int(duration * sr)
    audio = 0.001 * np.random.randn(n)
    pos = random.randint(int(sr * 0.1), int(sr * 0.3))
    hit_len = int(sr * 0.08)
    t = np.arange(hit_len) / sr
    decay = np.exp(-np.linspace(0, 20, hit_len))
    audio[pos : pos + hit_len] += 0.9 * np.random.randn(hit_len) * decay
    audio[pos : pos + hit_len] += 0.5 * np.sin(2 * np.pi * 320 * t) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.7, 0.95)).astype(np.float32)


def generate_impact_dropped_object(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate dropped item on floor: initial heavy impact + 1 tiny micro-bounce."""
    n = int(duration * sr)
    audio = 0.001 * np.random.randn(n)
    pos = random.randint(int(sr * 0.1), int(sr * 0.25))
    hit1_len = int(sr * 0.06)
    decay1 = np.exp(-np.linspace(0, 16, hit1_len))
    audio[pos : pos + hit1_len] += 0.85 * np.random.randn(hit1_len) * decay1
    # Tiny secondary bounce 45ms later
    pos2 = pos + hit1_len + int(sr * 0.04)
    if pos2 + int(sr * 0.03) < n:
        hit2_len = int(sr * 0.03)
        decay2 = np.exp(-np.linspace(0, 25, hit2_len))
        audio[pos2 : pos2 + hit2_len] += 0.25 * np.random.randn(hit2_len) * decay2
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.65, 0.9)).astype(np.float32)


def generate_impact_clap(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate a single loud hand clap: sharp acoustic impulse with rapid room decay."""
    n = int(duration * sr)
    audio = 0.001 * np.random.randn(n)
    pos = random.randint(int(sr * 0.1), int(sr * 0.4))
    clap_len = int(sr * 0.035)
    t = np.arange(clap_len) / sr
    decay = np.exp(-np.linspace(0, 22, clap_len))
    b, a = signal.butter(2, [1200 / (sr / 2), 4500 / (sr / 2)], btype="band")
    burst = signal.lfilter(b, a, np.random.randn(clap_len))
    audio[pos : pos + clap_len] += 0.9 * (burst / (np.max(np.abs(burst)) + 1e-8)) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.7, 0.95)).astype(np.float32)


# ==============================================================================
# Continuous Environmental & Transport Synthesizers (Rain, water, trains, hissing, music)
# ==============================================================================

def generate_env_rain(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate continuous rain falling: broadband filtered white/pink noise with random droplet hits."""
    n = int(duration * sr)
    b, a = signal.butter(2, [800 / (sr / 2), 7000 / (sr / 2)], btype="band")
    rain_noise = signal.lfilter(b, a, np.random.randn(n))
    audio = 0.4 * (rain_noise / (np.max(np.abs(rain_noise)) + 1e-8))
    # Add random soft droplet clicks
    for _ in range(random.randint(8, 20)):
        pos = random.randint(0, n - 200)
        c_len = random.randint(80, 250)
        audio[pos : pos + c_len] += 0.15 * np.random.randn(c_len) * np.hanning(c_len)
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.7)).astype(np.float32)


def generate_env_water_drops(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate water droplets dripping into a sink / basin (cavity resonance plinks)."""
    n = int(duration * sr)
    audio = 0.001 * np.random.randn(n)
    num_drops = random.randint(2, 5)
    for _ in range(num_drops):
        pos = random.randint(0, n - int(sr * 0.1))
        d_len = int(sr * 0.06)
        t = np.arange(d_len) / sr
        f_start = random.uniform(900, 1600)
        # Pitch upward frequency sweep characteristic of air cavity closure
        f_sweep = f_start + 400 * (t / (d_len / sr))
        decay = np.exp(-np.linspace(0, 15, d_len))
        audio[pos : pos + d_len] += 0.5 * np.sin(2 * np.pi * f_sweep * t) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.65)).astype(np.float32)


def generate_env_train(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate train passing: low-frequency rail rumble (40 - 160 Hz) and wheel track clatter."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    # Heavy low rail rumble
    rumble = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 165 * t)
    b, a = signal.butter(1, 400 / (sr / 2), btype="low")
    filtered_rumble = signal.lfilter(b, a, rumble + 0.3 * np.random.randn(n))
    audio = 0.6 * (filtered_rumble / (np.max(np.abs(filtered_rumble)) + 1e-8))
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.4, 0.8)).astype(np.float32)


def generate_env_spray_hiss(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate continuous aerosol spraying hiss: high-pressure turbulent jet (3 - 10 kHz)."""
    return generate_spray_continuous_hiss(duration=duration, sr=sr)


# ==============================================================================
# Aerosol Spray Specific Negative Synthesizers (Continuous & Pulsed Hiss, Sputter, Cap)
# ==============================================================================

def generate_spray_continuous_hiss(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate authentic continuous aerosol spray hiss: pressurized turbulent jet (2.8 - 10.5 kHz)."""
    n = int(duration * sr)
    # Pressurized turbulent air jet noise
    b, a = signal.butter(2, [2800 / (sr / 2), min(10500, sr / 2 - 100) / (sr / 2)], btype="band")
    hiss = signal.lfilter(b, a, np.random.randn(n))
    # Soft amplitude flutter (aerodynamic turbulence pulsation)
    t = np.linspace(0, duration, n)
    flutter = 1.0 + 0.12 * np.sin(2 * np.pi * random.uniform(12, 35) * t)
    audio = hiss * flutter
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.40, 0.80)).astype(np.float32)


def generate_spray_pulsed_bursts(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate intermittent pulsed aerosol spraying ('pssht... pssht...'): quick nozzle presses."""
    n = int(duration * sr)
    audio = 0.001 * np.random.randn(n)
    num_pulses = random.randint(2, 4)
    # Generate distinct spray pulses
    t_step = duration / (num_pulses + 1)
    for p_idx in range(num_pulses):
        p_center = (p_idx + 1) * t_step + random.uniform(-0.05, 0.05)
        p_dur = random.uniform(0.12, 0.28)
        p_samples = int(p_dur * sr)
        p_start = max(0, int((p_center - p_dur / 2) * sr))
        if p_start + p_samples < n:
            # Turbulent jet burst
            b, a = signal.butter(2, [2600 / (sr / 2), min(10000, sr / 2 - 100) / (sr / 2)], btype="band")
            burst = signal.lfilter(b, a, np.random.randn(p_samples))
            # Hanning window envelope on pulse
            env = np.hanning(p_samples)
            # Tiny plastic click at press onset (actuator stem depressing)
            click_len = min(60, p_samples)
            burst[:click_len] += 0.3 * np.random.randn(click_len)
            audio[p_start : p_start + p_samples] += burst * env
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.45, 0.85)).astype(np.float32)


def generate_spray_nozzle_sputter(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate low-pressure sputtering aerosol spray (liquid droplet spray, uneven discharge)."""
    n = int(duration * sr)
    b, a = signal.butter(2, [2200 / (sr / 2), min(9000, sr / 2 - 100) / (sr / 2)], btype="band")
    base_hiss = signal.lfilter(b, a, np.random.randn(n))
    # Random droplet sputters / micro-pops
    audio = 0.4 * (base_hiss / (np.max(np.abs(base_hiss)) + 1e-8))
    for _ in range(random.randint(6, 18)):
        pos = random.randint(0, n - 150)
        pop_len = random.randint(40, 120)
        audio[pos : pos + pop_len] += 0.25 * np.random.randn(pop_len) * np.hanning(pop_len)
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.35, 0.70)).astype(np.float32)


def generate_spray_handling_and_cap(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate handling of spray can without shaking: cap click removal, plastic tap, pickup."""
    n = int(duration * sr)
    audio = 0.001 * np.random.randn(n)
    # Plastic cap snap off
    pos = random.randint(int(sr * 0.1), int(sr * 0.3))
    click_len = int(sr * 0.025)
    t = np.arange(click_len) / sr
    decay = np.exp(-np.linspace(0, 18, click_len))
    audio[pos : pos + click_len] += 0.7 * np.sin(2 * np.pi * random.uniform(2200, 3800) * t) * decay
    # Secondary plastic flex
    pos2 = pos + int(sr * 0.06)
    if pos2 + click_len < n:
        audio[pos2 : pos2 + click_len] += 0.3 * np.sin(2 * np.pi * random.uniform(1600, 2600) * t) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.3, 0.65)).astype(np.float32)


def generate_env_music(duration: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate percussive music rhythm: kick drum and snare pattern."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    audio = 0.002 * np.random.randn(n)
    # 2 bass drum hits + 2 snare hits
    for pos, hit_type in [(int(0.1 * sr), "kick"), (int(0.45 * sr), "snare"), (int(0.8 * sr), "kick"), (int(1.15 * sr), "snare")]:
        if pos + int(0.1 * sr) < n:
            h_len = int(0.08 * sr)
            decay = np.exp(-np.linspace(0, 12, h_len))
            if hit_type == "kick":
                audio[pos : pos + h_len] += 0.8 * np.sin(2 * np.pi * 75 * np.linspace(0, 0.08, h_len)) * decay
            else:
                b, a = signal.butter(1, 1500 / (sr / 2), btype="high")
                sn = signal.lfilter(b, a, np.random.randn(h_len))
                audio[pos : pos + h_len] += 0.6 * (sn / (np.max(np.abs(sn)) + 1e-8)) * decay
    peak = np.max(np.abs(audio))
    return (audio / (peak + 1e-8) * random.uniform(0.4, 0.8)).astype(np.float32)
