"""Configuration for Two-Stage Precision Aerosol Spray Can Sound Detector.

Implements two-stage decision hyperparameters:
1. Stage 1: Kinematic shaking bursts candidate detection.
2. Stage 2: Deep spectro-temporal CNN classifier with hard-negative rejection.
3. Multi-window temporal persistence filter & alert cooldown.
"""

from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Raw category folders
RAW_SPRAY_DIR = RAW_DIR / "spray_can_shaking"
RAW_SPRAY_HISS_DIR = RAW_DIR / "aerosol_spray_hiss"
RAW_HARD_NEG_DIR = RAW_DIR / "hard_negatives"
RAW_AMBIENT_DIR = RAW_DIR / "ambient_negatives"
RAW_LOUD_IMPACT_DIR = RAW_DIR / "loud_impacts"
RAW_ENV_DIR = RAW_DIR / "continuous_environmental"

# Model Checkpoints & Evaluation artifacts
MODEL_SAVE_PATH = MODELS_DIR / "best_detector.pth"
BASELINE_MODEL_SAVE_PATH = MODELS_DIR / "baseline_detector.pth"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.json"
CALIBRATION_PATH = MODELS_DIR / "threshold_calibration.json"

for d in [RAW_SPRAY_DIR, RAW_SPRAY_HISS_DIR, RAW_HARD_NEG_DIR, RAW_AMBIENT_DIR, RAW_LOUD_IMPACT_DIR, RAW_ENV_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Audio Parameters
SAMPLE_RATE = 22050
WINDOW_DURATION = 1.0  # seconds
AUDIO_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)  # 22050 samples
STREAM_HOP_DURATION = 0.25  # 4 evaluations per second
HOP_SAMPLES = int(SAMPLE_RATE * STREAM_HOP_DURATION)

# Log-Mel Spectrogram Settings (Full 2D Time-Frequency Pattern)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 100.0
F_MAX = 11000.0
EXPECTED_TIME_FRAMES = 1 + (AUDIO_SAMPLES // HOP_LENGTH)  # 44 frames

# Stage 1: Kinematic Time-Sequence, Cadence & Cavity Resonance Parameters
STAGE1_MIN_RMS = 0.0004               # Sensitive energy floor for room/speaker audio
STAGE1_MIN_IMPACT_RATE = 2.5          # Minimum cadence for back-and-forth arm shaking (bursts/sec)
STAGE1_MAX_IMPACT_RATE = 15.0         # Maximum cadence for physical spray can shaking
STAGE1_MIN_BURSTS = 3                 # Must observe at least 3 distinct impact bursts in 1.0s window
STAGE1_MIN_SPAN_SECONDS = 0.35        # First to last burst must span at least 0.35s (rules out isolated single/double drops)
STAGE1_MIN_VALLEY_CONTRAST_DB = 4.2   # Deep valleys between strokes (rules out continuous rain, train, steady hissing)
STAGE1_MIN_CREST_FACTOR = 4.8         # Peak-to-RMS ratio: metallic impact bursts vs flat turbulent friction hiss (< 4.5)
STAGE1_MAX_INTERVAL_CV = 0.65         # Kinematic arm shaking regularity: CV of peak intervals <= 0.65 (rules out stochastic rain)
STAGE1_MIN_CANISTER_SPECTRAL_RATIO = 0.30  # High-to-low ratio: metal canister modal ringing (2.2-7.5 kHz) vs bubbles (200-1800 Hz)
CLIPPING_PEAK_THRESHOLD = 0.99        # Peak amplitude to identify digital clipping
CLIPPING_RATIO_THRESHOLD = 0.002      # Fraction of clipped samples identifying unusable audio

# Stage 2: Classifier Classes (Explicit 6-Class Confounder Breakdown)
# 0: ambient_and_speech (speech, vocalizations, breathing, room tone, keyboard, fan hum)
# 1: loud_impact (door slams, desk bangs, dropped books, claps, hammer hits)
# 2: continuous_environmental (rain, water drops, train rumble, percussive music)
# 3: hard_negative_rattle (coins, keys, pill bottles, loose screws/nuts in tin, bottle caps, glassware, liquid bottle)
# 4: aerosol_spray_hiss (continuous & pulsed aerosol spraying, nozzle discharge hiss, sputtering)
# 5: spray_can_shaking (ONLY TARGET: verified aerosol spray can shaken back and forth with agitator ball)
CLASS_NAMES = [
    "ambient_and_speech",
    "loud_impact",
    "continuous_environmental",
    "hard_negative_rattle",
    "aerosol_spray_hiss",
    "spray_can_shaking",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Conservative Decision & Gating Thresholds
CONSERVATIVE_THRESHOLD = 0.80         # Conservative confidence required for spray can class (>= 80%)
MAX_COMPETING_CONFIDENCE = 0.14       # Competing classes (spray hiss, rattles, impacts) must have low confidence (<= 14%)
MAX_ENV_CONFIDENCE = 0.10             # Environmental confounders (rain, water drops, train) must be strictly <= 10%
MIN_CONFIDENCE_MARGIN = 0.45          # Margin: P(spray) - max(P(competing)) >= 0.45
UNCERTAIN_ENTROPY_THRESHOLD = 0.65    # Normalized Shannon entropy threshold for ambiguous sounds

# Multi-Window Temporal Persistence & Cooldown Gating
REQUIRED_CONSECUTIVE_WINDOWS = 3      # Must match across >= 3 consecutive windows (>= 1.5s sustained shaking)
MIN_EVENT_DURATION_SECONDS = 1.0      # Minimum duration to confirm sustained shaking
ALERT_COOLDOWN_SECONDS = 0.5          # Cooldown to prevent repeating alerts for one episode (0.5s)

# Noise Cancellation Settings
ENABLE_NOISE_CANCELLATION = True      # Real-time adaptive spectral noise cancellation filter
NOISE_SUBTRACTION_FACTOR = 1.7        # Over-subtraction factor alpha for stationary noise
NOISE_FLOOR_FACTOR = 0.06             # Spectral floor factor beta (preserves acoustic transients, avoids musical noise)

# Privacy & Logging Settings
SAVE_MIC_AUDIO = False                # Do NOT store microphone audio unless explicitly configured

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
