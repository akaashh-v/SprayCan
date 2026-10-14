"""Dataset Builder for Two-Stage Precision Aerosol Spray Can Sound Detector.

Implements strict Group-Based Splitting & Label Auditing (Rule 6):
1. Audits labels in spray can shaking recordings:
   - Discards silence, digital clipping, and non-shaking handling.
   - Ensures ONLY verified mechanical ball rattling is labeled as spray_can_shaking.
2. Explicit Dedicated Negative Category for Aerosol Spray Hiss:
   - Continuous pressurized spray hisses
   - Intermittent pulsed spray bursts ('pssht... pssht...')
   - Nozzle sputter & discharge
   - Cap click & can handling
3. 6 Separate Acoustic Classes:
   0: ambient_and_speech (speech, room tone, fan hum, keyboard, breathing)
   1: loud_impact (door slams, desk bangs, dropped objects, claps)
   2: continuous_environmental (rain, water drops, train rumbles, percussive music)
   3: hard_negative_rattle (coins, keys, pill bottles, screws in tin, bottle caps, glassware, liquid bottle)
   4: aerosol_spray_hiss (continuous & pulsed aerosol spraying, nozzle discharge hiss, sputtering)
   5: spray_can_shaking (ONLY TARGET: verified aerosol spray can shaken back and forth)

Outputs PyTorch partitions: train.pt, val.pt, test.pt and split_manifest.json.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from audio_utils import (
    apply_spec_augment,
    augment_audio,
    compute_log_mel_spectrogram,
    generate_ambient_fan_hum,
    generate_ambient_keyboard,
    generate_ambient_room_noise,
    generate_ambient_speech,
    generate_env_music,
    generate_env_rain,
    generate_env_train,
    generate_env_water_drops,
    generate_hard_negative_bottle_caps,
    generate_hard_negative_coins,
    generate_hard_negative_containers,
    generate_hard_negative_glassware,
    generate_hard_negative_keys,
    generate_hard_negative_liquid_bottle,
    generate_hard_negative_pill_bottle,
    generate_hard_negative_screws_tin,
    generate_impact_clap,
    generate_impact_desk_bang,
    generate_impact_door_slam,
    generate_impact_dropped_object,
    generate_spray_continuous_hiss,
    generate_spray_handling_and_cap,
    generate_spray_nozzle_sputter,
    generate_spray_pulsed_bursts,
    load_audio,
    pad_or_trim,
)
from config import (
    AUDIO_SAMPLES,
    CLASS_NAMES,
    CLASS_TO_IDX,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_AMBIENT_DIR,
    RAW_ENV_DIR,
    RAW_HARD_NEG_DIR,
    RAW_LOUD_IMPACT_DIR,
    RAW_SPRAY_DIR,
    RAW_SPRAY_HISS_DIR,
    SAMPLE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_aerosol_spray_hiss_corpus() -> None:
    """Generate dedicated aerosol spray hiss recordings (continuous & pulsed spray, sputtering, cap click)."""
    logger.info("Generating realistic Dedicated Aerosol Spray Hiss recordings...")
    RAW_SPRAY_HISS_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        ("continuous_spray", generate_spray_continuous_hiss, 35),
        ("pulsed_spray_bursts", generate_spray_pulsed_bursts, 35),
        ("nozzle_sputter", generate_spray_nozzle_sputter, 35),
        ("handling_and_cap", generate_spray_handling_and_cap, 35),
    ]

    for prefix, gen_fn, count in generators:
        for i in range(count):
            dur = random.uniform(1.2, 2.5)
            audio = gen_fn(duration=dur, sr=SAMPLE_RATE)
            fpath = RAW_SPRAY_HISS_DIR / f"{prefix}_{i+1:03d}.wav"
            sf.write(str(fpath), audio, SAMPLE_RATE)


def generate_loud_impact_corpus() -> None:
    """Generate sudden high-intensity single impacts (door slams, desk bangs, dropped objects, claps)."""
    logger.info("Generating realistic Loud Sudden Impact recordings...")
    RAW_LOUD_IMPACT_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        ("door_slam", generate_impact_door_slam, 35),
        ("desk_bang", generate_impact_desk_bang, 35),
        ("dropped_object", generate_impact_dropped_object, 35),
        ("clap", generate_impact_clap, 35),
    ]

    for prefix, gen_fn, count in generators:
        for i in range(count):
            dur = random.uniform(1.2, 2.0)
            audio = gen_fn(duration=dur, sr=SAMPLE_RATE)
            fpath = RAW_LOUD_IMPACT_DIR / f"{prefix}_{i+1:03d}.wav"
            sf.write(str(fpath), audio, SAMPLE_RATE)


def generate_continuous_environmental_corpus() -> None:
    """Generate continuous environmental confounders (rain, water drops, train rumbles, music)."""
    logger.info("Generating realistic Continuous Environmental recordings...")
    RAW_ENV_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        ("rain", generate_env_rain, 35),
        ("water_drops", generate_env_water_drops, 35),
        ("train_rumble", generate_env_train, 35),
        ("percussive_music", generate_env_music, 35),
    ]

    for prefix, gen_fn, count in generators:
        for i in range(count):
            dur = random.uniform(1.5, 2.5)
            audio = gen_fn(duration=dur, sr=SAMPLE_RATE)
            fpath = RAW_ENV_DIR / f"{prefix}_{i+1:03d}.wav"
            sf.write(str(fpath), audio, SAMPLE_RATE)


def generate_hard_negative_corpus() -> None:
    """Generate diverse hard negative rattle recordings (coins, keys, pill bottles, bottle caps, glassware, etc.)."""
    logger.info("Generating realistic Hard Negative recordings...")
    RAW_HARD_NEG_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        ("coins_jar", generate_hard_negative_coins, 35),
        ("pill_bottle", generate_hard_negative_pill_bottle, 35),
        ("keys_jangle", generate_hard_negative_keys, 35),
        ("screws_tin", generate_hard_negative_screws_tin, 35),
        ("bottle_caps", generate_hard_negative_bottle_caps, 35),
        ("containers", generate_hard_negative_containers, 35),
        ("liquid_bottle", generate_hard_negative_liquid_bottle, 35),
        ("glassware", generate_hard_negative_glassware, 35),
    ]

    for prefix, gen_fn, count in generators:
        for i in range(count):
            dur = random.uniform(1.2, 2.5)
            audio = gen_fn(duration=dur, sr=SAMPLE_RATE)
            fpath = RAW_HARD_NEG_DIR / f"{prefix}_{i+1:03d}.wav"
            sf.write(str(fpath), audio, SAMPLE_RATE)


def generate_ambient_negative_corpus() -> None:
    """Generate ambient environmental recordings (room tone, speech, fan, typing)."""
    logger.info("Generating realistic Ambient Negative recordings...")
    RAW_AMBIENT_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        ("room_noise", generate_ambient_room_noise, 40),
        ("speech", generate_ambient_speech, 40),
        ("fan_hum", generate_ambient_fan_hum, 35),
        ("keyboard", generate_ambient_keyboard, 35),
    ]

    for prefix, gen_fn, count in generators:
        for i in range(count):
            dur = random.uniform(1.2, 2.5)
            audio = gen_fn(duration=dur, sr=SAMPLE_RATE)
            fpath = RAW_AMBIENT_DIR / f"{prefix}_{i+1:03d}.wav"
            sf.write(str(fpath), audio, SAMPLE_RATE)


def slice_audio_file(audio: np.ndarray, hop_sec: float = 0.5) -> List[np.ndarray]:
    """Slice long audio into overlapping 1.0s windows."""
    target_samples = AUDIO_SAMPLES
    hop_samples = int(SAMPLE_RATE * hop_sec)
    slices = []

    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if 0.001 < peak < 0.25:
        audio = audio * (0.5 / peak)

    if len(audio) <= target_samples:
        padded = pad_or_trim(audio, target_samples)
        if np.sqrt(np.mean(padded**2)) >= 0.001:
            slices.append(padded)
        return slices

    start = 0
    while start + target_samples <= len(audio):
        chunk = audio[start : start + target_samples]
        if np.sqrt(np.mean(chunk**2)) >= 0.001:
            slices.append(chunk)
        start += hop_samples

    return slices


def get_session_id(fpath: Path) -> str:
    """Extract recording session / collection period identifier from file path.

    Ensures that multiple clips sliced from the same recording session or device
    stay strictly together in the same split (train, val, or test) to prevent data leakage.
    """
    stem = fpath.stem.lower()
    if stem.startswith("session_"):
        parts = stem.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    if stem.startswith("preview_"):
        parts = stem.split("_")
        if len(parts) >= 2:
            return f"session_author_{parts[1]}"
        return "session_preview_misc"
    if stem.startswith("clip_"):
        parts = stem.split("_")
        if len(parts) >= 2:
            return f"session_clip_{parts[1]}"
        return "session_clip_misc"
    return f"session_recording_{stem.split('.')[0]}"


def audit_shaking_slice(window: np.ndarray) -> bool:
    """Audit slice extracted from positive spray folder.

    Rejects:
    - Silence or low-energy handling lulls (RMS < 0.002)
    - Clipped signals (Peak >= 1.05)
    - Flat / non-impulsive friction hiss (Crest Factor < 4.2)
    Retains authentic mechanical ball rattle impacts!
    """
    peak = np.max(np.abs(window)) if len(window) > 0 else 0.0
    rms = np.sqrt(np.mean(window**2)) if len(window) > 0 else 0.0
    if rms < 0.002 or peak >= 1.05:
        return False
    crest = peak / (rms + 1e-8)
    return crest >= 4.2


def mix_audio_at_snr(signal_audio: np.ndarray, noise_audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal and noise at specified Signal-to-Noise Ratio (dB)."""
    target_len = AUDIO_SAMPLES
    s = pad_or_trim(signal_audio, target_len)
    n = pad_or_trim(noise_audio, target_len)

    s_rms = np.sqrt(np.mean(s**2)) + 1e-8
    n_rms = np.sqrt(np.mean(n**2)) + 1e-8

    desired_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    scale = desired_n_rms / n_rms
    mixed = s + n * scale

    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)
    return mixed.astype(np.float32)


def build_full_dataset() -> None:
    """Extract Log-Mel spectrograms with strict Session-Based Splitting & Overlapping Mixtures."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # 1. Ensure all 6 raw corpora exist
    hiss_files = list(RAW_SPRAY_HISS_DIR.glob("*.wav"))
    if len(hiss_files) < 40:
        generate_aerosol_spray_hiss_corpus()
        hiss_files = list(RAW_SPRAY_HISS_DIR.glob("*.wav"))

    hard_neg_files = list(RAW_HARD_NEG_DIR.glob("*.wav"))
    if len(hard_neg_files) < 40:
        generate_hard_negative_corpus()
        hard_neg_files = list(RAW_HARD_NEG_DIR.glob("*.wav"))

    ambient_files = list(RAW_AMBIENT_DIR.glob("*.wav"))
    if len(ambient_files) < 40:
        generate_ambient_negative_corpus()
        ambient_files = list(RAW_AMBIENT_DIR.glob("*.wav"))

    impact_files = list(RAW_LOUD_IMPACT_DIR.glob("*.wav"))
    if len(impact_files) < 40:
        generate_loud_impact_corpus()
        impact_files = list(RAW_LOUD_IMPACT_DIR.glob("*.wav"))

    env_files = list(RAW_ENV_DIR.glob("*.wav"))
    if len(env_files) < 40:
        generate_continuous_environmental_corpus()
        env_files = list(RAW_ENV_DIR.glob("*.wav"))

    spray_files = sorted(list(RAW_SPRAY_DIR.glob("*.mp3")) + list(RAW_SPRAY_DIR.glob("*.wav")))

    logger.info(
        "Raw audio sources by class: SprayShaking=%d, SprayHiss=%d, Impact=%d, ContinuousEnv=%d, HardNegRattle=%d, AmbientSpeech=%d",
        len(spray_files), len(hiss_files), len(impact_files), len(env_files), len(hard_neg_files), len(ambient_files)
    )

    class_source_map = {
        "spray_can_shaking": spray_files,
        "aerosol_spray_hiss": sorted(hiss_files),
        "loud_impact": sorted(impact_files),
        "continuous_environmental": sorted(env_files),
        "hard_negative_rattle": sorted(hard_neg_files),
        "ambient_and_speech": sorted(ambient_files),
    }

    # 2. Strict Session / Time-Period Group-Based Splitting (Prevents Leakage)
    split_manifest: Dict[str, Dict[str, List[str]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for class_name, files in class_source_map.items():
        session_to_files: Dict[str, List[Path]] = {}
        for f in files:
            sid = get_session_id(f)
            session_to_files.setdefault(sid, []).append(f)

        session_keys = sorted(list(session_to_files.keys()))
        rng = random.Random(RANDOM_SEED + CLASS_TO_IDX[class_name])
        rng.shuffle(session_keys)

        n_sessions = len(session_keys)
        n_val_sess = max(1, int(n_sessions * 0.15))
        n_test_sess = max(1, int(n_sessions * 0.15))
        n_train_sess = max(1, n_sessions - n_val_sess - n_test_sess)

        train_sessions = session_keys[:n_train_sess]
        val_sessions = session_keys[n_train_sess : n_train_sess + n_val_sess]
        test_sessions = session_keys[n_train_sess + n_val_sess :]

        split_manifest["train"][class_name] = [
            f.name for sid in train_sessions for f in session_to_files[sid]
        ]
        split_manifest["val"][class_name] = [
            f.name for sid in val_sessions for f in session_to_files[sid]
        ]
        split_manifest["test"][class_name] = [
            f.name for sid in test_sessions for f in session_to_files[sid]
        ]

        logger.info(
            "Class '%s' split: Train=%d files (%d sessions), Val=%d files (%d sessions), Test=%d files (%d sessions)",
            class_name,
            len(split_manifest["train"][class_name]), len(train_sessions),
            len(split_manifest["val"][class_name]), len(val_sessions),
            len(split_manifest["test"][class_name]), len(test_sessions),
        )

    # Pre-generate library of station noises for realistic overlap mixing
    station_noise_pool = [
        generate_env_train(duration=2.0, sr=SAMPLE_RATE),
        generate_env_rain(duration=2.0, sr=SAMPLE_RATE),
        generate_ambient_speech(duration=2.0, sr=SAMPLE_RATE),
        generate_ambient_fan_hum(duration=2.0, sr=SAMPLE_RATE),
    ]

    # 3. Process Windows for Each Partition
    train_data: List[Tuple[np.ndarray, int]] = []
    val_data: List[Tuple[np.ndarray, int]] = []
    test_data: List[Tuple[np.ndarray, int]] = []
    test_conditions: List[Dict] = []

    for split_name, partition_target, is_train in [
        ("train", train_data, True),
        ("val", val_data, False),
        ("test", test_data, False),
    ]:
        logger.info("Processing partition '%s'...", split_name)
        for class_name, files_map in class_source_map.items():
            class_idx = CLASS_TO_IDX[class_name]
            allowed_names = set(split_manifest[split_name][class_name])
            split_files = [f for f in files_map if f.name in allowed_names]

            hop = 0.35 if (class_name == "spray_can_shaking" and is_train) else 0.50

            for fpath in tqdm(split_files, desc=f"{split_name}:{class_name}"):
                try:
                    audio = load_audio(fpath, sr=SAMPLE_RATE)
                    windows = slice_audio_file(audio, hop_sec=hop)
                    for win in windows:
                        if class_name == "spray_can_shaking":
                            if not audit_shaking_slice(win):
                                rms = np.sqrt(np.mean(win**2))
                                if 0.001 < rms < 0.015:
                                    spec = compute_log_mel_spectrogram(win, sr=SAMPLE_RATE)
                                    partition_target.append((spec, CLASS_TO_IDX["ambient_and_speech"]))
                                continue

                        spec = compute_log_mel_spectrogram(win, sr=SAMPLE_RATE)
                        partition_target.append((spec, class_idx))

                        if split_name == "test":
                            if class_name == "spray_can_shaking":
                                cond = "quiet_clean"
                            elif class_name == "aerosol_spray_hiss":
                                cond = "spray_hiss"
                            elif class_name == "hard_negative_rattle":
                                cond = "similar_rattles"
                            elif class_name == "continuous_environmental":
                                cond = "overlapping_train_announcement"
                            else:
                                cond = "busy_station"
                            test_conditions.append({
                                "index": len(test_data) - 1,
                                "class_name": class_name,
                                "condition": cond,
                                "snr_db": None,
                                "source_file": fpath.name,
                            })

                        # Positive Shaking Augmentations & Overlapping Mixtures:
                        if class_name == "spray_can_shaking":
                            if is_train:
                                # Standard augmentations
                                aug_win = augment_audio(win, sr=SAMPLE_RATE)
                                aug_spec = compute_log_mel_spectrogram(aug_win, sr=SAMPLE_RATE)
                                aug_spec = apply_spec_augment(aug_spec)
                                partition_target.append((aug_spec, class_idx))

                                # Overlapping station noise mixtures in training (+12 dB to +3 dB)
                                bg = random.choice(station_noise_pool)
                                snr = random.uniform(3.0, 14.0)
                                mixed_win = mix_audio_at_snr(win, bg, snr)
                                mixed_spec = compute_log_mel_spectrogram(mixed_win, sr=SAMPLE_RATE)
                                partition_target.append((mixed_spec, class_idx))

                            elif split_name == "test":
                                # Controlled Overlapping Mixtures in Held-Out Test Set:
                                # Evaluate performance across real SNR levels (+12 dB, +6 dB, 0 dB, -4 dB)
                                for snr_val, cond_label in [
                                    (12.0, "overlapping_train_announcement"),
                                    (6.0, "overlapping_train_announcement"),
                                    (0.0, "overlapping_train_announcement"),
                                    (-4.0, "masked_extreme"),
                                ]:
                                    bg = random.choice(station_noise_pool)
                                    mixed_win = mix_audio_at_snr(win, bg, snr_val)
                                    mixed_spec = compute_log_mel_spectrogram(mixed_win, sr=SAMPLE_RATE)
                                    test_data.append((mixed_spec, class_idx))
                                    test_conditions.append({
                                        "index": len(test_data) - 1,
                                        "class_name": "spray_can_shaking",
                                        "condition": cond_label,
                                        "snr_db": snr_val,
                                        "source_file": fpath.name,
                                    })

                        elif is_train:
                            # Negative class augmentations
                            aug_win = augment_audio(win, sr=SAMPLE_RATE)
                            aug_spec = compute_log_mel_spectrogram(aug_win, sr=SAMPLE_RATE)
                            partition_target.append((aug_spec, class_idx))

                except Exception as e:
                    logger.warning("Error processing %s in %s: %s", fpath.name, split_name, e)

    # Shuffle training and validation partitions internally
    random.Random(RANDOM_SEED).shuffle(train_data)
    random.Random(RANDOM_SEED + 1).shuffle(val_data)

    def to_tensors(partition_list):
        x = torch.from_numpy(np.array([item[0] for item in partition_list], dtype=np.float32)).unsqueeze(1)
        y = torch.tensor([item[1] for item in partition_list], dtype=torch.long)
        return {"spectrograms": x, "labels": y}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_pt = to_tensors(train_data)
    val_pt = to_tensors(val_data)
    test_pt = to_tensors(test_data)

    torch.save(train_pt, PROCESSED_DIR / "train.pt")
    torch.save(val_pt, PROCESSED_DIR / "val.pt")
    torch.save(test_pt, PROCESSED_DIR / "test.pt")

    # Save manifest for data leakage verification
    with open(PROCESSED_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    with open(PROCESSED_DIR / "test_conditions.json", "w", encoding="utf-8") as f:
        json.dump(test_conditions, f, indent=2)

    def count_labels(pt):
        lbls = pt["labels"].tolist()
        return {name: lbls.count(CLASS_TO_IDX[name]) for name in CLASS_NAMES}

    dataset_info = {
        "sample_rate": SAMPLE_RATE,
        "window_duration_sec": 1.0,
        "spectrogram_shape": list(train_pt["spectrograms"].shape[1:]),
        "total_samples": len(train_data) + len(val_data) + len(test_data),
        "partitions": {
            "train": {"counts": count_labels(train_pt), "total": len(train_data)},
            "val": {"counts": count_labels(val_pt), "total": len(val_data)},
            "test": {"counts": count_labels(test_pt), "total": len(test_data)},
        },
    }

    with open(PROCESSED_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info("Dataset built successfully!")
    logger.info("  Train total: %d samples", len(train_data))
    logger.info("  Val total:   %d samples", len(val_data))
    logger.info("  Test total:  %d samples", len(test_data))


if __name__ == "__main__":
    build_full_dataset()
