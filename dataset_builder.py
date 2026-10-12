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


def audit_shaking_slice(window: np.ndarray) -> bool:
    """Audit slice extracted from positive spray folder.

    Rejects:
    - Silence or low-energy handling lulls (RMS < 0.004)
    - Clipped signals (Peak >= 1.05)
    - Flat / non-impulsive friction hiss (Crest Factor < 6.0)
    Only retains authentic mechanical ball rattle impacts!
    """
    peak = np.max(np.abs(window)) if len(window) > 0 else 0.0
    rms = np.sqrt(np.mean(window**2)) if len(window) > 0 else 0.0
    if rms < 0.004 or peak >= 1.05:
        return False
    crest = peak / (rms + 1e-8)
    return crest >= 6.0


def build_full_dataset() -> None:
    """Extract Log-Mel spectrograms with strict Group-Based Splitting & Label Auditing (Rule 6)."""
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

    # 2. Strict Group-Based Splitting by Source Recording File (Rule 6)
    split_manifest: Dict[str, Dict[str, List[str]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for class_name, files in class_source_map.items():
        rng = random.Random(RANDOM_SEED + CLASS_TO_IDX[class_name])
        shuffled = list(files)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_val = max(1, int(n_total * 0.15))
        n_test = max(1, int(n_total * 0.15))
        n_train = n_total - n_val - n_test

        train_srcs = shuffled[:n_train]
        val_srcs = shuffled[n_train : n_train + n_val]
        test_srcs = shuffled[n_train + n_val :]

        split_manifest["train"][class_name] = [f.name for f in train_srcs]
        split_manifest["val"][class_name] = [f.name for f in val_srcs]
        split_manifest["test"][class_name] = [f.name for f in test_srcs]

        logger.info(
            "Class '%s' split: Train=%d sources, Val=%d sources, Test=%d sources",
            class_name, len(train_srcs), len(val_srcs), len(test_srcs)
        )

    # 3. Process Windows for Each Partition with Label Auditing
    train_data: List[Tuple[np.ndarray, int]] = []
    val_data: List[Tuple[np.ndarray, int]] = []
    test_data: List[Tuple[np.ndarray, int]] = []

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
                        # Label Audit for Positive Spray Class:
                        if class_name == "spray_can_shaking":
                            if not audit_shaking_slice(win):
                                # If slice is quiet room tone or non-shaking handling, route to ambient
                                rms = np.sqrt(np.mean(win**2))
                                if 0.001 < rms < 0.015:
                                    spec = compute_log_mel_spectrogram(win, sr=SAMPLE_RATE)
                                    partition_target.append((spec, CLASS_TO_IDX["ambient_and_speech"]))
                                continue

                        # Base log-mel spectrogram
                        spec = compute_log_mel_spectrogram(win, sr=SAMPLE_RATE)
                        partition_target.append((spec, class_idx))

                        # Training augmentations
                        if is_train:
                            num_augs = 2 if class_name == "spray_can_shaking" else 1
                            for _ in range(num_augs):
                                aug_win = augment_audio(win, sr=SAMPLE_RATE)
                                aug_spec = compute_log_mel_spectrogram(aug_win, sr=SAMPLE_RATE)
                                aug_spec = apply_spec_augment(aug_spec)
                                partition_target.append((aug_spec, class_idx))
                except Exception as e:
                    logger.warning("Error processing %s in %s: %s", fpath.name, split_name, e)

    # Shuffle each partition internally
    random.Random(RANDOM_SEED).shuffle(train_data)
    random.Random(RANDOM_SEED + 1).shuffle(val_data)
    random.Random(RANDOM_SEED + 2).shuffle(test_data)

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
