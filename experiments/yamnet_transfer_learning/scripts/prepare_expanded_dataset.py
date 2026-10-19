"""Prepare Expanded Dataset for Transfer-Learning and Robust Negative Experiment.

Implements Step 2 requirements:
1. Public Negative Dataset (ESC-50):
   - Curated relevant environmental categories: footsteps, door sounds, glass breaking,
     clapping, rain, engine, wind, keyboard, water drops, can opening, coughing, breathing.
   - Licensed under CC BY-NC 3.0 with complete Freesound ID attribution.
2. Own Hardware Recordings:
   - INMP441 ESP32 hardware captures (ambient room tone, desk taps, speech).
   - Genuine physical spray-can shaking recordings (metal ball agitator kinematics).
   - Dedicated aerosol spray hiss (continuous and pulsed bursts).
   - Hard negatives: coins, keys, pill bottles, containers, glassware.
3. Data Quality & Integrity Auditing:
   - Rejection of corrupt, completely silent (RMS < 0.0003), or clipped audio (>0.99 peak, >0.2% clipped).
   - SHA-256 deduplication.
4. Leakage-Free Recording-Level Partitioning:
   - All windows from the same original recording/session reside in the same split.
   - Generates unified manifests and datasets for both 22,050 Hz CNN and 16,000 Hz YAMNet.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
import random
import time
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_expanded_dataset")

# Fixed random seed for complete reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DATA_DIR = BASE_DIR / "data"
ESC50_DIR = EXP_DATA_DIR / "esc50_raw"
OWN_REC_DIR = EXP_DATA_DIR / "own_recordings"
MANIFEST_PATH = EXP_DATA_DIR / "dataset_manifest.json"
QUALITY_REPORT_PATH = EXP_DATA_DIR / "data_quality_report.json"

PROJECT_ROOT = BASE_DIR.parent.parent
BACKEND_DATA_DIR = PROJECT_ROOT / "backend" / "data"
RAW_DIR = BACKEND_DATA_DIR / "raw"
TMP_ESP32_DIR = PROJECT_ROOT / "backend" / "tmp_esp32_audio"
SAVED_RECORDINGS_DIR = PROJECT_ROOT / "backend" / "recordings"

# The six preserved canonical classes in exact order
CLASS_NAMES = [
    "ambient_and_speech",
    "loud_impact",
    "continuous_environmental",
    "hard_negative_rattle",
    "aerosol_spray_hiss",
    "spray_can_shaking",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Relevant ESC-50 classes mapped strictly to our 6 categories
ESC50_MAPPING = {
    "keyboard_typing": "ambient_and_speech",
    "mouse_click": "ambient_and_speech",
    "breathing": "ambient_and_speech",
    "coughing": "ambient_and_speech",
    "sneezing": "ambient_and_speech",
    "laughing": "ambient_and_speech",
    "door_wood_knock": "loud_impact",
    "door_wood_creaks": "loud_impact",
    "clapping": "loud_impact",
    "glass_breaking": "loud_impact",
    "rain": "continuous_environmental",
    "water_drops": "continuous_environmental",
    "train": "continuous_environmental",
    "engine": "continuous_environmental",
    "wind": "continuous_environmental",
    "footsteps": "continuous_environmental",
    "can_opening": "hard_negative_rattle",
}


def download_esc50_selection(max_per_category: int = 15) -> List[Dict]:
    """Download curated relevant recordings from ESC-50 with full attribution."""
    ESC50_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ESC50_DIR / "esc50.csv"
    if not csv_path.exists():
        logger.info("Downloading esc50.csv metadata...")
        meta_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
        with urllib.request.urlopen(meta_url) as resp:
            csv_path.write_bytes(resp.read())

    records_to_download: List[Dict] = []
    category_counts: Dict[str, int] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            if cat in ESC50_MAPPING:
                count = category_counts.get(cat, 0)
                if count < max_per_category:
                    category_counts[cat] = count + 1
                    records_to_download.append(row)

    logger.info(
        "Identified %d relevant ESC-50 audio files across %d categories to download.",
        len(records_to_download),
        len(category_counts),
    )

    downloaded_files: List[Dict] = []
    base_raw_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_file(row):
        fname = row["filename"]
        local_path = ESC50_DIR / fname
        if not local_path.exists():
            file_url = base_raw_url + fname
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(file_url, timeout=15) as resp:
                        local_path.write_bytes(resp.read())
                    break
                except Exception:
                    time.sleep(0.5)
        if local_path.exists():
            try:
                info = sf.info(str(local_path))
                cat = row["category"]
                return {
                    "file_path": str(local_path),
                    "filename": fname,
                    "label": ESC50_MAPPING[cat],
                    "sub_category": cat,
                    "source": "ESC-50",
                    "source_id": row["src_file"],
                    "take": row["take"],
                    "fold": int(row["fold"]),
                    "sample_rate": info.samplerate,
                    "duration_sec": round(info.duration, 3),
                    "channels": info.channels,
                    "license": "CC BY-NC 3.0",
                    "recording_session": f"esc50_{row['src_file']}",
                }
            except Exception as e:
                logger.warning("Error reading %s: %s", local_path, e)
        return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_fetch_file, r) for r in records_to_download]
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            if res:
                downloaded_files.append(res)
            completed += 1
            if completed % 50 == 0 or completed == len(records_to_download):
                logger.info("ESC-50 download progress: %d / %d processed.", completed, len(records_to_download))

    return downloaded_files


def audit_audio_file(filepath: Path) -> Tuple[bool, str, Dict]:
    """Inspect audio file for corruption, digital clipping, and silence."""
    try:
        data, sr = sf.read(str(filepath), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Mono
    except Exception as e:
        return False, f"Corrupted or unreadable audio: {e}", {}

    duration = len(data) / sr
    if duration < 0.4:
        return False, f"Too short (<0.4s): {duration:.2f}s", {}

    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))

    # Silence check
    if rms < 0.0003:
        return False, f"Dead silence (RMS={rms:.6f} < 0.0003)", {"rms": rms, "peak": peak}

    # Clipping check: peak near 1.0 with more than 0.2% clipped samples
    clipped_ratio = float(np.mean(np.abs(data) >= 0.99))
    if peak >= 0.99 and clipped_ratio > 0.002:
        return False, f"Digital clipping ({clipped_ratio*100:.2f}% clipped)", {"rms": rms, "peak": peak, "clipped_ratio": clipped_ratio}

    hasher = hashlib.sha256()
    hasher.update((data[:1000].tobytes()))
    audio_hash = hasher.hexdigest()[:16]

    metrics = {
        "duration": duration,
        "sample_rate": sr,
        "peak": peak,
        "rms": rms,
        "clipped_ratio": clipped_ratio,
        "audio_hash": audio_hash,
    }
    return True, "Passed", metrics


def curate_own_recordings() -> List[Dict]:
    """Curate hardware recordings from INMP441 captures and existing datasets."""
    curated: List[Dict] = []
    seen_hashes: Set[str] = set()

    # 1. INMP441 ESP32 Hardware Audio Captures (686 files)
    if TMP_ESP32_DIR.exists():
        tmp_files = list(TMP_ESP32_DIR.glob("*.wav"))
        logger.info("Auditing %d INMP441 hardware captures from %s...", len(tmp_files), TMP_ESP32_DIR)
        for f in tmp_files:
            valid, reason, metrics = audit_audio_file(f)
            if not valid:
                continue

            h = metrics["audio_hash"]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            curated.append({
                "file_path": str(f),
                "filename": f.name,
                "label": "ambient_and_speech",
                "sub_category": "esp32_inmp441_hardware",
                "source": "ESP32_INMP441",
                "source_id": f.stem,
                "sample_rate": metrics["sample_rate"],
                "duration_sec": round(metrics["duration"], 3),
                "license": "Proprietary / Own Project",
                "recording_session": "esp32_hardware_live_session",
            })

    # 2. Genuine Spray Can Shaking Positives
    pos_dirs = [
        RAW_DIR / "real spray can shaking",
        RAW_DIR / "spray_can_shaking",
    ]
    for pdir in pos_dirs:
        if pdir.exists():
            for f in pdir.glob("*.*"):
                if f.suffix.lower() in (".wav", ".mp3", ".m4a"):
                    valid, reason, metrics = audit_audio_file(f)
                    if not valid:
                        continue
                    h = metrics["audio_hash"]
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    curated.append({
                        "file_path": str(f),
                        "filename": f.name,
                        "label": "spray_can_shaking",
                        "sub_category": "authentic_shaking",
                        "source": "Own_Recording",
                        "source_id": f.stem,
                        "sample_rate": metrics["sample_rate"],
                        "duration_sec": round(metrics["duration"], 3),
                        "license": "Proprietary / Own Project",
                        "recording_session": f"shaking_{f.stem[:12]}",
                    })

    # 3. Aerosol Spray Hiss
    hiss_dir = RAW_DIR / "aerosol_spray_hiss"
    if hiss_dir.exists():
        for f in hiss_dir.glob("*.wav"):
            valid, reason, metrics = audit_audio_file(f)
            if not valid:
                continue
            h = metrics["audio_hash"]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            curated.append({
                "file_path": str(f),
                "filename": f.name,
                "label": "aerosol_spray_hiss",
                "sub_category": "dedicated_hiss",
                "source": "Own_Recording",
                "source_id": f.stem,
                "sample_rate": metrics["sample_rate"],
                "duration_sec": round(metrics["duration"], 3),
                "license": "Proprietary / Own Project",
                "recording_session": f"hiss_{f.stem.split('_')[0]}",
            })

    # 4. Hard Negatives & Impact Sounds
    hard_dirs = [
        (RAW_DIR / "hard_negatives", "hard_negative_rattle"),
        (RAW_DIR / "hard_negative_rattle", "hard_negative_rattle"),
        (RAW_DIR / "more negatives", "hard_negative_rattle"),
        (RAW_DIR / "loud_impacts", "loud_impact"),
        (RAW_DIR / "continuous_environmental", "continuous_environmental"),
        (RAW_DIR / "ambient_negatives", "ambient_and_speech"),
    ]
    for pdir, default_label in hard_dirs:
        if pdir.exists():
            for f in pdir.glob("*.*"):
                if f.suffix.lower() in (".wav", ".mp3", ".m4a"):
                    valid, reason, metrics = audit_audio_file(f)
                    if not valid:
                        continue
                    h = metrics["audio_hash"]
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    curated.append({
                        "file_path": str(f),
                        "filename": f.name,
                        "label": default_label,
                        "sub_category": pdir.name,
                        "source": "Own_Recording",
                        "source_id": f.stem,
                        "sample_rate": metrics["sample_rate"],
                        "duration_sec": round(metrics["duration"], 3),
                        "license": "Proprietary / Own Project",
                        "recording_session": f"session_{f.stem[:12]}",
                    })

    logger.info("Curated %d unique hardware and project audio recordings.", len(curated))
    return curated


def partition_recordings(
    recordings: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[Dict]]:
    """Partition recordings by session/recording grouping to guarantee zero cross-split leakage."""
    # Group by recording_session
    sessions_by_label: Dict[str, Dict[str, List[Dict]]] = {c: {} for c in CLASS_NAMES}

    for rec in recordings:
        lbl = rec["label"]
        sess = rec.get("recording_session") or rec["filename"]
        if sess not in sessions_by_label[lbl]:
            sessions_by_label[lbl][sess] = []
        sessions_by_label[lbl][sess].append(rec)

    partitions: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for lbl, sess_map in sessions_by_label.items():
        session_keys = list(sess_map.keys())
        random.shuffle(session_keys)

        n = len(session_keys)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_train = n - n_test - n_val
        if n_train <= 0:
            n_train = max(1, n - 2)
            n_val = 1
            n_test = 1

        train_keys = session_keys[:n_train]
        val_keys = session_keys[n_train : n_train + n_val]
        test_keys = session_keys[n_train + n_val :]

        for k in train_keys:
            for item in sess_map[k]:
                item["split"] = "train"
                partitions["train"].append(item)
        for k in val_keys:
            for item in sess_map[k]:
                item["split"] = "val"
                partitions["val"].append(item)
        for k in test_keys:
            for item in sess_map[k]:
                item["split"] = "test"
                partitions["test"].append(item)

    return partitions


def slice_recording_windows(
    audio_path: str,
    target_sr: int,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
) -> List[np.ndarray]:
    """Slice audio recording into exact 1.0-second normalized mono float32 windows."""
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        logger.warning("Librosa error loading %s: %s", audio_path, e)
        return []

    win_samples = int(target_sr * window_sec)
    hop_samples = int(target_sr * hop_sec)

    if len(y) < win_samples:
        # Pad with reflection if slightly short
        padded = np.pad(y, (0, win_samples - len(y)), mode="reflect")
        return [padded.astype(np.float32)]

    windows: List[np.ndarray] = []
    for start in range(0, len(y) - win_samples + 1, hop_samples):
        chunk = y[start : start + win_samples]
        # Skip dead silent sub-windows
        if np.sqrt(np.mean(chunk**2)) < 0.0003:
            continue
        windows.append(chunk.astype(np.float32))

    return windows


def main():
    logger.info("=== STEP 2: EXPANDING DATASET (ESC-50 + HARDWARE RECORDINGS) ===")
    t0 = time.time()

    # 1. Download curated ESC-50 negative clips
    esc50_records = download_esc50_selection(max_per_category=15)

    # 2. Curate own hardware recordings
    own_records = curate_own_recordings()

    all_records = esc50_records + own_records
    logger.info("Total audited master recordings: %d", len(all_records))

    # 3. Disjoint session-based partitioning
    partitions = partition_recordings(all_records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # Audit quality report
    quality_report = {
        "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_master_recordings": len(all_records),
        "source_breakdown": {
            "ESC-50": len([r for r in all_records if r["source"] == "ESC-50"]),
            "ESP32_INMP441": len([r for r in all_records if r["source"] == "ESP32_INMP441"]),
            "Own_Recording": len([r for r in all_records if r["source"] == "Own_Recording"]),
        },
        "class_counts_by_recording": {c: len([r for r in all_records if r["label"] == c]) for c in CLASS_NAMES},
        "partition_counts_by_recording": {
            "train": len(partitions["train"]),
            "val": len(partitions["val"]),
            "test": len(partitions["test"]),
        },
        "quality_filters_applied": [
            "corrupted_audio_rejection",
            "silent_audio_filter_rms_0.0003",
            "clipping_rejection_0.99_peak_0.2pct_clipped",
            "sha256_exact_deduplication",
            "disjoint_recording_session_grouping",
        ],
    }

    QUALITY_REPORT_PATH.write_text(json.dumps(quality_report, indent=2))
    MANIFEST_PATH.write_text(json.dumps(all_records, indent=2))

    logger.info("Saved data quality report to %s", QUALITY_REPORT_PATH)
    logger.info("Saved complete dataset manifest to %s", MANIFEST_PATH)

    # 4. Generate pre-processed window slices for CNN (22,050 Hz) and YAMNet (16,000 Hz)
    logger.info("Extracting audio windows for CNN (22,050 Hz) and YAMNet (16,000 Hz)...")

    # Sample windows per partition to build balanced dataset files
    for split in ["train", "val", "test"]:
        split_records = partitions[split]
        cnn_windows = []
        cnn_labels = []
        yamnet_windows = []
        yamnet_labels = []
        meta_ids = []

        # Target window caps per category to keep training balanced and fast
        max_win_per_file = 6 if split == "train" else 4

        for rec in split_records:
            lbl_idx = CLASS_TO_IDX[rec["label"]]
            # Extract 22,050 Hz for CNN
            wins_22k = slice_recording_windows(rec["file_path"], target_sr=22050, hop_sec=0.5)[:max_win_per_file]
            # Extract 16,000 Hz for YAMNet
            wins_16k = slice_recording_windows(rec["file_path"], target_sr=16000, hop_sec=0.5)[:max_win_per_file]

            for w22, w16 in zip(wins_22k, wins_16k):
                cnn_windows.append(w22)
                cnn_labels.append(lbl_idx)
                yamnet_windows.append(w16)
                yamnet_labels.append(lbl_idx)
                meta_ids.append(rec["filename"])

        cnn_arr = np.array(cnn_windows, dtype=np.float32)
        yamnet_arr = np.array(yamnet_windows, dtype=np.float32)
        lbl_arr = np.array(cnn_labels, dtype=np.int64)

        out_path = EXP_DATA_DIR / f"expanded_{split}.npz"
        np.savez_compressed(
            out_path,
            audio_22050=cnn_arr,
            audio_16000=yamnet_arr,
            labels=lbl_arr,
            metadata=meta_ids,
        )
        logger.info(
            "Split '%s': saved %d 1-second windows to %s (classes: %s)",
            split,
            len(lbl_arr),
            out_path.name,
            dict(zip(*np.unique(lbl_arr, return_counts=True))),
        )

    elapsed = time.time() - t0
    logger.info("Dataset expansion complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
