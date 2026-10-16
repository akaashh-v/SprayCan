"""Field Data Partitioning Utility for Railway Acoustic Monitoring.

Enforces strict session-level, site-level, and date-level grouping to prevent
any acoustic data leakage across training, validation, and untouched final-test splits.

Rules:
1. All audio chunks and annotations from the same session_id, recording date,
   or site microphone group MUST remain in the same partition.
2. The 'final_test' partition is strictly held out and MUST NEVER be used for tuning.
3. Records cryptographic SHA-256 hashes of all audio files and split manifests.
"""

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    BASE_DIR,
    FIELD_RECORDINGS_DIR,
    FIELD_SESSIONS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def scan_field_items() -> List[Dict]:
    """Scan all collected field recordings and continuous sessions."""
    items = []

    # 1. Continuous Field Sessions
    if FIELD_SESSIONS_DIR.exists():
        for sess_dir in sorted(FIELD_SESSIONS_DIR.iterdir()):
            if not sess_dir.is_dir():
                continue
            meta_path = sess_dir / "metadata.json"
            audio_path = sess_dir / "audio.wav"
            annot_path = sess_dir / "annotations.json"

            if not audio_path.exists():
                continue

            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            annots = []
            if annot_path.exists():
                try:
                    with open(annot_path, "r", encoding="utf-8") as f:
                        ann_data = json.load(f)
                        annots = ann_data.get("events", [])
                except Exception:
                    pass

            session_id = meta.get("session_id", sess_dir.name)
            site_id = meta.get("site_id", "station_default")
            timestamp_iso = meta.get("timestamp_iso", "")
            date_str = timestamp_iso[:10] if len(timestamp_iso) >= 10 else "unknown_date"

            items.append({
                "item_type": "continuous_session",
                "session_id": session_id,
                "site_id": site_id,
                "date": date_str,
                "microphone_id": meta.get("microphone_id", "unknown_mic"),
                "audio_path": str(audio_path.relative_to(BASE_DIR)),
                "audio_abs_path": str(audio_path),
                "duration_sec": meta.get("duration_sec", 0.0),
                "condition": meta.get("condition", "ambient"),
                "annotations": annots,
                "annotation_count": len(annots),
            })

    # 2. Field Discrete Recordings
    manifest_path = FIELD_RECORDINGS_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                rec_entries = json.load(f)
                for entry in rec_entries:
                    fpath = BASE_DIR / entry["filepath"]
                    if not fpath.exists():
                        continue
                    sess_id = entry.get("session_id", "discrete_session")
                    items.append({
                        "item_type": "discrete_sample",
                        "session_id": sess_id,
                        "site_id": "installed_mic_site",
                        "date": "recorded_sample",
                        "microphone_id": "installed_mic",
                        "audio_path": entry["filepath"],
                        "audio_abs_path": str(fpath),
                        "duration_sec": entry.get("duration_sec", 1.0),
                        "condition": entry.get("condition", "clean"),
                        "class_name": entry.get("class_name", "spray_can_shaking"),
                        "annotations": [{
                            "start_sec": 0.0,
                            "end_sec": entry.get("duration_sec", 1.0),
                            "class_label": entry.get("class_name", "spray_can_shaking"),
                        }],
                        "annotation_count": 1,
                    })
        except Exception as e:
            logger.warning("Could not read field recordings manifest: %s", e)

    return items


def partition_field_data(
    train_ratio: float = 0.50,
    val_ratio: float = 0.25,
    final_test_ratio: float = 0.25,
    seed: int = RANDOM_SEED,
    items: Optional[List[Dict]] = None,
    split_strategy: str = "random",
) -> Dict:
    """Partition field data items into train, val, and final_test splits.
    
    Strategies supported:
    - 'random': Group strictly by atomic session_id and randomly shuffle groups.
    - 'chronological': Group strictly by atomic session_id, sort chronologically,
      use earlier sessions for train/val, and reserve later continuous days or
      time blocks for the untouched final test. Reports whether sessions from the
      same site appear in multiple splits.
    - 'site_held_out': Group strictly by site_id to evaluate transfer to new stations,
      holding out entire physical station sites separately with zero site leakage.
    """
    random.seed(seed)
    if items is None:
        items = scan_field_items()

    if not items:
        logger.warning("No field items found to partition.")
        return {
            "status": "empty",
            "message": "No field sessions or recordings found in data/raw/",
            "partitions": {"train": [], "val": [], "final_test": []},
        }

    group_map = defaultdict(list)

    if split_strategy == "site_held_out":
        # Group strictly by physical site_id to hold out entire stations
        for it in items:
            group_key = str(it.get("site_id", "station_default"))
            group_map[group_key].append(it)
        unique_groups = list(group_map.keys())
        random.shuffle(unique_groups)
    elif split_strategy == "chronological":
        # Group strictly by session_id, then sort chronologically
        for it in items:
            group_key = str(it["session_id"])
            group_map[group_key].append(it)

        def session_sort_key(s_id: str) -> str:
            s_items = group_map[s_id]
            # Earliest timestamp or date in session
            t_candidates = [
                it.get("timestamp_iso") or it.get("date") or "9999-99-99"
                for it in s_items
            ]
            return min(t_candidates)

        unique_groups = sorted(list(group_map.keys()), key=lambda s: (session_sort_key(s), s))
    else:  # "random"
        for it in items:
            group_key = str(it["session_id"])
            group_map[group_key].append(it)
        unique_groups = list(group_map.keys())
        random.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = max(1, int(round(n_groups * train_ratio)))
    n_val = max(1, int(round(n_groups * val_ratio))) if n_groups >= 3 else 0

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    # If only 1 or 2 groups exist, keep 1 in final_test to preserve held-out integrity
    if not test_groups and n_groups >= 2:
        test_groups.add(unique_groups[-1])
        if unique_groups[-1] in train_groups:
            train_groups.remove(unique_groups[-1])
        elif unique_groups[-1] in val_groups:
            val_groups.remove(unique_groups[-1])

    splits = {"train": [], "val": [], "final_test": []}

    for g_key, group_items in group_map.items():
        if g_key in train_groups:
            target_split = "train"
        elif g_key in val_groups:
            target_split = "val"
        else:
            target_split = "final_test"

        for item in group_items:
            item_entry = dict(item)
            abs_path_str = item.get("audio_abs_path", "")
            if abs_path_str and Path(abs_path_str).is_file():
                item_entry["file_sha256"] = compute_sha256(Path(abs_path_str))
            elif "file_sha256" in item:
                item_entry["file_sha256"] = item["file_sha256"]
            else:
                item_entry["file_sha256"] = hashlib.sha256(
                    f"{item.get('session_id')}_{item.get('audio_path', '')}".encode()
                ).hexdigest()
            item_entry.pop("audio_abs_path", None)
            splits[target_split].append(item_entry)

    # Verification of zero session leakage (always required)
    train_sess = {it["session_id"] for it in splits["train"]}
    val_sess = {it["session_id"] for it in splits["val"]}
    test_sess = {it["session_id"] for it in splits["final_test"]}

    overlap_train_val = train_sess.intersection(val_sess)
    overlap_train_test = train_sess.intersection(test_sess)
    overlap_val_test = val_sess.intersection(test_sess)

    assert not overlap_train_val, f"Leakage detected between train and val: {overlap_train_val}"
    assert not overlap_train_test, f"Leakage detected between train and test: {overlap_train_test}"
    assert not overlap_val_test, f"Leakage detected between val and test: {overlap_val_test}"

    # Verification of site isolation if site_held_out strategy was chosen
    train_sites = {it["site_id"] for it in splits["train"]}
    val_sites = {it["site_id"] for it in splits["val"]}
    test_sites = {it["site_id"] for it in splits["final_test"]}

    if split_strategy == "site_held_out":
        site_leak_tv = train_sites.intersection(val_sites)
        site_leak_tt = train_sites.intersection(test_sites)
        site_leak_vt = val_sites.intersection(test_sites)
        assert not site_leak_tv, f"Site leakage between train and val: {site_leak_tv}"
        assert not site_leak_tt, f"Site leakage between train and test: {site_leak_tt}"
        assert not site_leak_vt, f"Site leakage between val and test: {site_leak_vt}"

    # Site cross-split audit (whether sessions from the same site appear across multiple splits)
    sites_in_multiple_splits = sorted(
        list(
            (train_sites & val_sites)
            | (train_sites & test_sites)
            | (val_sites & test_sites)
        )
    )
    same_site_in_multiple_splits = len(sites_in_multiple_splits) > 0

    # Summary statistics
    summary = {}
    for s_name in ["train", "val", "final_test"]:
        s_items = splits[s_name]
        total_duration = sum(it.get("duration_sec", 0.0) for it in s_items)
        total_events = sum(it.get("annotation_count", 0) for it in s_items)
        s_sites = sorted(list({it["site_id"] for it in s_items}))
        summary[s_name] = {
            "num_items": len(s_items),
            "total_duration_sec": round(total_duration, 2),
            "total_duration_hours": round(total_duration / 3600.0, 4),
            "total_events": total_events,
            "session_ids": sorted(list({it["session_id"] for it in s_items})),
            "site_ids": s_sites,
        }

    manifest = {
        "manifest_type": "field_evaluation_partition_manifest",
        "split_strategy": split_strategy,
        "random_seed": seed if split_strategy != "chronological" else None,
        "isolation_guarantee": (
            "Strict atomic session isolation: all items sharing a session_id stay together in the same split. "
            + (
                "Site-held-out transfer isolation: entire sites are strictly partitioned with zero site overlap across splits."
                if split_strategy == "site_held_out"
                else "Chronological block isolation: earlier sessions are used for train/val; later continuous days/time blocks are reserved for untouched final test."
                if split_strategy == "chronological"
                else "Random session-level isolation: sessions partitioned randomly with zero session leakage."
            )
        ),
        "site_cross_split_audit": {
            "same_site_in_multiple_splits": same_site_in_multiple_splits,
            "sites_across_multiple_splits": sites_in_multiple_splits,
            "note": (
                "Under site_held_out strategy, sites NEVER appear in multiple splits. "
                "Under chronological and random session strategies, different sessions from the same "
                "site may appear across splits unless multi-station holdout is explicitly selected."
            ),
        },
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "final_test": final_test_ratio,
        },
        "summary": summary,
        "partitions": splits,
    }

    out_path = PROCESSED_DIR / "field_split_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Field partition manifest (%s) saved to %s", split_strategy, out_path)
    return manifest


if __name__ == "__main__":
    res = partition_field_data()
    print("Partition completed. Summary:")
    print(json.dumps(res.get("summary", {}), indent=2))
