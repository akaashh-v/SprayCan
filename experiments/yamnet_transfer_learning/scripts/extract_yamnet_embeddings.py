"""Extract and cache frozen YAMNet embeddings from expanded dataset partitions.

Implements Step 4 requirements:
1. Uses official pretrained YAMNet implementation from TensorFlow Hub.
2. Input format: Mono float32 waveforms sampled at 16,000 Hz, normalized [-1, 1].
3. Extracts 1,024-dimensional embeddings per frame, mean-pooled across the 1.0s window.
4. Caches extracted embeddings to disk (yamnet_train.npz, yamnet_val.npz, yamnet_test.npz).
5. Saves extraction metadata, configuration, class map, and normalization details.
"""

import json
import logging
from pathlib import Path
import time
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yamnet_extractor")

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = EXP_DATA_DIR / "yamnet_embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "ambient_and_speech",
    "loud_impact",
    "continuous_environmental",
    "hard_negative_rattle",
    "aerosol_spray_hiss",
    "spray_can_shaking",
]

YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000


def load_yamnet_model():
    """Load official YAMNet model from TensorFlow Hub."""
    logger.info("Loading YAMNet model from %s...", YAMNET_HUB_URL)
    yamnet = hub.load(YAMNET_HUB_URL)
    logger.info("YAMNet model loaded successfully.")
    return yamnet


def extract_embeddings_for_split(
    yamnet,
    audio_16k: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 1024-dimensional mean-pooled YAMNet embeddings for each 1.0s clip."""
    num_samples = len(audio_16k)
    logger.info("Extracting YAMNet embeddings for %d clips (batch size = %d)...", num_samples, batch_size)

    embeddings_list = []

    for i in range(num_samples):
        waveform = audio_16k[i].astype(np.float32)
        # Ensure [-1, 1] bounds
        max_abs = np.max(np.abs(waveform))
        if max_abs > 1.0:
            waveform = waveform / max_abs

        # YAMNet inference on single waveform
        # Output: scores (N, 521), embeddings (N, 1024), log_mel_spectrogram (N_frames, 64)
        _, clip_embeddings, _ = yamnet(waveform)

        # Mean pooling across temporal frames for this 1.0-sec window -> (1024,)
        pooled = tf.reduce_mean(clip_embeddings, axis=0).numpy()
        embeddings_list.append(pooled)

        if (i + 1) % 500 == 0 or (i + 1) == num_samples:
            logger.info("  Processed %d / %d clips...", i + 1, num_samples)

    features = np.array(embeddings_list, dtype=np.float32)
    return features, labels


def main():
    logger.info("=== STEP 4A: EXTRACTING & CACHING YAMNET EMBEDDINGS ===")
    t0 = time.time()

    # Verify input data exists
    train_npz = EXP_DATA_DIR / "expanded_train.npz"
    val_npz = EXP_DATA_DIR / "expanded_val.npz"
    test_npz = EXP_DATA_DIR / "expanded_test.npz"

    if not (train_npz.exists() and val_npz.exists() and test_npz.exists()):
        raise FileNotFoundError("Missing expanded dataset partitions in " + str(EXP_DATA_DIR))

    yamnet = load_yamnet_model()

    # Process all three partitions
    splits = [
        ("train", train_npz, EMBEDDINGS_DIR / "yamnet_train.npz"),
        ("val", val_npz, EMBEDDINGS_DIR / "yamnet_val.npz"),
        ("test", test_npz, EMBEDDINGS_DIR / "yamnet_test.npz"),
    ]

    metadata = {
        "extractor": "YAMNet (TF Hub: https://tfhub.dev/google/yamnet/1)",
        "input_sample_rate": SAMPLE_RATE,
        "input_audio_length_samples": SAMPLE_RATE,
        "input_normalization": "float32 in [-1.0, 1.0]",
        "embedding_dimension": 1024,
        "temporal_pooling": "mean across temporal frames",
        "class_mapping": {i: name for i, name in enumerate(CLASS_NAMES)},
        "splits": {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for split_name, src_path, dst_path in splits:
        logger.info("Processing split: %s (%s)...", split_name, src_path.name)
        data = np.load(src_path)
        audio_16k = data["audio_16000"]
        labels = data["labels"]

        features, targets = extract_embeddings_for_split(yamnet, audio_16k, labels)

        # Save cached embeddings
        np.savez_compressed(
            dst_path,
            features=features,
            labels=targets,
        )
        logger.info("Saved %s embeddings: features shape %s to %s", split_name, features.shape, dst_path.name)

        metadata["splits"][split_name] = {
            "num_windows": len(targets),
            "features_shape": list(features.shape),
            "file": dst_path.name,
            "class_distribution": {
                CLASS_NAMES[c]: int(np.sum(targets == c)) for c in range(len(CLASS_NAMES))
            },
        }

    # Save metadata JSON
    config_path = EMBEDDINGS_DIR / "yamnet_embedding_config.json"
    config_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved embedding config metadata to %s", config_path)

    elapsed = time.time() - t0
    logger.info("YAMNet feature extraction completed in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
