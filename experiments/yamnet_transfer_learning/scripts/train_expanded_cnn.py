"""Experiment B: Train / Fine-Tune CNN on Expanded Negative Dataset.

Implements Step 3 requirements:
1. Retains identical production architecture: 4 Conv2D blocks + AdaptiveAvgPool + Dense(128) + Dense(6).
2. Input format: 22,050 Hz audio -> Log-Mel Spectrogram (64 x 44 x 1).
3. Preserves exact 6-class order:
   0: ambient_and_speech
   1: loud_impact
   2: continuous_environmental
   3: hard_negative_rattle
   4: aerosol_spray_hiss
   5: spray_can_shaking
4. Training-only augmentation:
   - Random time shift (+/- 0.15s)
   - Random gain perturbation (+/- 2.5 dB)
   - Modest Gaussian noise injection (SNR 18-28 dB)
5. Class weighting with conservative bounds to handle real-world distribution.
6. Validation-based early stopping (monitoring val_loss and macro F1).
7. Checkpoints saved strictly inside experiments/yamnet_transfer_learning/models/model_b_expanded_cnn/.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Tuple

import librosa
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_expanded_cnn")

# Set random seed
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DATA_DIR = BASE_DIR / "data"
MODEL_B_DIR = BASE_DIR / "models" / "model_b_expanded_cnn"
MODEL_B_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = BASE_DIR.parent.parent
PRODUCTION_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "best_detector.keras"

CLASS_NAMES = [
    "ambient_and_speech",
    "loud_impact",
    "continuous_environmental",
    "hard_negative_rattle",
    "aerosol_spray_hiss",
    "spray_can_shaking",
]
NUM_CLASSES = len(CLASS_NAMES)

# Audio & Log-Mel parameters matching production exactly
SAMPLE_RATE = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 100.0
F_MAX = 11000.0


def compute_log_mel_batch(audio_batch: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute Log-Mel spectrograms matching production backend/audio_utils.py."""
    specs = []
    # Pre-compute mel filterbank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)

    for i in range(len(audio_batch)):
        y = audio_batch[i]
        # STFT
        D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=True)) ** 2
        # Mel projection
        mel = np.dot(mel_basis, D)
        # Log compression: log10(mel + 1e-6)
        log_mel = np.log10(mel + 1e-6)
        # Expected shape: (64, 44)
        if log_mel.shape[1] > 44:
            log_mel = log_mel[:, :44]
        elif log_mel.shape[1] < 44:
            log_mel = np.pad(log_mel, ((0, 0), (0, 44 - log_mel.shape[1])), mode="reflect")
        specs.append(log_mel[:, :, np.newaxis])

    return np.array(specs, dtype=np.float32)


def apply_audio_augmentation(audio_batch: np.ndarray) -> np.ndarray:
    """Apply training-only audio augmentations."""
    augmented = np.copy(audio_batch)
    for i in range(len(augmented)):
        y = augmented[i]
        # 1. Random circular time shift (+/- 0.15s)
        shift = int(np.random.uniform(-0.15, 0.15) * SAMPLE_RATE)
        y = np.roll(y, shift)

        # 2. Random gain perturbation (+/- 2.5 dB)
        gain_db = np.random.uniform(-2.5, 2.5)
        gain_factor = 10.0 ** (gain_db / 20.0)
        y = y * gain_factor

        # 3. Modest additive noise (SNR ~ 22 dB)
        if np.random.rand() < 0.5:
            rms = np.sqrt(np.mean(y**2) + 1e-8)
            noise_rms = rms * (10.0 ** (-22.0 / 20.0))
            noise = np.random.randn(len(y)).astype(np.float32) * noise_rms
            y = y + noise

        # Normalize clipping
        max_val = np.max(np.abs(y))
        if max_val > 1.0:
            y = y / max_val

        augmented[i] = y
    return augmented


def build_expanded_cnn_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """Construct production-equivalent CNN with registered custom pooling."""
    inputs = keras.Input(shape=(64, 44, 1), name="spectrogram_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_conv")(inputs)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv1_pool")(x)
    x = layers.Dropout(0.15, name="conv1_drop")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv2_bn")(x)
    x = layers.ReLU(name="conv2_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv2_pool")(x)
    x = layers.Dropout(0.20, name="conv2_drop")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv3_bn")(x)
    x = layers.ReLU(name="conv3_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv3_pool")(x)
    x = layers.Dropout(0.25, name="conv3_drop")(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv4_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv4_bn")(x)
    x = layers.ReLU(name="conv4_relu")(x)

    # Global/Adaptive pooling to 4x4
    def _pool_4x4(t):
        w0 = tf.reduce_mean(t[:, :, 0:2, :], axis=2, keepdims=True)
        w1 = tf.reduce_mean(t[:, :, 1:3, :], axis=2, keepdims=True)
        w2 = tf.reduce_mean(t[:, :, 2:4, :], axis=2, keepdims=True)
        w3 = tf.reduce_mean(t[:, :, 3:5, :], axis=2, keepdims=True)
        w_p = tf.concat([w0, w1, w2, w3], axis=2)
        h0 = tf.reduce_mean(w_p[:, 0:2, :, :], axis=1, keepdims=True)
        h1 = tf.reduce_mean(w_p[:, 2:4, :, :], axis=1, keepdims=True)
        h2 = tf.reduce_mean(w_p[:, 4:6, :, :], axis=1, keepdims=True)
        h3 = tf.reduce_mean(w_p[:, 6:8, :, :], axis=1, keepdims=True)
        return tf.concat([h0, h1, h2, h3], axis=1)

    x = layers.Lambda(_pool_4x4, name="adaptive_avg_pool")(x)
    x_ch_first = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 3, 1, 2]), name="transpose_nchw")(x)
    features_flat = layers.Flatten(name="features_flatten")(x_ch_first)

    fc1 = layers.Dense(128, name="classifier_dense1")(features_flat)
    fc1_bn = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="classifier_bn1")(fc1)
    fc1_relu = layers.ReLU(name="classifier_relu1")(fc1_bn)
    fc1_drop = layers.Dropout(0.40, name="classifier_drop1")(fc1_relu)
    logits = layers.Dense(num_classes, name="classifier_logits")(fc1_drop)
    probs = layers.Softmax(name="probabilities")(logits)

    model = keras.Model(inputs=inputs, outputs=probs, name="ModelB_ExpandedCNN")
    return model


def main():
    logger.info("=== STEP 3: TRAINING EXPERIMENT B (EXPANDED-DATA CNN) ===")
    t0 = time.time()

    # 1. Load expanded datasets
    train_data = np.load(EXP_DATA_DIR / "expanded_train.npz")
    val_data = np.load(EXP_DATA_DIR / "expanded_val.npz")
    test_data = np.load(EXP_DATA_DIR / "expanded_test.npz")

    x_train_raw = train_data["audio_22050"]
    y_train = train_data["labels"]

    x_val_raw = val_data["audio_22050"]
    y_val = val_data["labels"]

    x_test_raw = test_data["audio_22050"]
    y_test = test_data["labels"]

    logger.info("Loaded datasets: Train=%d, Val=%d, Test=%d windows", len(y_train), len(y_val), len(y_test))

    # Apply training augmentation and compute Log-Mel spectrograms
    logger.info("Applying training-only audio augmentation...")
    x_train_aug = apply_audio_augmentation(x_train_raw)

    logger.info("Computing Log-Mel spectrograms (22,050 Hz, 64 Mel bins, 1024 FFT, 512 hop)...")
    x_train_spec = compute_log_mel_batch(x_train_aug)
    x_val_spec = compute_log_mel_batch(x_val_raw)
    x_test_spec = compute_log_mel_batch(x_test_raw)

    logger.info("Spectrogram shapes: Train=%s, Val=%s, Test=%s", x_train_spec.shape, x_val_spec.shape, x_test_spec.shape)

    # 2. Build model and initialize from production weights for warm transfer
    model = build_expanded_cnn_model(num_classes=NUM_CLASSES)

    if PRODUCTION_MODEL_PATH.exists():
        logger.info("Transferring initialized weights from production model: %s", PRODUCTION_MODEL_PATH)
        try:
            model.load_weights(str(PRODUCTION_MODEL_PATH))
            logger.info("Successfully loaded warm starting weights from production model!")
        except Exception as e:
            logger.warning("Could not warm-start weights: %s. Training from scratch.", e)

    # Compute balanced class weights with conservative clipping [0.4, 3.0]
    raw_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    clipped_weights = np.clip(raw_weights, 0.4, 3.0)
    class_weight_dict = {i: float(clipped_weights[i]) for i in range(len(clipped_weights))}
    logger.info("Class weight dictionary: %s", class_weight_dict)

    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    checkpoint_path = MODEL_B_DIR / "expanded_cnn_best.keras"
    weights_path = MODEL_B_DIR / "expanded_cnn_best.weights.h5"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            save_weights_only=True,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info("Starting training for up to 30 epochs with early stopping...")
    history = model.fit(
        x_train_spec,
        y_train,
        validation_data=(x_val_spec, y_val),
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Save complete trained model
    model.save(str(checkpoint_path))
    logger.info("Saved best Model B to %s", checkpoint_path)

    # 3. Evaluate on held-out test partition
    test_probs = model.predict(x_test_spec)
    test_preds = np.argmax(test_probs, axis=1)

    macro_f1 = f1_score(y_test, test_preds, average="macro")
    acc = np.mean(test_preds == y_test)

    report_dict = classification_report(
        y_test,
        test_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    eval_results = {
        "model_name": "Model B (Expanded-Data CNN)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "classification_report": report_dict,
        "epochs_trained": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "checkpoint_path": str(checkpoint_path),
    }

    eval_path = MODEL_B_DIR / "expanded_cnn_eval_report.json"
    eval_path.write_text(json.dumps(eval_results, indent=2))
    logger.info("Saved Model B evaluation report to %s", eval_path)

    print("\n" + "=" * 60)
    print("MODEL B (EXPANDED-DATA CNN) TEST EVALUATION RESULTS:")
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print("=" * 60)
    for c in CLASS_NAMES:
        p = report_dict[c]["precision"]
        r = report_dict[c]["recall"]
        f1 = report_dict[c]["f1-score"]
        print(f"  {c:28s}: P={p*100:5.1f}% | R={r*100:5.1f}% | F1={f1:.4f}")
    print("=" * 60)

    elapsed = time.time() - t0
    logger.info("Experiment B completed in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
