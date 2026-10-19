"""Train a 6-Class Transfer-Learning Classifier on Frozen YAMNet Embeddings.

Implements Step 4 requirements:
1. Input: 1,024-dimensional frozen YAMNet embeddings from yamnet_embeddings/yamnet_train.npz.
2. Architecture: Small MLP head with 6 outputs matching exact class order:
   0: ambient_and_speech
   1: loud_impact
   2: continuous_environmental
   3: hard_negative_rattle
   4: aerosol_spray_hiss
   5: spray_can_shaking
3. Evaluates on held-out recording-level test partition (yamnet_test.npz).
4. Saves checkpoint to models/model_c_yamnet_classifier/yamnet_classifier_best.keras.
5. Generates detailed evaluation report with per-class metrics, false positive rates, and missed shaking events.
"""

import json
import logging
from pathlib import Path
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_yamnet_classifier")

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "yamnet_embeddings"
MODEL_C_DIR = BASE_DIR / "models" / "model_c_yamnet_classifier"
MODEL_C_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "ambient_and_speech",
    "loud_impact",
    "continuous_environmental",
    "hard_negative_rattle",
    "aerosol_spray_hiss",
    "spray_can_shaking",
]
NUM_CLASSES = len(CLASS_NAMES)


def build_yamnet_classifier_model(input_dim: int = 1024, num_classes: int = NUM_CLASSES) -> keras.Model:
    """Construct lightweight classification head on top of YAMNet embeddings."""
    inputs = keras.Input(shape=(input_dim,), name="yamnet_embedding_input")
    x = layers.Dense(128, name="classifier_dense1")(inputs)
    x = layers.BatchNormalization(name="classifier_bn1")(x)
    x = layers.ReLU(name="classifier_relu1")(x)
    x = layers.Dropout(0.35, name="classifier_drop1")(x)

    x = layers.Dense(64, name="classifier_dense2")(x)
    x = layers.BatchNormalization(name="classifier_bn2")(x)
    x = layers.ReLU(name="classifier_relu2")(x)
    x = layers.Dropout(0.25, name="classifier_drop2")(x)

    logits = layers.Dense(num_classes, name="classifier_logits")(x)
    probs = layers.Softmax(name="probabilities")(logits)

    model = keras.Model(inputs=inputs, outputs=probs, name="ModelC_YAMNetClassifier")
    return model


def main():
    logger.info("=== STEP 4B: TRAINING YAMNET TRANSFER-LEARNING CLASSIFIER ===")
    t0 = time.time()

    train_npz = EMBEDDINGS_DIR / "yamnet_train.npz"
    val_npz = EMBEDDINGS_DIR / "yamnet_val.npz"
    test_npz = EMBEDDINGS_DIR / "yamnet_test.npz"

    if not (train_npz.exists() and val_npz.exists() and test_npz.exists()):
        raise FileNotFoundError("Cached YAMNet embeddings missing. Run extract_yamnet_embeddings.py first.")

    train_data = np.load(train_npz)
    val_data = np.load(val_npz)
    test_data = np.load(test_npz)

    x_train, y_train = train_data["features"], train_data["labels"]
    x_val, y_val = val_data["features"], val_data["labels"]
    x_test, y_test = test_data["features"], test_data["labels"]

    logger.info("Loaded embeddings: Train=%s, Val=%s, Test=%s", x_train.shape, x_val.shape, x_test.shape)

    # Compute balanced class weights with conservative clipping
    raw_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    clipped_weights = np.clip(raw_weights, 0.4, 3.0)
    class_weight_dict = {i: float(clipped_weights[i]) for i in range(len(clipped_weights))}
    logger.info("Class weight dictionary: %s", class_weight_dict)

    model = build_yamnet_classifier_model(input_dim=x_train.shape[1], num_classes=NUM_CLASSES)
    model.summary(print_fn=logger.info)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = MODEL_C_DIR / "yamnet_classifier_best.keras"
    weights_path = MODEL_C_DIR / "yamnet_classifier_best.weights.h5"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
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

    logger.info("Training YAMNet classifier for up to 50 epochs...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Save full model checkpoint
    model.save(str(checkpoint_path))
    logger.info("Saved Model C classifier to %s", checkpoint_path)

    # Evaluate on held-out test split
    test_probs = model.predict(x_test)
    test_preds = np.argmax(test_probs, axis=1)

    macro_f1 = f1_score(y_test, test_preds, average="macro")
    acc = np.mean(test_preds == y_test)
    cm = confusion_matrix(y_test, test_preds, labels=list(range(NUM_CLASSES)))

    report_dict = classification_report(
        y_test,
        test_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Calculate false positive and missed event metrics specifically for spray_can_shaking (Class 5)
    target_idx = 5
    true_shaking_mask = (y_test == target_idx)
    non_shaking_mask = ~true_shaking_mask

    total_shaking = int(np.sum(true_shaking_mask))
    detected_shaking = int(np.sum((test_preds == target_idx) & true_shaking_mask))
    missed_shaking = total_shaking - detected_shaking
    missed_shaking_rate = (missed_shaking / total_shaking) if total_shaking > 0 else 0.0

    false_alarms = int(np.sum((test_preds == target_idx) & non_shaking_mask))
    total_non_shaking = int(np.sum(non_shaking_mask))
    false_alarm_rate = (false_alarms / total_non_shaking) if total_non_shaking > 0 else 0.0

    # Specifically for hard negative rattle (Class 3)
    hard_neg_mask = (y_test == 3)
    hard_neg_total = int(np.sum(hard_neg_mask))
    hard_neg_false_alarms = int(np.sum((test_preds == target_idx) & hard_neg_mask))

    eval_results = {
        "model_name": "Model C (YAMNet Transfer Learning)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "spray_can_shaking_metrics": {
            "total_test_events": total_shaking,
            "detected_events": detected_shaking,
            "missed_events": missed_shaking,
            "missed_event_rate": float(missed_shaking_rate),
            "false_alarms_total": false_alarms,
            "false_alarm_rate": float(false_alarm_rate),
            "hard_negative_rattle_false_alarms": hard_neg_false_alarms,
            "hard_negative_rattle_total": hard_neg_total,
        },
        "epochs_trained": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "checkpoint_path": str(checkpoint_path),
    }

    eval_path = MODEL_C_DIR / "yamnet_eval_report.json"
    eval_path.write_text(json.dumps(eval_results, indent=2))
    logger.info("Saved Model C evaluation report to %s", eval_path)

    print("\n" + "=" * 60)
    print("MODEL C (YAMNET TRANSFER LEARNING) TEST EVALUATION RESULTS:")
    print(f"Overall Accuracy:  {acc*100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Shaking Miss Rate: {missed_shaking_rate*100:.2f}% ({missed_shaking}/{total_shaking})")
    print(f"False Alarm Rate:  {false_alarm_rate*100:.2f}% ({false_alarms}/{total_non_shaking})")
    print(f"Hard Negative FA:  {hard_neg_false_alarms}/{hard_neg_total}")
    print("=" * 60)
    for c in CLASS_NAMES:
        p = report_dict[c]["precision"]
        r = report_dict[c]["recall"]
        f1 = report_dict[c]["f1-score"]
        print(f"  {c:28s}: P={p*100:5.1f}% | R={r*100:5.1f}% | F1={f1:.4f}")
    print("=" * 60)

    elapsed = time.time() - t0
    logger.info("Experiment C training completed in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
