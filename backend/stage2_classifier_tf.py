"""Stage 2: Full Spectro-Temporal Deep CNN Classifier (TensorFlow / Keras Implementation).

Matches the architecture, parameters, class order, and mathematical behavior of
SprayCanShakingCNN from backend/stage2_classifier.py:
- Input shape: (Batch, 64, 44, 1) or (Batch, 1, 64, 44) (Log-Mel Spectrogram 64x44)
- 4 Convolutional blocks with BatchNorm, ReLU, MaxPool2D, and Dropout
- Exact PyTorch-equivalent Adaptive Average Pooling to (4, 4)
- 128-dim fully connected bottleneck with BatchNorm1D, ReLU, Dropout(0.40)
- 6-Class Softmax classification output

Preserved Class Order:
  0: ambient_and_speech
  1: loud_impact
  2: continuous_environmental
  3: hard_negative_rattle
  4: aerosol_spray_hiss
  5: spray_can_shaking (Target)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Re-export evaluation and config parameters for complete drop-in compatibility
from config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    CONSERVATIVE_THRESHOLD,
    MAX_COMPETING_CONFIDENCE,
    MAX_ENV_CONFIDENCE,
    MIN_CONFIDENCE_MARGIN,
    NUM_CLASSES,
    UNCERTAIN_ENTROPY_THRESHOLD,
)
from stage2_classifier import evaluate_candidate_prediction

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    layers = None
    TF_AVAILABLE = False


def build_spray_can_cnn_tf(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (64, 44, 1),
    name: str = "SprayCanShakingCNNTF",
) -> "keras.Model":
    """Construct Keras Functional CNN model matching PyTorch SprayCanShakingCNN."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed in the current environment.")

    inputs = keras.Input(shape=input_shape, name="spectrogram_input")

    # Block 1: Micro-transient edge & onset detection
    # PyTorch: Conv2d(1, 32, k=3, p=1), BN(32, eps=1e-5), ReLU, MaxPool2d(2, 2), Dropout(0.15)
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv1_conv")(inputs)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="conv1_pool")(x)  # -> (32, 22, 32)
    x = layers.Dropout(0.15, name="conv1_drop")(x)

    # Block 2: Frequency band interaction & cavity resonance
    # PyTorch: Conv2d(32, 64, k=3, p=1), BN(64, eps=1e-5), ReLU, MaxPool2d(2, 2), Dropout(0.20)
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv2_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv2_bn")(x)
    x = layers.ReLU(name="conv2_relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="conv2_pool")(x)  # -> (16, 11, 64)
    x = layers.Dropout(0.20, name="conv2_drop")(x)

    # Block 3: Rhythmic temporal pattern & cadence continuity
    # PyTorch: Conv2d(64, 128, k=3, p=1), BN(128, eps=1e-5), ReLU, MaxPool2d(2, 2), Dropout(0.25)
    x = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv3_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv3_bn")(x)
    x = layers.ReLU(name="conv3_relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="conv3_pool")(x)  # -> (8, 5, 128)
    x = layers.Dropout(0.25, name="conv3_drop")(x)

    # Block 4: High-level semantic acoustic shape
    # PyTorch: Conv2d(128, 128, k=3, p=1), BN(128, eps=1e-5), ReLU -> (8, 5, 128)
    x = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv4_conv")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="conv4_bn")(x)
    x = layers.ReLU(name="conv4_relu")(x)

    # Exact PyTorch AdaptiveAvgPool2d((4, 4)) emulation on (B, 8, 5, 128):
    # PyTorch adaptive pooling formulas on 8x5 -> 4x4:
    #   Height indices (8 -> 4): rows [0,1], [2,3], [4,5], [6,7]
    #   Width indices (5 -> 4): cols [0,1], [1,2], [2,3], [3,4]
    def _adaptive_avg_pool_4x4(t):
        # Pool width dimension (axis=2)
        w0 = tf.reduce_mean(t[:, :, 0:2, :], axis=2, keepdims=True)
        w1 = tf.reduce_mean(t[:, :, 1:3, :], axis=2, keepdims=True)
        w2 = tf.reduce_mean(t[:, :, 2:4, :], axis=2, keepdims=True)
        w3 = tf.reduce_mean(t[:, :, 3:5, :], axis=2, keepdims=True)
        w_pooled = tf.concat([w0, w1, w2, w3], axis=2)  # (B, 8, 4, 128)

        # Pool height dimension (axis=1)
        h0 = tf.reduce_mean(w_pooled[:, 0:2, :, :], axis=1, keepdims=True)
        h1 = tf.reduce_mean(w_pooled[:, 2:4, :, :], axis=1, keepdims=True)
        h2 = tf.reduce_mean(w_pooled[:, 4:6, :, :], axis=1, keepdims=True)
        h3 = tf.reduce_mean(w_pooled[:, 6:8, :, :], axis=1, keepdims=True)
        return tf.concat([h0, h1, h2, h3], axis=1)  # (B, 4, 4, 128)

    x = layers.Lambda(_adaptive_avg_pool_4x4, name="adaptive_avg_pool")(x)

    # Transpose (B, H=4, W=4, C=128) -> (B, C=128, H=4, W=4)
    # This guarantees that Keras Flatten() matches PyTorch's (C, H, W) index flattening order!
    x_ch_first = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 3, 1, 2]), name="transpose_nchw")(x)
    features_flat = layers.Flatten(name="features_flatten")(x_ch_first)  # (B, 2048)

    # Classification Head
    # PyTorch: Linear(2048, 128), BN1d(128), ReLU, Dropout(0.40), Linear(128, num_classes)
    fc1 = layers.Dense(128, use_bias=True, name="classifier_dense1")(features_flat)
    fc1_bn = layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9, name="classifier_bn1")(fc1)
    fc1_relu = layers.ReLU(name="classifier_relu1")(fc1_bn)
    fc1_drop = layers.Dropout(0.40, name="classifier_drop1")(fc1_relu)
    logits = layers.Dense(num_classes, use_bias=True, name="classifier_logits")(fc1_drop)
    probs = layers.Softmax(name="probabilities")(logits)

    model = keras.Model(inputs=inputs, outputs=[probs, logits, features_flat], name=name)
    return model


class SprayCanShakingCNNTF:
    """Drop-in TensorFlow/Keras replacement for PyTorch SprayCanShakingCNN."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        weights_path: Optional[Union[str, Path]] = None,
    ):
        if not TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow is not available. Please install tensorflow to use the TF detector backend."
            )
        self.num_classes = num_classes
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = build_spray_can_cnn_tf(num_classes=self.num_classes)
        self._call_graph = tf.function(lambda inp: self.model(inp, training=False), reduce_retracing=True)
        if self.weights_path and self.weights_path.exists():
            self.load_weights(self.weights_path)

    def _prepare_input(self, x: Union[np.ndarray, "tf.Tensor"]) -> "tf.Tensor":
        """Convert input array to shape (B, 64, 44, 1) float32."""
        if hasattr(x, "detach"):  # Handle incoming PyTorch tensor
            x = x.detach().cpu().numpy()
        arr = np.asarray(x, dtype=np.float32)

        if arr.ndim == 2:
            # Single spectrogram: (64, 44) -> (1, 64, 44, 1)
            arr = arr[np.newaxis, :, :, np.newaxis]
        elif arr.ndim == 3:
            # (Batch, 64, 44) -> (Batch, 64, 44, 1)
            arr = arr[:, :, :, np.newaxis]
        elif arr.ndim == 4:
            if arr.shape[1] == 1 and arr.shape[2] == 64 and arr.shape[3] == 44:
                # PyTorch channels_first format (Batch, 1, 64, 44) -> (Batch, 64, 44, 1)
                arr = np.transpose(arr, (0, 2, 3, 1))
            elif arr.shape[-1] != 1 and arr.shape[1] == 64:
                arr = arr[:, :, :, np.newaxis]

        return tf.convert_to_tensor(arr, dtype=tf.float32)

    def predict_proba(self, x: Union[np.ndarray, "tf.Tensor"]) -> np.ndarray:
        """Compute Softmax class probabilities array of shape (Batch, num_classes)."""
        tensor_in = self._prepare_input(x)
        probs, _, _ = self._call_graph(tensor_in)
        return probs.numpy()

    def forward(self, x: Union[np.ndarray, "tf.Tensor"]) -> np.ndarray:
        """Compute raw unnormalized logits of shape (Batch, num_classes)."""
        tensor_in = self._prepare_input(x)
        _, logits, _ = self._call_graph(tensor_in)
        return logits.numpy()

    def extract_features(self, x: Union[np.ndarray, "tf.Tensor"]) -> np.ndarray:
        """Extract L2-normalized 2048-dim feature embeddings."""
        tensor_in = self._prepare_input(x)
        _, _, features = self._call_graph(tensor_in)
        feat_np = features.numpy()
        norm = np.linalg.norm(feat_np, axis=1, keepdims=True)
        return feat_np / (norm + 1e-8)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        """Load trained weights from .keras or .weights.h5 file."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Model weights file not found: {p}")
        try:
            # Check if loading full model or weights
            if p.suffix in (".keras", ".h5"):
                self.model.load_weights(str(p))
            else:
                self.model.load_weights(str(p))
        except Exception as e:
            # Fallback to load full model
            try:
                loaded = keras.models.load_model(str(p))
                self.model = loaded
            except Exception as inner_err:
                raise RuntimeError(f"Could not load weights from {p}: {e} / {inner_err}")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save complete model to .keras file."""
        self.model.save(str(filepath))
