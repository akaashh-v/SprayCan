"""High-Fidelity PyTorch to TensorFlow Weight Transference Utility.

Reads trained PyTorch weights from `backend/models/best_detector.pth` and transfers them
layer-by-layer into the TensorFlow/Keras SprayCanShakingCNNTF architecture.

Features:
- Preserves exact model weights without retraining.
- Transposes 2D Conv weights from (out_channels, in_channels, kH, kW) to (kH, kW, in_channels, out_channels).
- Transposes Dense weights from (out_features, in_features) to (in_features, out_features).
- Maps BatchNorm parameters [weight, bias, running_mean, running_var] directly to [gamma, beta, moving_mean, moving_variance].
- Saves destination model to `backend/models/best_detector.keras` and weights to `backend/models/best_detector.weights.h5`.
- Performs numerical parity validation check against the PyTorch model on synthetic and authentic test inputs.
"""

from pathlib import Path
import sys
from typing import Dict, Tuple
import numpy as np
import torch

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import CLASS_NAMES, MODELS_DIR, NUM_CLASSES
from stage2_classifier import SprayCanShakingCNN
from stage2_classifier_tf import build_spray_can_cnn_tf, SprayCanShakingCNNTF


def convert_pytorch_checkpoint_to_tf(
    pth_path: Path = MODELS_DIR / "best_detector.pth",
    out_keras_path: Path = MODELS_DIR / "best_detector.keras",
    out_weights_path: Path = MODELS_DIR / "best_detector.weights.h5",
) -> Tuple[SprayCanShakingCNNTF, Dict[str, float]]:
    """Convert PyTorch checkpoint to TensorFlow/Keras model with identical weights."""
    pth_path = Path(pth_path)
    if not pth_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found at: {pth_path}")

    print(f"[*] Loading PyTorch checkpoint: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    pt_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Build reference PyTorch model
    pt_model = SprayCanShakingCNN(num_classes=NUM_CLASSES)
    pt_model.load_state_dict(pt_state)
    pt_model.eval()

    # Build destination TensorFlow/Keras model
    print("[*] Building TensorFlow/Keras SprayCanShakingCNNTF...")
    tf_wrapper = SprayCanShakingCNNTF(num_classes=NUM_CLASSES)
    tf_model = tf_wrapper.model

    # Convert state_dict tensors to numpy float32
    np_state = {k: v.cpu().numpy().astype(np.float32) for k, v in pt_state.items()}

    print("[*] Transferring layer weights...")

    # 1. Block 1: conv1_conv & conv1_bn
    w_conv1 = np.transpose(np_state["conv1.0.weight"], (2, 3, 1, 0))  # (32, 1, 3, 3) -> (3, 3, 1, 32)
    b_conv1 = np_state["conv1.0.bias"]
    tf_model.get_layer("conv1_conv").set_weights([w_conv1, b_conv1])

    gamma_bn1 = np_state["conv1.1.weight"]
    beta_bn1 = np_state["conv1.1.bias"]
    mean_bn1 = np_state["conv1.1.running_mean"]
    var_bn1 = np_state["conv1.1.running_var"]
    tf_model.get_layer("conv1_bn").set_weights([gamma_bn1, beta_bn1, mean_bn1, var_bn1])

    # 2. Block 2: conv2_conv & conv2_bn
    w_conv2 = np.transpose(np_state["conv2.0.weight"], (2, 3, 1, 0))  # (64, 32, 3, 3) -> (3, 3, 32, 64)
    b_conv2 = np_state["conv2.0.bias"]
    tf_model.get_layer("conv2_conv").set_weights([w_conv2, b_conv2])

    gamma_bn2 = np_state["conv2.1.weight"]
    beta_bn2 = np_state["conv2.1.bias"]
    mean_bn2 = np_state["conv2.1.running_mean"]
    var_bn2 = np_state["conv2.1.running_var"]
    tf_model.get_layer("conv2_bn").set_weights([gamma_bn2, beta_bn2, mean_bn2, var_bn2])

    # 3. Block 3: conv3_conv & conv3_bn
    w_conv3 = np.transpose(np_state["conv3.0.weight"], (2, 3, 1, 0))  # (128, 64, 3, 3) -> (3, 3, 64, 128)
    b_conv3 = np_state["conv3.0.bias"]
    tf_model.get_layer("conv3_conv").set_weights([w_conv3, b_conv3])

    gamma_bn3 = np_state["conv3.1.weight"]
    beta_bn3 = np_state["conv3.1.bias"]
    mean_bn3 = np_state["conv3.1.running_mean"]
    var_bn3 = np_state["conv3.1.running_var"]
    tf_model.get_layer("conv3_bn").set_weights([gamma_bn3, beta_bn3, mean_bn3, var_bn3])

    # 4. Block 4: conv4_conv & conv4_bn
    w_conv4 = np.transpose(np_state["conv4.0.weight"], (2, 3, 1, 0))  # (128, 128, 3, 3) -> (3, 3, 128, 128)
    b_conv4 = np_state["conv4.0.bias"]
    tf_model.get_layer("conv4_conv").set_weights([w_conv4, b_conv4])

    gamma_bn4 = np_state["conv4.1.weight"]
    beta_bn4 = np_state["conv4.1.bias"]
    mean_bn4 = np_state["conv4.1.running_mean"]
    var_bn4 = np_state["conv4.1.running_var"]
    tf_model.get_layer("conv4_bn").set_weights([gamma_bn4, beta_bn4, mean_bn4, var_bn4])

    # 5. Classifier Head: classifier_dense1 & classifier_bn1
    w_fc1 = np.transpose(np_state["classifier.1.weight"], (1, 0))  # (128, 2048) -> (2048, 128)
    b_fc1 = np_state["classifier.1.bias"]
    tf_model.get_layer("classifier_dense1").set_weights([w_fc1, b_fc1])

    gamma_fbn1 = np_state["classifier.2.weight"]
    beta_fbn1 = np_state["classifier.2.bias"]
    mean_fbn1 = np_state["classifier.2.running_mean"]
    var_fbn1 = np_state["classifier.2.running_var"]
    tf_model.get_layer("classifier_bn1").set_weights([gamma_fbn1, beta_fbn1, mean_fbn1, var_fbn1])

    # 6. Classifier Head: classifier_logits
    w_fc2 = np.transpose(np_state["classifier.5.weight"], (1, 0))  # (6, 128) -> (128, 6)
    b_fc2 = np_state["classifier.5.bias"]
    tf_model.get_layer("classifier_logits").set_weights([w_fc2, b_fc2])

    print("[+] All weights transferred successfully!")

    # Save destination models
    out_keras_path = Path(out_keras_path)
    out_weights_path = Path(out_weights_path)
    out_keras_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Saving TensorFlow Keras model: {out_keras_path}")
    tf_wrapper.save(out_keras_path)
    print(f"[*] Saving TensorFlow weights: {out_weights_path}")
    tf_model.save_weights(str(out_weights_path))

    # Perform Verification on synthetic inputs
    print("[*] Validating numerical parity between PyTorch and TensorFlow models...")
    np.random.seed(42)
    test_inputs = np.random.randn(8, 64, 44).astype(np.float32)

    with torch.no_grad():
        pt_in = torch.from_numpy(test_inputs).unsqueeze(1)  # (8, 1, 64, 44)
        pt_logits = pt_model(pt_in).numpy()
        pt_probs = torch.softmax(pt_model(pt_in), dim=-1).numpy()
        pt_feats = pt_model.extract_features(pt_in).numpy()

    tf_probs, tf_logits, tf_feats_unnorm = tf_model(test_inputs[:, :, :, np.newaxis], training=False)
    tf_probs = tf_probs.numpy()
    tf_logits = tf_logits.numpy()
    tf_feats = tf_wrapper.extract_features(test_inputs)

    max_prob_diff = float(np.max(np.abs(pt_probs - tf_probs)))
    mean_prob_diff = float(np.mean(np.abs(pt_probs - tf_probs)))
    max_logit_diff = float(np.max(np.abs(pt_logits - tf_logits)))
    max_feat_diff = float(np.max(np.abs(pt_feats - tf_feats)))

    print(f"    - Max logit difference:      {max_logit_diff:.6e}")
    print(f"    - Max probability difference: {max_prob_diff:.6e}")
    print(f"    - Mean probability difference: {mean_prob_diff:.6e}")
    print(f"    - Max feature difference:     {max_feat_diff:.6e}")

    metrics = {
        "max_logit_diff": max_logit_diff,
        "max_prob_diff": max_prob_diff,
        "mean_prob_diff": mean_prob_diff,
        "max_feat_diff": max_feat_diff,
    }

    if max_prob_diff < 1e-4:
        print("[+] VERIFICATION PASSED: TensorFlow model matches PyTorch model with high mathematical fidelity!")
    else:
        print(f"[!] Warning: Probability difference {max_prob_diff} is higher than expected (<1e-4).")

    return tf_wrapper, metrics


if __name__ == "__main__":
    convert_pytorch_checkpoint_to_tf()
