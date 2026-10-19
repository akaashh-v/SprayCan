"""Unit and Integration Tests for TensorFlow Stage 2 Migration.

Verifies:
1. TensorFlow model construction & parameter shapes
2. Converted weights loading and stability
3. Numerical equivalence with original PyTorch best_detector.pth (< 1e-4 divergence)
4. Class order and label preservation (6 classes)
5. Zero false alarms on hard negative audio
6. Detector engine dual-backend operation (pytorch <-> tensorflow)
"""

from pathlib import Path
import unittest
import numpy as np
import torch

from backend.audio_utils import compute_log_mel_spectrogram, load_audio, pad_or_trim
from backend.config import (
    AUDIO_SAMPLES,
    CLASS_NAMES,
    CONSERVATIVE_THRESHOLD,
    MODELS_DIR,
    NUM_CLASSES,
    SAMPLE_RATE,
)
from backend.detector_engine import SprayCanDetectorEngine
from backend.stage2_classifier import SprayCanShakingCNN, evaluate_candidate_prediction
from backend.stage2_classifier_tf import build_spray_can_cnn_tf, SprayCanShakingCNNTF


class TestTensorFlowMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pth_path = MODELS_DIR / "best_detector.pth"
        cls.keras_path = MODELS_DIR / "best_detector.keras"
        cls.demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo_samples"

        # Reference PyTorch Model
        cls.pt_model = SprayCanShakingCNN(num_classes=NUM_CLASSES)
        ckpt = torch.load(cls.pth_path, map_location="cpu", weights_only=False)
        cls.pt_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        cls.pt_model.eval()

        # Converted TensorFlow Model
        cls.tf_wrapper = SprayCanShakingCNNTF(num_classes=NUM_CLASSES, weights_path=cls.keras_path)

    def test_class_names_and_indices_preserved(self):
        """Class order must match exactly: ambient, impact, environmental, rattle, hiss, shaking."""
        expected_classes = [
            "ambient_and_speech",
            "loud_impact",
            "continuous_environmental",
            "hard_negative_rattle",
            "aerosol_spray_hiss",
            "spray_can_shaking",
        ]
        self.assertEqual(CLASS_NAMES, expected_classes)
        self.assertEqual(NUM_CLASSES, 6)

    def test_synthetic_input_parity(self):
        """Synthetic random noise spectrograms must yield < 1e-4 difference across all classes."""
        np.random.seed(123)
        specs = np.random.randn(5, 64, 44).astype(np.float32)

        with torch.no_grad():
            pt_in = torch.from_numpy(specs).unsqueeze(1)
            pt_probs = torch.softmax(cls_pt := self.pt_model(pt_in), dim=-1).numpy()

        tf_probs = self.tf_wrapper.predict_proba(specs)
        max_diff = float(np.max(np.abs(pt_probs - tf_probs)))
        self.assertLess(max_diff, 1e-4)

    def test_authentic_spray_can_audio_parity(self):
        """Authentic spray can sample must evaluate to the same probabilities in PT and TF."""
        wav_file = self.demo_dir / "authentic_spray_can.wav"
        if not wav_file.exists():
            self.skipTest("authentic_spray_can.wav not found")

        audio = load_audio(wav_file, sr=SAMPLE_RATE)
        spec = compute_log_mel_spectrogram(audio[:AUDIO_SAMPLES], sr=SAMPLE_RATE)

        with torch.no_grad():
            pt_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            pt_probs = self.pt_model.predict_proba(pt_in).numpy()[0]

        tf_probs = self.tf_wrapper.predict_proba(spec)[0]
        max_diff = float(np.max(np.abs(pt_probs - tf_probs)))
        self.assertLess(max_diff, 1e-5)

        # Both must identify spray_can_shaking
        pt_class = CLASS_NAMES[np.argmax(pt_probs)]
        tf_class = CLASS_NAMES[np.argmax(tf_probs)]
        self.assertEqual(pt_class, "spray_can_shaking")
        self.assertEqual(tf_class, "spray_can_shaking")

    def test_hard_negative_coins_rejection_parity(self):
        """Coins rattling hard negative must be classified as hard_negative_rattle in both."""
        wav_file = self.demo_dir / "coins_rattling.wav"
        if not wav_file.exists():
            self.skipTest("coins_rattling.wav not found")

        audio = load_audio(wav_file, sr=SAMPLE_RATE)
        spec = compute_log_mel_spectrogram(audio[:AUDIO_SAMPLES], sr=SAMPLE_RATE)

        with torch.no_grad():
            pt_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            pt_probs = self.pt_model.predict_proba(pt_in).numpy()[0]

        tf_probs = self.tf_wrapper.predict_proba(spec)[0]
        self.assertLess(float(np.max(np.abs(pt_probs - tf_probs))), 1e-5)

        pt_class = CLASS_NAMES[np.argmax(pt_probs)]
        tf_class = CLASS_NAMES[np.argmax(tf_probs)]
        self.assertEqual(pt_class, "hard_negative_rattle")
        self.assertEqual(tf_class, "hard_negative_rattle")

    def test_detector_engine_dual_backend_seamless_switching(self):
        """SprayCanDetectorEngine must support both backends and produce matching outputs."""
        wav_file = self.demo_dir / "authentic_spray_can.wav"
        if not wav_file.exists():
            self.skipTest("authentic_spray_can.wav not found")

        audio = load_audio(wav_file, sr=SAMPLE_RATE)

        engine_pt = SprayCanDetectorEngine(backend="pytorch")
        engine_tf = SprayCanDetectorEngine(backend="tensorflow")

        self.assertEqual(engine_pt.backend, "pytorch")
        self.assertEqual(engine_tf.backend, "tensorflow")

        res_pt = engine_pt.evaluate_window(audio[:AUDIO_SAMPLES])
        res_tf = engine_tf.evaluate_window(audio[:AUDIO_SAMPLES])

        self.assertEqual(res_pt["is_stage1_candidate"], res_tf["is_stage1_candidate"])
        self.assertEqual(res_pt["predicted_class"], res_tf["predicted_class"])
        self.assertAlmostEqual(res_pt["p_spray_can"], res_tf["p_spray_can"], places=3)
        self.assertEqual(res_pt["status"], res_tf["status"])

    def test_feature_embedding_dimension_and_parity(self):
        """Feature embeddings must have shape (B, 2048) with L2 norm = 1.0."""
        spec = np.random.randn(1, 64, 44).astype(np.float32)
        pt_feat = self.pt_model.extract_features(torch.from_numpy(spec).unsqueeze(1)).numpy()
        tf_feat = self.tf_wrapper.extract_features(spec)

        self.assertEqual(tf_feat.shape, (1, 2048))
        norm = float(np.linalg.norm(tf_feat))
        self.assertAlmostEqual(norm, 1.0, places=4)
        self.assertLess(float(np.max(np.abs(pt_feat - tf_feat))), 1e-4)


if __name__ == "__main__":
    unittest.main()
