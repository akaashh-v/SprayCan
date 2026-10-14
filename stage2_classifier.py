"""Stage 2: Full Spectro-Temporal Deep CNN Classifier for Spray Can Confirmation.

Evaluates the complete 2D time-frequency pattern (Log-Mel Spectrogram 64x44)
to confirm authentic aerosol spray can shaking vs. diverse hard negatives:
- Coins in jar / pocket
- Pill bottles & Tic-Tacs
- Keys jangling
- Loose metal parts / screws
- Glassware clinking
- Liquid bottles sloshing
- Ambient room tone, speech, fan hums

Never relies on a single frequency or loudness threshold.
Outputs 3-class probabilities and uncertainty entropy metric.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    CONSERVATIVE_THRESHOLD,
    DEVICE,
    MAX_COMPETING_CONFIDENCE,
    MAX_ENV_CONFIDENCE,
    MIN_CONFIDENCE_MARGIN,
    NUM_CLASSES,
    UNCERTAIN_ENTROPY_THRESHOLD,
)


class SprayCanShakingCNN(nn.Module):
    """Deep 4-Block 2D Convolutional Network for Spectro-Temporal Shaking Classification."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Block 1: Micro-transient edge & onset detection
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (32, 32, 22)
            nn.Dropout2d(0.15),
        )

        # Block 2: Frequency band interaction & cavity resonance
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (64, 16, 11)
            nn.Dropout2d(0.20),
        )

        # Block 3: Rhythmic temporal pattern & cadence continuity
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (128, 8, 5)
            nn.Dropout2d(0.25),
        )

        # Block 4: High-level semantic acoustic shape
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (128, 4, 4)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input tensor shape: (B, 1, 64, 44)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Softmax class probabilities."""
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized 2048-dim feature embeddings for acoustic comparison."""
        self.eval()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        flat = torch.flatten(x, 1)
        norm = torch.norm(flat, p=2, dim=1, keepdim=True)
        return flat / (norm + 1e-8)


def evaluate_candidate_prediction(
    probs: np.ndarray,
    conservative_threshold: float = CONSERVATIVE_THRESHOLD,
    max_competing_confidence: float = MAX_COMPETING_CONFIDENCE,
    max_env_confidence: float = MAX_ENV_CONFIDENCE,
    min_confidence_margin: float = MIN_CONFIDENCE_MARGIN,
    entropy_threshold: float = UNCERTAIN_ENTROPY_THRESHOLD,
) -> Dict:
    """Evaluate 6-class prediction probabilities with conservative gating and uncertainty estimation.

    Classes:
      0: ambient_and_speech
      1: loud_impact
      2: continuous_environmental (rain, water drops, train rumbles)
      3: hard_negative_rattle
      4: aerosol_spray_hiss
      5: spray_can_shaking
    """
    idx_spray = CLASS_TO_IDX["spray_can_shaking"]
    p_spray = float(probs[idx_spray])

    # Competing class probabilities
    competing_probs = {}
    for name in CLASS_NAMES:
        if name != "spray_can_shaking":
            competing_probs[name] = float(probs[CLASS_TO_IDX[name]])

    max_competing_name = max(competing_probs, key=competing_probs.get)
    max_competing_val = float(competing_probs[max_competing_name])
    margin = float(p_spray - max_competing_val)

    # Compute Normalized Shannon Entropy: H = -sum(p * log(p)) / log(K)
    p_safe = np.clip(probs, 1e-8, 1.0)
    entropy = float(-np.sum(p_safe * np.log(p_safe)) / np.log(len(probs)))

    is_uncertain = entropy >= entropy_threshold
    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    p_spray_hiss = float(competing_probs.get("aerosol_spray_hiss", 0.0))
    p_impact = float(competing_probs.get("loud_impact", 0.0))
    p_env = float(competing_probs.get("continuous_environmental", 0.0))
    p_rattle = float(competing_probs.get("hard_negative_rattle", 0.0))
    p_ambient = float(competing_probs.get("ambient_and_speech", 0.0))

    # Separate physical mimics from additive environmental background:
    # Physical mimics (spray hiss, hard rattles, impacts) must never trigger an alert.
    physical_mimics = {
        "aerosol_spray_hiss": p_spray_hiss,
        "hard_negative_rattle": p_rattle,
        "loud_impact": p_impact,
    }
    max_mimic_name = max(physical_mimics, key=physical_mimics.get)
    max_mimic_val = float(physical_mimics[max_mimic_name])
    mimic_margin = float(p_spray - max_mimic_val)

    # Conservative Confirmation Rules for Railway Monitoring:
    # 1. Spray can probability must meet conservative threshold (tau).
    # 2. Confusable physical mimics (spray hiss, keys/coins, impact bangs) must remain strictly low (<= max_competing_confidence).
    # 3. Target class must hold a solid margin over physical mimics (>= min_confidence_margin).
    # 4. Target must dominate over station environmental noise (p_spray > p_env).
    # 5. Shannon entropy must not indicate high acoustic ambiguity (< entropy_threshold).
    is_confirmed_spray = (
        (p_spray >= conservative_threshold) and
        (max_mimic_val <= max_competing_confidence) and
        (mimic_margin >= min_confidence_margin) and
        (p_spray > p_env) and
        (p_spray > p_ambient) and
        not is_uncertain
    )

    if is_confirmed_spray:
        verdict = "CONFIRMED_SPRAY_CAN"
    elif is_uncertain:
        verdict = "UNCERTAIN_AMBIGUOUS_SOUND"
    elif p_spray_hiss >= max(0.18, max_competing_confidence) or predicted_class == "aerosol_spray_hiss":
        verdict = "AEROSOL_SPRAY_HISS_REJECTED"
    elif p_rattle >= max(0.18, max_competing_confidence) or predicted_class == "hard_negative_rattle":
        verdict = "HARD_NEGATIVE_RATTLE_REJECTED"
    elif p_impact >= max(0.20, max_competing_confidence) or predicted_class == "loud_impact":
        verdict = "LOUD_IMPACT_REJECTED"
    elif predicted_class == "continuous_environmental" and p_spray < 0.40:
        verdict = "CONTINUOUS_ENVIRONMENTAL_REJECTED"
    elif predicted_class == "ambient_and_speech" and p_spray < 0.40:
        verdict = "AMBIENT_AND_SPEECH"
    else:
        # Ambiguous candidate: target presence is detectable but does not meet conservative margin
        verdict = "UNCERTAIN_AMBIGUOUS_SOUND"

    return {
        "is_confirmed_spray": is_confirmed_spray,
        "verdict": verdict,
        "p_spray_can": round(p_spray, 4),
        "p_aerosol_spray_hiss": round(p_spray_hiss, 4),
        "p_loud_impact": round(p_impact, 4),
        "p_continuous_environmental": round(p_env, 4),
        "p_hard_negative": round(p_rattle, 4),
        "p_ambient": round(p_ambient, 4),
        "max_competing_confidence": round(max_competing_val, 4),
        "max_competing_class": max_competing_name,
        "confidence_margin": round(margin, 4),
        "entropy": round(entropy, 3),
        "is_uncertain": is_uncertain,
        "predicted_class": predicted_class,
        "class_probabilities": {name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)},
    }
