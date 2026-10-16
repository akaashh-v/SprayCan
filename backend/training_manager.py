"""One-Click Model Training, Evaluation, Versioning, and Safe Activation Engine.

Provides:
- Thread-safe background training worker without blocking the Flask server.
- Concurrency locking preventing accidental simultaneous training jobs.
- Clean cancellation mechanism.
- Real-time training progress, loss tracking, and log streaming.
- Candidate checkpoint isolation (never overwrites models/best_detector.pth during training).
- Side-by-side evaluation of candidate model vs. active model on held-out test splits.
- Atomic activation and rollback with audit logging and instant detector engine reloading.
"""

import copy
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from admin_auth import log_audit_action
from config import (
    BASE_DIR,
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_IDX,
    CONSERVATIVE_THRESHOLD,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MAX_COMPETING_CONFIDENCE,
    MIN_CONFIDENCE_MARGIN,
    MODEL_SAVE_PATH,
    MODELS_DIR,
    NUM_CLASSES,
    PROCESSED_DIR,
    RANDOM_SEED,
    UNCERTAIN_ENTROPY_THRESHOLD,
    WEIGHT_DECAY,
)
from dataset_manager import get_db, validate_dataset_for_training
from stage2_classifier import SprayCanShakingCNN, evaluate_candidate_prediction

logger = logging.getLogger(__name__)

CANDIDATES_DIR = MODELS_DIR / "candidate_checkpoints"
CHECKPOINTS_BACKUP_DIR = MODELS_DIR / "checkpoints"
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def init_training_db() -> None:
    """Initialize training jobs and model versions tables in dataset_metadata.db."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Training Jobs Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                status TEXT NOT NULL DEFAULT 'queued', -- queued, validating, training, evaluating, completed, failed, cancelled
                admin_user TEXT NOT NULL,
                total_epochs INTEGER NOT NULL,
                current_epoch INTEGER DEFAULT 0,
                train_loss REAL,
                val_loss REAL,
                val_spray_f1 REAL,
                progress_pct INTEGER DEFAULT 0,
                candidate_checkpoint_path TEXT,
                metrics_json TEXT,
                error_message TEXT,
                logs_json TEXT
            )
            """
        )

        # Model Versions Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id TEXT PRIMARY KEY,
                job_id TEXT,
                checkpoint_path TEXT NOT NULL,
                created_at REAL NOT NULL,
                created_at_iso TEXT NOT NULL,
                created_by TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0,
                activated_at REAL,
                activated_by TEXT,
                evaluation_metrics_json TEXT,
                acceptance_status TEXT, -- 'passed', 'failed', 'review_required'
                FOREIGN KEY (job_id) REFERENCES training_jobs (id)
            )
            """
        )
        conn.commit()

        # Ensure existing production model is registered as active version if not present
        cursor.execute("SELECT COUNT(*) as count FROM model_versions WHERE is_active = 1")
        active_count = cursor.fetchone()["count"]
        if active_count == 0 and MODEL_SAVE_PATH.exists():
            now = time.time()
            iso_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
            cursor.execute(
                """
                INSERT INTO model_versions (
                    id, job_id, checkpoint_path, created_at, created_at_iso, created_by,
                    is_active, activated_at, activated_by, evaluation_metrics_json, acceptance_status
                )
                VALUES ('v1_baseline_production', NULL, ?, ?, ?, 'system', 1, ?, 'system', '{}', 'passed')
                """,
                (str(MODEL_SAVE_PATH.relative_to(BASE_DIR)).replace("\\", "/"), now, iso_time, now),
            )
            conn.commit()


init_training_db()


class TrainingManager:
    """Manages thread-safe asynchronous training execution, job tracking, and model activation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active_job_id: Optional[str] = None
        self._cancellation_requested = False
        self._live_job_state: Dict = {}

    def get_active_job_id(self) -> Optional[str]:
        with self._lock:
            return self._active_job_id

    def is_training_running(self) -> bool:
        with self._lock:
            return self._active_job_id is not None

    def request_cancellation(self, job_id: str, admin_user: str) -> Tuple[bool, str]:
        with self._lock:
            if self._active_job_id != job_id:
                return False, f"Job '{job_id}' is not currently running."
            self._cancellation_requested = True
            self._append_log(job_id, f"Cancellation requested by administrator '{admin_user}'. Terminating cleanly...")
            log_audit_action(admin_user, "cancel_training_job", {"job_id": job_id})
            return True, "Cancellation signal sent."

    def _append_log(self, job_id: str, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        logger.info("[Training %s] %s", job_id, message)
        if self._live_job_state.get("id") == job_id:
            self._live_job_state.setdefault("logs", []).append(entry)

    def get_job_status(self, job_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch status and recent logs for a job (either active or historical)."""
        with self._lock:
            if job_id is None or (self._active_job_id and job_id == self._active_job_id):
                if self._live_job_state:
                    return copy.deepcopy(self._live_job_state)

        # Fallback to database
        with get_db() as conn:
            cursor = conn.cursor()
            if job_id:
                cursor.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
            else:
                cursor.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            try:
                data["logs"] = json.loads(data["logs_json"]) if data.get("logs_json") else []
            except Exception:
                data["logs"] = []
            try:
                data["metrics"] = json.loads(data["metrics_json"]) if data.get("metrics_json") else {}
            except Exception:
                data["metrics"] = {}
            return data

    def start_training_job(
        self,
        admin_user: str,
        epochs: int = 15,
        fine_tune: bool = True,
        rebuild_dataset: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Launch a controlled background training job."""
        with self._lock:
            if self._active_job_id is not None:
                return None, f"Another training job ({self._active_job_id}) is currently in progress."

            job_id = f"train_job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            self._active_job_id = job_id
            self._cancellation_requested = False

            now = time.time()
            self._live_job_state = {
                "id": job_id,
                "status": "queued",
                "admin_user": admin_user,
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "total_epochs": epochs,
                "current_epoch": 0,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "val_spray_f1": 0.0,
                "progress_pct": 0,
                "logs": [f"[{time.strftime('%H:%M:%S')}] Training job {job_id} queued by {admin_user}"],
                "candidate_checkpoint_path": None,
                "metrics": {},
                "error_message": None,
            }

            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO training_jobs (id, created_at, status, admin_user, total_epochs, progress_pct, logs_json)
                    VALUES (?, ?, 'queued', ?, ?, 0, ?)
                    """,
                    (job_id, now, admin_user, epochs, json.dumps(self._live_job_state["logs"])),
                )
                conn.commit()

        log_audit_action(
            admin_user,
            "launch_training_job",
            {"job_id": job_id, "epochs": epochs, "fine_tune": fine_tune, "rebuild_dataset": rebuild_dataset},
        )

        thread = threading.Thread(
            target=self._run_training_worker,
            args=(job_id, admin_user, epochs, fine_tune, rebuild_dataset),
            daemon=True,
        )
        thread.start()
        return job_id, None

    def _run_training_worker(
        self,
        job_id: str,
        admin_user: str,
        epochs: int,
        fine_tune: bool,
        rebuild_dataset: bool,
    ) -> None:
        start_time = time.time()
        self._live_job_state["started_at"] = start_time
        self._live_job_state["status"] = "validating"
        self._append_log(job_id, "Starting pre-training dataset validation...")

        candidate_rel_path = f"models/candidate_checkpoints/{job_id}.pth"
        candidate_abs_path = BASE_DIR / candidate_rel_path

        try:
            # 1. Dataset Pre-Flight Validation
            val_report = validate_dataset_for_training()
            if not val_report["is_eligible_for_training"]:
                err_msg = "; ".join(val_report["validation_errors"])
                raise ValueError(f"Dataset validation failed: {err_msg}")

            self._append_log(
                job_id,
                f"Dataset validated successfully: {val_report['total_accepted_recordings']} accepted recordings "
                f"({val_report['positives_count']} positive, {val_report['negatives_count']} negative).",
            )

            # 2. Rebuild Dataset Partitions if requested
            if rebuild_dataset:
                self._live_job_state["status"] = "preparing_dataset"
                self._append_log(job_id, "Building feature partitions (train.pt, val.pt, test.pt) with group-based session isolation...")
                from dataset_builder import build_full_dataset
                build_full_dataset()
                self._append_log(job_id, "Feature extraction complete. Partitions saved to data/processed/")

            if self._cancellation_requested:
                raise InterruptedError("Training cancelled by administrator.")

            # 3. Load Data Tensors
            self._live_job_state["status"] = "training"
            self._append_log(job_id, f"Initializing neural network optimization on compute device: {DEVICE}")

            train_data = torch.load(PROCESSED_DIR / "train.pt", weights_only=True)
            val_data = torch.load(PROCESSED_DIR / "val.pt", weights_only=True)
            test_data = torch.load(PROCESSED_DIR / "test.pt", weights_only=True)

            train_loader = DataLoader(
                TensorDataset(train_data["spectrograms"], train_data["labels"]),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(val_data["spectrograms"], val_data["labels"]),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            test_loader = DataLoader(
                TensorDataset(test_data["spectrograms"], test_data["labels"]),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            # Class weights to handle class imbalance
            class_counts = np.bincount(train_data["labels"].numpy(), minlength=NUM_CLASSES)
            total_samples = len(train_data["labels"])
            weights = total_samples / (NUM_CLASSES * np.maximum(1, class_counts).astype(np.float32))
            class_weights_t = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

            model = SprayCanShakingCNN(num_classes=NUM_CLASSES).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights_t)

            best_val_f1 = -1.0
            best_epoch = 0

            # Learning rate
            effective_lr = LEARNING_RATE * 0.3 if fine_tune and MODEL_SAVE_PATH.exists() else LEARNING_RATE

            if fine_tune and MODEL_SAVE_PATH.exists():
                self._append_log(job_id, f"Fine-tuning from active model checkpoint with LR={effective_lr:.1e}...")
                ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state)
            else:
                self._append_log(job_id, f"Training model from scratch for {epochs} epochs with LR={effective_lr:.1e}...")

            optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

            # 4. Training Loop
            for epoch in range(1, epochs + 1):
                if self._cancellation_requested:
                    raise InterruptedError("Training cancelled by administrator.")

                model.train()
                epoch_loss = 0.0

                for x_b, y_b in train_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    optimizer.zero_grad()
                    logits = model(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(y_b)

                scheduler.step()
                epoch_loss /= len(train_loader.dataset)

                # Validation Evaluation
                model.eval()
                val_loss = 0.0
                val_preds, val_targets = [], []

                with torch.no_grad():
                    for x_b, y_b in val_loader:
                        x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                        logits = model(x_b)
                        loss = criterion(logits, y_b)
                        val_loss += loss.item() * len(y_b)
                        preds = torch.argmax(logits, dim=-1)
                        val_preds.extend(preds.cpu().numpy().tolist())
                        val_targets.extend(y_b.cpu().numpy().tolist())

                val_loss /= len(val_loader.dataset)
                val_preds_arr = np.array(val_preds)
                val_targets_arr = np.array(val_targets)

                # Compute Target Shaking Class F1
                spray_idx = CLASS_TO_IDX["spray_can_shaking"]
                tp = np.sum((val_preds_arr == spray_idx) & (val_targets_arr == spray_idx))
                fp = np.sum((val_preds_arr == spray_idx) & (val_targets_arr != spray_idx))
                fn = np.sum((val_preds_arr != spray_idx) & (val_targets_arr == spray_idx))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                val_spray_f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                pct = int((epoch / epochs) * 80)  # 0 to 80% for training phase
                self._live_job_state["current_epoch"] = epoch
                self._live_job_state["train_loss"] = round(epoch_loss, 4)
                self._live_job_state["val_loss"] = round(val_loss, 4)
                self._live_job_state["val_spray_f1"] = round(val_spray_f1, 4)
                self._live_job_state["progress_pct"] = pct

                self._append_log(
                    job_id,
                    f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Spray F1: {val_spray_f1:.4f}",
                )

                # Track best candidate model
                if val_spray_f1 > best_val_f1 or epoch == 1:
                    best_val_f1 = val_spray_f1
                    best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "val_f1": best_val_f1,
                            "num_classes": NUM_CLASSES,
                            "class_names": CLASS_NAMES,
                            "job_id": job_id,
                            "created_at": time.time(),
                        },
                        candidate_abs_path,
                    )

            self._append_log(job_id, f"Training complete! Best candidate model checkpoint saved at Epoch {best_epoch} with Val F1: {best_val_f1:.4f}")

            # 5. Candidate Evaluation on Held-Out Test Set
            self._live_job_state["status"] = "evaluating"
            self._live_job_state["progress_pct"] = 85
            self._append_log(job_id, "Evaluating candidate model against unseen held-out test partition...")

            # Load best candidate checkpoint for evaluation
            candidate_ckpt = torch.load(candidate_abs_path, map_location=DEVICE, weights_only=False)
            candidate_model = SprayCanShakingCNN(num_classes=NUM_CLASSES).to(DEVICE)
            candidate_model.load_state_dict(candidate_ckpt["model_state_dict"])
            candidate_model.eval()

            candidate_metrics = self._evaluate_model_split(candidate_model, test_loader)
            self._live_job_state["progress_pct"] = 95

            # Also evaluate current active model on same test split for direct comparison
            active_metrics = {}
            if MODEL_SAVE_PATH.exists():
                try:
                    active_ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
                    active_model = SprayCanShakingCNN(num_classes=NUM_CLASSES).to(DEVICE)
                    active_model.load_state_dict(active_ckpt.get("model_state_dict", active_ckpt))
                    active_model.eval()
                    active_metrics = self._evaluate_model_split(active_model, test_loader)
                except Exception as e:
                    logger.warning("Could not evaluate active baseline model: %s", e)

            # 6. Safety & Acceptance Checks
            shaking_recall = candidate_metrics["shaking_recall"]
            hard_neg_fp = candidate_metrics["false_positives_by_class"].get("hard_negative_rattle", 0)
            hiss_fp = candidate_metrics["false_positives_by_class"].get("aerosol_spray_hiss", 0)

            passed_checks = True
            reasons = []

            if shaking_recall < 0.70:
                passed_checks = False
                reasons.append(f"Target shaking recall ({shaking_recall*100:.1f}%) is below minimum acceptance threshold of 70%.")

            if hard_neg_fp > 5:
                passed_checks = False
                reasons.append(f"Excessive false positives on hard-negative rattles ({hard_neg_fp} false alarms).")

            if hiss_fp > 5:
                passed_checks = False
                reasons.append(f"Excessive false positives on spray hiss ({hiss_fp} false alarms).")

            acceptance_status = "passed" if passed_checks else "review_required"

            full_eval_report = {
                "candidate": candidate_metrics,
                "active_baseline": active_metrics,
                "acceptance_status": acceptance_status,
                "acceptance_reasons": reasons,
                "best_epoch": best_epoch,
                "best_val_f1": round(best_val_f1, 4),
            }

            completed_time = time.time()
            self._live_job_state["status"] = "completed"
            self._live_job_state["completed_at"] = completed_time
            self._live_job_state["progress_pct"] = 100
            self._live_job_state["candidate_checkpoint_path"] = candidate_rel_path
            self._live_job_state["metrics"] = full_eval_report

            self._append_log(
                job_id,
                f"Evaluation finished! Status: {acceptance_status.upper()}. "
                f"Target Shaking Recall: {shaking_recall*100:.1f}%, Precision: {candidate_metrics['shaking_precision']*100:.1f}%. "
                f"Candidate checkpoint is ready for review.",
            )

            # Register candidate in model_versions table
            version_id = f"v_{int(completed_time)}_{uuid.uuid4().hex[:6]}"
            iso_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_time))

            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO model_versions (
                        id, job_id, checkpoint_path, created_at, created_at_iso, created_by,
                        is_active, evaluation_metrics_json, acceptance_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        version_id,
                        job_id,
                        candidate_rel_path,
                        completed_time,
                        iso_time,
                        admin_user,
                        json.dumps(full_eval_report),
                        acceptance_status,
                    ),
                )
                conn.execute(
                    """
                    UPDATE training_jobs
                    SET status = 'completed', completed_at = ?, progress_pct = 100,
                        candidate_checkpoint_path = ?, metrics_json = ?, logs_json = ?
                    WHERE id = ?
                    """,
                    (
                        completed_time,
                        candidate_rel_path,
                        json.dumps(full_eval_report),
                        json.dumps(self._live_job_state["logs"]),
                        job_id,
                    ),
                )
                conn.commit()

            log_audit_action(
                admin_user,
                "training_job_completed",
                {
                    "job_id": job_id,
                    "version_id": version_id,
                    "acceptance_status": acceptance_status,
                    "shaking_recall": shaking_recall,
                },
            )

        except InterruptedError as e:
            self._live_job_state["status"] = "cancelled"
            self._live_job_state["completed_at"] = time.time()
            self._live_job_state["error_message"] = str(e)
            self._append_log(job_id, f"Job cancelled: {str(e)}")
            with get_db() as conn:
                conn.execute(
                    "UPDATE training_jobs SET status = 'cancelled', completed_at = ?, error_message = ?, logs_json = ? WHERE id = ?",
                    (time.time(), str(e), json.dumps(self._live_job_state.get("logs", [])), job_id),
                )
                conn.commit()

        except Exception as e:
            logger.error("Training worker failed: %s", e, exc_info=True)
            self._live_job_state["status"] = "failed"
            self._live_job_state["completed_at"] = time.time()
            self._live_job_state["error_message"] = str(e)
            self._append_log(job_id, f"ERROR: Training job failed - {str(e)}")
            with get_db() as conn:
                conn.execute(
                    "UPDATE training_jobs SET status = 'failed', completed_at = ?, error_message = ?, logs_json = ? WHERE id = ?",
                    (time.time(), str(e), json.dumps(self._live_job_state.get("logs", [])), job_id),
                )
                conn.commit()

        finally:
            with self._lock:
                self._active_job_id = None
                self._cancellation_requested = False

    def _evaluate_model_split(self, model: nn.Module, loader: DataLoader) -> Dict:
        """Run standard evaluation and compute 6x6 confusion matrix and per-class metrics."""
        all_probs: List[np.ndarray] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(DEVICE)
                logits = model(x_b)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(y_b.numpy().tolist())

        probs_arr = np.concatenate(all_probs, axis=0)
        labels_arr = np.array(all_labels)

        # Gated alert evaluation
        spray_idx = CLASS_TO_IDX["spray_can_shaking"]
        total_spray = int(np.sum(labels_arr == spray_idx))
        total_negatives = len(labels_arr) - total_spray

        tp = 0
        fn = 0
        fp_by_class = {c: 0 for c in CLASS_NAMES if c != "spray_can_shaking"}
        conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

        preds = np.argmax(probs_arr, axis=-1)
        for true_l, pred_l in zip(labels_arr, preds):
            conf_matrix[true_l, pred_l] += 1

        for i in range(len(labels_arr)):
            p_vec = probs_arr[i]
            true_label = labels_arr[i]
            true_name = CLASS_NAMES[true_label]

            eval_res = evaluate_candidate_prediction(
                p_vec,
                conservative_threshold=CONSERVATIVE_THRESHOLD,
                max_competing_confidence=MAX_COMPETING_CONFIDENCE,
                min_confidence_margin=MIN_CONFIDENCE_MARGIN,
                entropy_threshold=UNCERTAIN_ENTROPY_THRESHOLD,
            )

            is_alert = eval_res["is_confirmed_spray"]
            if true_label == spray_idx:
                if is_alert:
                    tp += 1
                else:
                    fn += 1
            else:
                if is_alert:
                    fp_by_class[true_name] += 1

        total_fp = sum(fp_by_class.values())
        prec = tp / (tp + total_fp) if (tp + total_fp) > 0 else 1.0
        rec = tp / total_spray if total_spray > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "total_test_samples": len(labels_arr),
            "total_spray_samples": total_spray,
            "total_negative_samples": total_negatives,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives_total": total_fp,
            "shaking_precision": round(float(prec), 4),
            "shaking_recall": round(float(rec), 4),
            "shaking_f1": round(float(f1), 4),
            "false_positives_by_class": fp_by_class,
            "confusion_matrix": conf_matrix.tolist(),
        }

    def list_model_versions(self) -> List[Dict]:
        """Fetch all model versions with their evaluation summaries."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM model_versions
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
            versions = []
            for r in rows:
                v = dict(r)
                v["version_id"] = v["id"]
                try:
                    v["metrics"] = json.loads(v["evaluation_metrics_json"]) if v.get("evaluation_metrics_json") else {}
                except Exception:
                    v["metrics"] = {}
                cand_metrics = v["metrics"].get("candidate", {})
                v["val_recall_spray"] = cand_metrics.get("shaking_recall")
                v["val_f1_spray"] = cand_metrics.get("shaking_f1")
                versions.append(v)
            return versions

    def activate_model_version(self, version_id: str, admin_user: str, engine_instance=None) -> Tuple[bool, Optional[str]]:
        """Atomically activate a candidate model version and reload detector engine."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_versions WHERE id = ?", (version_id,))
            target = cursor.fetchone()
            if not target:
                return False, f"Model version '{version_id}' not found."

            target_ckpt_path = BASE_DIR / target["checkpoint_path"]
            if not target_ckpt_path.exists():
                return False, f"Checkpoint file {target_ckpt_path} does not exist on disk."

            # Verify checkpoint decodability before making any changes
            try:
                test_ckpt = torch.load(target_ckpt_path, map_location="cpu", weights_only=False)
                state = test_ckpt.get("model_state_dict", test_ckpt)
                test_model = SprayCanShakingCNN(num_classes=NUM_CLASSES)
                test_model.load_state_dict(state)
            except Exception as e:
                return False, f"Failed verification of candidate checkpoint: {str(e)}"

            # 1. Backup current active model
            if MODEL_SAVE_PATH.exists():
                now_ts = int(time.time())
                backup_path = CHECKPOINTS_BACKUP_DIR / f"active_backup_{now_ts}.pth"
                shutil.copyfile(MODEL_SAVE_PATH, backup_path)
                logger.info("Backed up active model to %s", backup_path)

            # 2. Atomically copy candidate to active model path
            target_path_resolved = Path(target_ckpt_path).resolve()
            save_path_resolved = Path(MODEL_SAVE_PATH).resolve()
            if target_path_resolved != save_path_resolved:
                shutil.copyfile(target_ckpt_path, MODEL_SAVE_PATH)


            # 3. Update database records
            now = time.time()
            cursor.execute("UPDATE model_versions SET is_active = 0 WHERE is_active = 1")
            cursor.execute(
                """
                UPDATE model_versions 
                SET is_active = 1, activated_at = ?, activated_by = ?
                WHERE id = ?
                """,
                (now, admin_user, version_id),
            )
            conn.commit()

        # 4. Hot reload live engine
        if engine_instance is not None:
            try:
                engine_instance.reload_model(MODEL_SAVE_PATH)
                logger.info("Live engine successfully hot-reloaded with activated model version %s", version_id)
            except Exception as e:
                logger.error("Failed to hot-reload live engine: %s", e)

        log_audit_action(
            admin_user,
            "activate_model_version",
            {"version_id": version_id, "checkpoint_path": target["checkpoint_path"]},
        )
        return True, None

    def rollback_to_version(self, target_version_id: str, admin_user: str, engine_instance=None) -> Tuple[bool, Optional[str]]:
        """Rollback active model to a previous version."""
        return self.activate_model_version(target_version_id, admin_user, engine_instance)


# Global singleton instance
training_manager = TrainingManager()

# Module-level convenience delegators
start_training_job = training_manager.start_training_job
get_job_status = training_manager.get_job_status
request_cancellation = training_manager.request_cancellation
list_model_versions = training_manager.list_model_versions
activate_model_version = training_manager.activate_model_version
rollback_to_version = training_manager.rollback_to_version
get_active_job_id = training_manager.get_active_job_id
is_training_running = training_manager.is_training_running

