# Precision Aerosol Spray Can Sound Detector

A robust, production-grade **Two-Stage Acoustic Event Detection (AED)** system designed to raise an alert **only when strong, specific evidence of an aerosol spray can being shaken is present**, while strictly eliminating false alerts on similar rattling sounds (keys, coins, pill bottles, loose screws, glassware, liquid bottles) and ambient environmental noises.

---

## 1. System Architecture: Two-Stage Decision Pipeline

```
Raw Audio Stream (22,050 Hz)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Kinematic Shaking Burst Candidate Pre-Filter      │
│ • Energy Gate: RMS >= 0.006 (Rejects quiet ambient/silence) │
│ • Cadence: 2.0 to 8.5 impacts/sec (Physical hand shaking)   │
│ • Broadband Rattle Ratio: >= 25% energy in 2.0 - 9.0 kHz    │
│                                                             │
│ TREAT STRICTLY AS UNCONFIRMED CANDIDATE — NEVER ALERTS ALONE│
└─────────────────────────────────────────────────────────────┘
         │
         ├── [NOT A CANDIDATE] ──► Output "STATUS: SILENT" or "NO SHAKING CADENCE"
         │
         ▼ [CANDIDATE DETECTED]
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Deep Spectro-Temporal CNN Classifier               │
│ • Full 2D Log-Mel Spectrogram (64 Mels × 44 Time Frames)    │
│ • 4 Convolutional Blocks + BatchNorm + Dropout + Dense Head │
│ • 3-Class Softmax: Ambient, Hard-Negative Rattle, Spray Can │
│ • Normalized Shannon Entropy Uncertainty Gating             │
│   (Rejects ambiguous / out-of-distribution sounds)          │
└─────────────────────────────────────────────────────────────┘
         │
         ├── [P(spray) < Conservative Threshold (75%-85%)] ──► "HARD NEGATIVE RATTLE (IGNORED)" / No Alert
         ├── [Entropy >= 0.70 (Ambiguous Signal)]           ──► "UNCERTAIN / AMBIGUOUS SOUND" / No Alert
         │
         ▼ [P(spray) >= Threshold & Entropy < 0.70]
┌─────────────────────────────────────────────────────────────┐
│ MULTI-WINDOW TEMPORAL PERSISTENCE & COOLDOWN GATING         │
│ • Requires >= 3 consecutive windows (>= 0.75s persistence)  │
│   Rules out isolated clicks, single clinks, transient bumps │
│ • Cooldown: 3.0-second quiet window after verified alert    │
│   Prevents notification storming during ongoing episode     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
     🚨 CONFIRMED ALERT: AEROSOL SPRAY CAN SHAKING DETECTED!
```

---

## 2. Training Data & Conservative Threshold Calibration

The detector has been trained on an expanded positive corpus combining:
1. **Your 12 authentic spray-can recordings** (`1.mp3` through `12.mp3`).
2. **The 50-clip dataset** from `C:\Users\User\Downloads\spray_can_shaking_dataset_50_clips` containing 50 real audio crop WAV files across 25 master source recordings (covering mini cans, empty cans, full cans, paint cans, and various distances/angles).
3. **Diverse hard-negative rattling sounds**: coins in jars, keys jangling, pill bottles/Tic-Tacs, loose screws/nuts in tin, glassware clinking, and liquid bottles sloshing.
4. **Ambient noise negatives**: room tone, background speech formants, mechanical keyboard typing, and HVAC fan hums.

### Dataset Partitions (4,900 Total Windows):
- **Train Set**: 3,430 samples (Spray Can Shaking: 2,650, Hard Negatives: 441, Ambient: 339)
- **Validation Set**: 735 samples (Spray Can Shaking: 594, Hard Negatives: 69, Ambient: 72)
- **Test Set**: 735 samples (Spray Can Shaking: 556, Hard Negatives: 96, Ambient: 83)

### Calibration Sweep Results (Unseen 735-Sample Test Partition):
- **Best Validation Spray F1**: **0.9966**
- **Hard-Negative False Alert Rate (FPR)**: **0.0%** at conservative threshold ($\tau = 0.80$)
- **Ambient False Alert Rate**: **0.0%**
- **Spray Can Recall (TPR)**: **97.3% – 99.5%**
- **Precision**: **100.0%**
- **F1 Score**: **0.9863**

---

## 3. Project Architecture: Frontend & Backend Separation

The repository is modularly organized into independent **`frontend/`** and **`backend/`** packages:

```
Spray/
├── frontend/                     # Client-side user interface
│   ├── index.html                # Main UI dashboard and audio diagnostic console
│   ├── css/
│   │   └── style.css             # Dark-mode aesthetic styles, meters, oscilloscope
│   ├── js/
│   │   └── app.js                # Web Audio API streaming, real-time telemetry, admin manager
│   ├── assets/                   # Static branding, architecture diagrams, media
│   └── static/                   # Static assets mirror for backwards compatibility
│
├── backend/                      # Server-side acoustic AI & event detection engine
│   ├── app.py                    # Flask REST API & Web Audio event stream server
│   ├── config.py                 # Hyperparameters, directories, calibration thresholds
│   ├── detector_engine.py        # 2-Stage Acoustic Event Detection (AED) engine
│   ├── stage1_shaking_detector.py # Kinematic shaking burst & cadence pre-filter
│   ├── stage2_classifier.py      # Deep Spectro-Temporal CNN PyTorch classifier
│   ├── audio_utils.py            # Polyphase resampling, Log-Mel spectrogram computation
│   ├── dataset_builder.py        # Dataset slicing, synthetic confounder generation
│   ├── dataset_manager.py        # Dataset versioning & metadata SQLite manager
│   ├── training_manager.py       # Candidate retraining & model rollback manager
│   ├── admin_auth.py             # Administrative authentication & audit logging
│   ├── noise_canceller.py        # Adaptive spectral noise cancellation
│   ├── realtime_mic.py           # CLI real-time microphone stream listener
│   ├── esp32_receiver.py         # Remote ESP32 hardware audio receiver
│   ├── models/                   # Active & candidate CNN weights and calibration metrics
│   ├── data/                     # Raw corpora, review queue, databases, and logs
│   └── tests/                    # Comprehensive unit and integration test suite
│
├── run.py                        # Unified root launcher (starts backend & serves frontend)
├── requirements.txt              # Core Python dependencies
└── vercel.json                   # Serverless deployment configuration
```

---

## 4. Quickstart & Usage

### A. Run the Unified Application
Run the root launcher script to start the backend and automatically serve the frontend:
```bash
python run.py
```
*(Alternatively: `python backend/app.py`)*

Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

### B. Run Real-Time Microphone Detection in Terminal
Continuous acoustic event listener on your physical microphone with ANSI telemetry meters:
```bash
# List available audio devices
python backend/realtime_mic.py --list-devices

# Stream from default microphone
python backend/realtime_mic.py
```

### C. Run the Test Suites
Validate detector behavior, audio resampling, and admin management:
```bash
python -m unittest discover -s backend/tests -p "test_*.py"
```

### D. Retrain Model & Run Threshold Sweep
```bash
# Slices raw audio into dataset partitions
python backend/dataset_builder.py

# Trains SprayCanShakingCNN and sweeps conservative threshold
python backend/train_and_evaluate.py
```

---

## 4. Key Design Decisions

1. **Why Frequency / Loudness Alone Fails:** Keys, coin jars, and glass clinking share overlapping high frequencies ($3 - 8$ kHz) and loud transients with spray can agitator balls. The full 2D spectro-temporal time pattern (rise time $< 10$ ms, hollow cylindrical body resonance $2 - 6$ kHz, periodic bouncing cadence $2.0 - 8.5$ Hz) is required to reliably distinguish them.
2. **Kinematic Pre-Filtering (Stage 1):** Eliminates CPU/GPU waste by rejecting non-shaking ambient audio and silence before neural evaluation.
3. **Multi-Window Persistence ($N \ge 3$):** Single clicks from tapping or putting keys on a table are ruled out because they do not persist over 3 consecutive overlapping windows ($\ge 0.75$s).
4. **Post-Alert Cooldown ($3.0$s):** Once spray can shaking is confirmed, notifications are throttled so a single continuous shaking episode does not trigger dozens of alerts.
