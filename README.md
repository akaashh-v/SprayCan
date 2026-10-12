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
- **Hard-Negative False Alert Rate (FPR)**: **0.0%** at conservative threshold ($\tau = 0.95$)
- **Ambient False Alert Rate**: **0.0%**
- **Spray Can Recall (TPR)**: **97.3% – 99.5%**
- **Precision**: **100.0%**
- **F1 Score**: **0.9863**

---

## 3. Quickstart & Usage

### A. Run the Interactive Web Dashboard
The web app includes real-time microphone streaming, canvas oscilloscope waveform, 1-click sandbox tests, and file upload diagnostics:
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

### B. Run Real-Time Microphone Detection in Terminal
Continuous acoustic event listener on your physical microphone with ANSI telemetry meters:
```bash
# List available audio devices
python realtime_mic.py --list-devices

# Stream from default microphone
python realtime_mic.py

# Or select a specific microphone device index
python realtime_mic.py --device 1
```

### C. Test One-Click Demo Samples & Live Diagnostics
Run the web dashboard and click any of the built-in one-click demo test buttons:
- **Authentic Spray Can Samples:** Standard aerosol, mini can, empty can, spray paint metal ball rattle.
- **Hard-Negative Samples (Must NOT alert):** Keys jangling, coins in jar, pill bottles, glassware clinking, liquid bottle sloshing.
- **Ambient / Speech Samples (Must NOT alert):** Ambient room silence, background speech formants.

### D. Retrain Model & Run Threshold Sweep
```bash
# Slices raw audio into dataset partitions
python dataset_builder.py

# Trains SprayCanShakingCNN and sweeps conservative threshold
python train_and_evaluate.py
```

---

## 4. Key Design Decisions

1. **Why Frequency / Loudness Alone Fails:** Keys, coin jars, and glass clinking share overlapping high frequencies ($3 - 8$ kHz) and loud transients with spray can agitator balls. The full 2D spectro-temporal time pattern (rise time $< 10$ ms, hollow cylindrical body resonance $2 - 6$ kHz, periodic bouncing cadence $2.0 - 8.5$ Hz) is required to reliably distinguish them.
2. **Kinematic Pre-Filtering (Stage 1):** Eliminates CPU/GPU waste by rejecting non-shaking ambient audio and silence before neural evaluation.
3. **Multi-Window Persistence ($N \ge 3$):** Single clicks from tapping or putting keys on a table are ruled out because they do not persist over 3 consecutive overlapping windows ($\ge 0.75$s).
4. **Post-Alert Cooldown ($3.0$s):** Once spray can shaking is confirmed, notifications are throttled so a single continuous shaking episode does not trigger dozens of alerts.
