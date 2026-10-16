# Railway Platform Acoustic Spray Can Sound Monitoring: Operator Field Manual & Shadow-Mode Evaluation Protocol

---

## 1. System Overview & Shadow-Mode Protocol

This manual specifies the operational and acoustic data governance protocols for evaluating the **Acoustic Aerosol Spray Can Sound Detector** at railway station platforms.

### 1.1 Shadow-Mode Operating Rule
The detector operates strictly in **Shadow Mode (Advisory Only)**:
- **No Automated Enforcement:** The detector will **never** automatically dispatch security personnel, trigger station sirens, lock turnstiles, or initiate police alerts.
- **Human Visual Verification Required:** Every advisory acoustic alert requires direct human visual verification (e.g., via platform CCTV review or scheduled station patrols) before taking any physical response.
- **Strict Non-Inference Policy:** 
  > **DO NOT infer graffiti activity, criminal intent, or individual guilt from acoustic evidence.**
  > The detector identifies only the physical mechanical acoustic signature of an aerosol canister's internal agitator ball striking canister walls. Audio signals do not identify persons, prove legal intent, or verify that paint was discharged onto station property.

---

## 2. Acoustic Hardware Setup & Microphone Placement

### 2.1 Environmental Noise Challenges
Railway platforms exhibit extreme dynamic range acoustic noise:
- Diesel locomotives and freight pass-bys: **85–95 dBA** low-frequency rumble (40–300 Hz).
- Public address (PA) horn loudspeakers: **75–85 dBA** reverberant speech and chime tones.
- Compressed air brake discharges and wheel flange squeal: **70–90 dBA** high-frequency friction transients (3–9 kHz).
- Acoustic spray can shaking rattle: **55–65 dBA** SPL at 1 meter, decaying to **~45–50 dBA** at 5 meters.

### 2.2 Recommended Microphone Installation
1. **Microphone Type:** Directional cardioid or half-cardioid boundary (PZM) condenser microphone with wide dynamic range ($\ge 110\text{ dB}$) and flat frequency response up to 12 kHz.
2. **Mounting Position:**
   - Mount microphones downward on platform canopies, shelter ceilings, or pillars facing target blind spots or walls.
   - Recommended acoustic standoff distance: **1.0 to 3.5 meters** from potential target surfaces.
   - Maximum standoff distance: **5.0 meters**. Beyond 5.0 meters, platform ambient noise will acoustically mask can-shaking transients, resulting in missed detections.
3. **Resampling Fidelity:**
   - Microphone hardware streams raw native audio (typically 44.1 kHz or 48.0 kHz 16/24-bit PCM).
   - Audio is resampled server-side using **polyphase sinc anti-aliasing** (`scipy.signal.resample_poly` with a Kaiser window filter) down to the model sample rate of **22,050 Hz**.
   - Stopband attenuation is $-57.8\text{ dB}$, eliminating digital aliasing distortions and high-frequency foldover.

---

## 3. Data Governance, Audio Retention & Operator Authorization

To comply with railway telecommunications governance, public transit surveillance regulations, and data protection standards:

### 3.1 Operator Authorization Requirement
- **Mandatory Authorization:** Operational audio capture must be **formally approved by railway site management and telecommunications governance** prior to activating microphones at the station.
- **Scope Limitation:** Monitoring is restricted strictly to designated aerosol graffiti target zones (e.g., train sheds, stabling yards, canopy walls) and must not be used for general public acoustic surveillance.

### 3.2 24-Hour Ephemeral Retention Behavior
- **Raw Continuous Audio Storage:** Continuous microphone audio is never stored permanently. Audio is captured only for empirical shadow-mode benchmark evaluation.
- **Retention Period:** Continuous field session recordings (`data/raw/field_sessions/`) and ephemeral review queue snippets (`data/review_queue/`) are held for a maximum Time-To-Live (TTL) of **24 hours (`AUDIO_RETENTION_HOURS = 24`)**.
- **Deletion Method:** Expired files are deleted using standard operating system unlinking (`Path.unlink()` and `shutil.rmtree()`). This removes filesystem directory pointers and releases storage blocks back to the OS. It is **not** a multi-pass cryptographic disk wipe or hardware shredding.
- **Purge Triggers:** Deletion is triggered automatically during review queue inspection cycles and can be manually initiated by operators at any time:
  ```bash
  curl -X POST http://127.0.0.1:5000/api/field_session/purge_expired
  ```

### 3.3 Access Controls & Data Privacy
- **File System Permissions:** Audio buffers are stored in local application directories accessible only by the dedicated monitoring service process and authenticated site engineering accounts.
- **Anonymized Metadata Persistence:** Only non-personal acoustic descriptors (confidence scores, entropy, margins, duration, condition tags, reviewer timestamps) are retained in structured JSON evaluation logs.

---

## 4. Continuous Audio Capture Workflow

Operators can capture continuous real-site audio streams for empirical benchmark evaluation.

### 4.1 Capturing a Continuous Field Session
1. Navigate to the **Record Ground Truth** or **Field Sessions** tab on the dashboard (`http://127.0.0.1:5000/`).
2. Enter the field session parameters:
   - **Session ID:** e.g., `session_hbf_platform3_north_20261013`
   - **Site Identifier:** e.g., `Station Berlin Hbf - Platform 3`
   - **Microphone ID / Type:** e.g., `AKG P170 Boundary #01`
   - **Operator ID:** e.g., `field_tech_04`
   - **Background Condition:** `quiet_clean`, `busy_station`, `train_passby`, `station_announcement`, or `rain`
   - **Approximate Distance:** `1.0m`, `3.0m`, or `5.0m`
3. Click **Start Continuous Recording**.
4. To finish, click **Stop & Save Session**. The recording is saved to `data/raw/field_sessions/<session_id>/audio.wav` with `metadata.json`.

---

## 5. Trained Reviewer Event Annotation Protocol

Human ground-truth labeling is the foundation of field evaluation. Reviewers must inspect continuous recordings and mark the exact onset and offset times of acoustic events.

### 5.1 Annotation Taxonomy
Reviewers must classify sound intervals into one of five strictly acoustic categories:

| Class Identifier | Description & Acoustic Signatures | Ground Truth Role |
| :--- | :--- | :--- |
| `spray_can_shaking` | **ONLY POSITIVE TARGET:** Repeated, rhythmic metallic impact bursts (2.5–15 Hz cadence) produced by internal steel agitator ball against canister walls. | **Target (Positive)** |
| `aerosol_spray_hiss` | Continuous or pulsed gas hiss, nozzle valve sputtering, cap removal/handling. | **Negative (Confounder)** |
| `hard_negative_rattle` | Non-canister rattling sounds: keys jangling, loose coins, pill bottles, toolboxes, glassware, hardware nuts/bolts. | **Negative (Confounder)** |
| `continuous_environmental` | Railway platform noise: train diesel rumble, wheel screech, track squeal, rain on canopy, HVAC fans. | **Negative (Confounder)** |
| `ambient_and_speech` | Ordinary platform background: passenger conversation, footwear impacts, public address announcements. | **Negative (Confounder)** |

### 5.2 Marking Intervals
1. Use the pre-save audio player or dedicated annotation tool.
2. Mark the exact interval: `start_sec` and `end_sec` (e.g., `start_sec: 14.25`, `end_sec: 18.60`).
3. Select the acoustic class label and acoustic condition.
4. Set reviewer verification confidence (`confirmed_high`, `medium`, `low`).
5. Annotations are saved via `POST /api/field_session/annotate` to `data/raw/field_sessions/<session_id>/annotations.json`.

---

## 6. Session-Level Data Partitioning (Zero Data Leakage)

To avoid optimistic evaluation bias, **all audio recordings from the same session, recording day, or microphone site must remain in the same partition**. Never split chunks from the same continuous recording across training, validation, and test sets.

### 6.1 Partitioning Procedure
Execute the session-isolated partition tool:
```bash
python partition_field_data.py
```
This utility:
1. Groups all files and annotations by `(site_id, session_id, date)`.
2. Allocates groups into three isolated partitions:
   - **`train` (50%):** Available for baseline comparisons and acoustic contrastive modeling.
   - **`val` (25%):** Used for validation analysis and sensitivity tradeoff exploration.
   - **`final_test` (25%):** **Untouched, held-out test split. NEVER used for tuning.**
3. Verifies zero group overlap across splits and computes SHA-256 hashes for all files.
4. Outputs `data/processed/field_split_manifest.json`.

---

## 7. Full Streaming Detector Evaluation

Evaluating streaming detectors requires testing the **complete multi-stage engine on continuous audio**, rather than testing isolated, pre-segmented 1.0-second windows.

### 7.1 Running Streaming Evaluation
Run the canonical streaming evaluator:
```bash
python evaluate_streaming_field.py
```

### 7.2 What the Evaluator Exercises
- **Continuous Sliding Window:** Streams audio with 0.25-second hop intervals (4 evaluations/sec) and a 1.0-second rolling buffer.
- **Stage 1 Kinematic Detector:** Tests for physical impact burst cadence (2.5–15 Hz), burst count ($\ge 3$), and valley contrast.
- **Adaptive Spectral Noise Canceller:** Tracks background noise floor dynamically.
- **Stage 2 Multi-Gated CNN:** Enforces production conservative threshold ($\tau = 0.80$), competing class caps ($\le 0.20$), margin ($\ge 0.35$), and entropy ($< 0.70$).
- **Multi-Window Persistence:** Requires $\ge 2$ consecutive positive windows ($\ge 1.25\text{s}$ confirmation delay).
- **Cooldown Window:** Applies a 3.0-second alert cooldown, collapsing multiple positive windows during an ongoing event into **1 single Alert Episode**.

### 7.3 Output Metrics & Statistical Uncertainty
1. **Event-Level Detection Rate (Recall):** $\text{Detected Target Events} / \text{Total Target Events}$.
   - Accompanied by **Wilson Score 95% Confidence Interval**.
2. **Detection Latency (Delay):** Median and range of $t_{\text{alert}} - t_{\text{start}}$ in seconds.
3. **False Alert Episodes Per Hour (FA/hr):** $\text{False Alert Episodes} / \text{Total Negative Audio Hours}$.
   - Accompanied by **Poisson Exact 95% Confidence Interval** ($\chi^2$ exact quantiles).
4. **The Rule of Three for Short Tests:**
   If zero false alerts are observed across $T$ hours of negative audio, the 95% upper confidence bound on the true platform false alert rate is:
   $$\lambda_{95\%} \approx \frac{-\ln(0.05)}{T} \approx \frac{3.0}{T}\text{ alerts per hour}$$
   *Example:* If a test evaluates 6 hours of negative audio with 0 false alerts, the true false alert rate could still be as high as $3.0 / 6 = 0.50\text{ alerts/hour}$ (1 alert every 2 hours). **Zero false alerts in a short test is NOT a guarantee of zero false alerts.**

---

## 8. Threshold Tuning Governance

- **Production Threshold Locked:** The production threshold remains strictly locked at **$\tau = 0.80$**.
- **No Tuning on Final Test:** **Tuning hyperparameters on the `final_test` set is strictly prohibited.**
- **Future Tuning Workflow:** If threshold adjustments are proposed:
  1. Tune exclusively on the `val` partition.
  2. Document the complete validation Precision-Recall tradeoff curve.
  3. Once a new candidate threshold is approved, evaluate it **once** on the untouched `final_test` set.
  4. Compare the held-out result against the canonical benchmark before any deployment decision.

---

## 9. Deployment Readiness Criteria

> [!CAUTION]
> **FIELD DEPLOYMENT READINESS GATE**
> 
> The spray-can sound detector **CANNOT be declared field-ready for automated or unattended deployment** until the following empirical milestones have been satisfied:
> 
> 1. **Continuous Field Benchmark Duration:** Minimum of **100+ continuous hours** of platform recordings through installed microphones across diverse operating schedules (morning rush hour, off-peak midday, late-night freight, weekend passenger surges).
> 2. **Adverse Weather & Environmental Coverage:** Minimum of **20+ hours** of continuous rain, wind gust, and platform washing conditions with zero unmitigated false alarm bursts.
> 3. **Empirical False Alert Rate:** Confirmed field false alert rate $\le 0.05\text{ false alerts/hour}$ (fewer than 1 false alarm per 20 operating hours) at the 95% Poisson confidence upper bound.
> 4. **Human Review Certification:** Operational staff trained to visually confirm all advisory shadow-mode alerts via CCTV prior to any platform intervention.
