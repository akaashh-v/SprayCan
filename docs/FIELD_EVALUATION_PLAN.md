# Authorized Real-Site Shadow-Mode Field Evaluation Plan
## Precision Acoustic Spray-Can Shaking Sound Monitoring System

> **STATUS: NOT FIELD-VALIDATED**  
> *Notice: Operational field evaluation at railway sites is strictly blocked until written site authorization and privacy/governance approvals are signed and on file. All testing documented herein uses existing or synthetic sample audio. Live microphone recording on platform property is NOT active.*

---

## 1. Executive Summary & Regulatory Authority

This document defines the formal operational plan for the real-site, shadow-mode field evaluation of the **Two-Stage Acoustic Aerosol Spray-Can Shaking Sound Detector**.

### 1.1 Governance Authority & Approval Gates (Requirement 1)
Prior to connecting any physical microphone to live recording equipment at a railway station, train maintenance shed, or stabling yard, the following written authorizations **MUST** be signed, filed in the project governance registry, and countersigned by the Lead Project Engineer:

| Approval Role | Responsible Entity | Required Sign-off Scope | Approval Status |
| :--- | :--- | :--- | :--- |
| **Railway Site Director** | Station / Infrastructure Authority | Physical access, mounting locations, electrical safety, non-interference with train operations | **PENDING (Unresolved)** |
| **Data Protection Officer (DPO)** | Corporate Privacy & Legal Office | Compliance with transit surveillance laws, GDPR/national privacy statutes, passenger acoustic privacy | **PENDING (Unresolved)** |
| **Telecom / IT Security Lead** | Rail Network Communications | Network isolation, restricted filesystem access, ephemeral storage compliance | **PENDING (Unresolved)** |
| **Lead Acoustics Engineer** | System Engineering Team | Microphone calibration, production pipeline integrity, sensor health monitoring | **ACTIVE (Local Ready)** |

> [!CAUTION]
> **STOP CONDITION:** If written railway-site authorization or privacy/governance approval is absent, incomplete, or revoked, **ALL SITE AUDIO CAPTURE IS STRICTLY PROHIBITED**. The evaluation must immediately stop at planning and local synthetic/sample audio testing.

---

## 2. Operating Principles & Shadow-Mode Invariants (Requirement 2)

During the field evaluation phase, the detection system operates under strict shadow-mode constraints:

1. **Threshold Lock ($\tau = 0.80$):**
   - The production conservative threshold is locked at $\tau = 0.80$ (`CONSERVATIVE_THRESHOLD = 0.80`).
   - Retraining the neural network or modifying decision thresholds during field data collection is strictly prohibited.
2. **Advisory Alerts Only:**
   - Alerts are logged to `data/shadow_mode_alerts.jsonl` with flag `shadow_mode=True`, `dispatch_authorized=False`, and `automatic_enforcement=False`.
   - The system is completely decoupled from security dispatch, sirens, turnstiles, or law enforcement notification.
3. **Mandatory Human Visual Verification:**
   - Every acoustic alert is purely advisory and requires an authorized human operator to inspect platform CCTV or conduct a visual patrol before taking any action.
4. **Strict Acoustic Non-Inference Policy:**
   - **Audio signals measure physical mechanical vibrations only.** The system detects the resonant rattle of an internal steel ball striking canister walls.
   - **Operators and reviewers must NOT infer graffiti activity, vandalism intent, guilt, or identity from audio recordings.** Audio data cannot establish who is holding an object or what their legal intentions are.

---

## 3. Current Feature Verification: Implemented vs. Proposed

An audit of the active codebase establishes the exact boundary between operational code and proposed field extensions:

| System Component | Implemented in Project Code | Proposed / Planned for Field Deployment | Verification Status |
| :--- | :--- | :--- | :--- |
| **Microphone Ingestion** | Browser Web Audio API recording (`POST /api/record_sample`), continuous session API (`POST /api/field_session/save_continuous`), local terminal mic streaming (`realtime_mic.py`), polyphase sinc anti-aliasing resampling (`audio_utils.py` to 22,050 Hz). | Hardware daemon for remote PoE/IP platform boundary microphones; automatic stream auto-reconnect on network drop. | **Partially Implemented** (Local & API ready; remote IP mic daemon proposed) |
| **Ground Truth Annotation** | Structured session annotation API (`POST /api/field_session/annotate`), multi-class taxonomy validation against `CLASS_NAMES`, confidence levels, start/end interval validation. | Dedicated multi-hour visual waveform scrubber UI; inter-reviewer Cohen's Kappa agreement calculation. | **Implemented & Verified** (Scrubber UI enhancement proposed) |
| **Data Retention & Privacy** | Automatic TTL purge after 24 hours (`AUDIO_RETENTION_HOURS = 24`, `_purge_expired_review_items()`, `POST /api/field_session/purge_expired`), standard filesystem unlinking (`Path.unlink`, `shutil.rmtree`), bounded review queue (50 items). | Role-based JWT/TLS API authentication; certified cryptographic storage erasure (`srm` / NIST SP 800-88); encrypted volume containers for locked final-test split. | **Implemented & Verified** (Hardware cryptographic wipe proposed) |
| **Shadow Alerts & Review** | Append-only JSONL shadow alert log (`data/shadow_mode_alerts.jsonl`), review queue metadata (`data/review_queue/`), event-end detection with rearming after $\ge 2$ quiet windows, ambiguous gap tagging (`UNCERTAIN_AMBIGUOUS_SOUND`). | Direct Webhook integration into railway CCTV Video Management System (VMS) displays. | **Implemented & Verified** (VMS integration proposed) |
| **Evaluation Framework** | Streaming continuous evaluator (`evaluate_streaming_field.py`), episode clustering, event matching, latency tracking, Wilson score & exact Poisson 95% CIs, Rule of Three ($3.0/T$), chronological & site-held-out partitioning (`partition_field_data.py`). | Batch evaluation orchestrator across multi-site distributed servers. | **Implemented & Verified** |

---

## 4. Field Capture Protocol & Hardware Configuration (Requirement 3)

### 4.1 Authorized Capture Boundaries
- **Spatial Boundaries:** Monitoring is restricted strictly to designated test areas (e.g., Station Berlin Hbf - Platform 3 West Canopy or Maintenance Yard Bay 4). Microphones must not be directed toward ticket counters, passenger waiting rooms, or commercial retail stalls.
- **Temporal Boundaries:** Continuous recording is permitted only during approved collection shifts (e.g., Monday 06:00 to Friday 22:00 during scheduled test blocks).

### 4.2 Acoustic Hardware & Signal Pipeline
1. **Microphone Specification:** Directional cardioid or boundary (PZM) condenser microphone, SPL capability $\ge 120\text{ dB}$, frequency response flat within $\pm 2\text{ dB}$ across 100 Hz – 11,000 Hz.
2. **Mounting:** Mounted downward on ceiling/canopy beam at a standoff distance of **1.5m – 3.0m** from target surfaces. Standoff distance must never exceed 5.0m.
3. **Resampling Pipeline:** Hardware captures native PCM (44.1 kHz or 48.0 kHz 24-bit). The production pipeline resamples audio to **22,050 Hz mono** using polyphase sinc anti-aliasing (`scipy.signal.resample_poly` with a Kaiser window, stopband attenuation $-57.8\text{ dB}$) to preserve transient canister ringing without digital aliasing.
4. **Metadata Logging:** Every session must record:
   - Active model SHA-256 hash (`ecac9edfa5012e4770460ccc740d495256afe54c0893f7273f226ca59682bec5`)
   - Configuration parameters ($\tau = 0.80$, $\Delta t_{\text{hop}} = 0.25\text{s}$, $T_{\text{win}} = 1.0\text{s}$)
   - Microphone ID, hardware gain setting, site ID, operator ID, start/stop epoch, and acoustic condition tags.

---

## 5. Event Labeling Process & Controlled Testing Safety (Requirement 4)

Human ground-truth labeling provides the objective baseline against which detector performance is measured.

### 5.1 Labeling Taxonomy
Reviewers must label every continuous segment into one of five mutually exclusive acoustic categories:

| Acoustic Class Label | Physical Description & Key Transients | Ground-Truth Category |
| :--- | :--- | :--- |
| `spray_can_shaking` | **SOLE POSITIVE TARGET:** Rhythmic metallic impact bursts (cadence 2.5–15.0 Hz, 3–9 kHz modal cavity resonance) from internal steel ball striking canister wall. | **Target Event (Positive)** |
| `aerosol_spray_hiss` | Continuous or pulsed gas discharge hiss, nozzle turbulence, cap removal. No shaking impact cadence. | **Confounder (Negative)** |
| `hard_negative_rattle` | Keys jangling, loose coins, pill bottles, toolboxes, loose bolts, rattling glassware. | **Confounder (Negative)** |
| `continuous_environmental` | Diesel locomotive rumble, brake air exhaust, wheel flange squeal, rain on canopy, HVAC fans. | **Confounder (Negative)** |
| `ambient_and_speech` | Footsteps, platform vocal chatter, public address announcements, baggage wheels. | **Confounder (Negative)** |

### 5.2 Controlled Shaking Safety Rules
When gathering positive shake samples for detection sensitivity benchmarking:
1. **Safe Authorized Zone Only:** Conduct tests only in an approved maintenance bay or platform area under safety escort.
2. **Zero Paint Discharge:** **NEVER spray paint, discharge aerosol contents, or create any physical mark or vandalism event.** Use exhausted or clean water aerosol test canisters with intact agitator balls.
3. **Log Contextual Factors:** Document can type (standard 400ml, mini 150ml, tall 600ml), fill state (full, half, near-empty), standoff distance (1.0m, 3.0m, 5.0m), and shaking orientation (vertical, horizontal, angled).

---

## 6. Sample Sizing Mathematics & Statistical Validity (Requirement 5)

Validating railway field readiness requires statistically defensible sample sizes for **both** false alerts and event detection rate. **100 hours of continuous negative audio alone is NOT adequate validation**, because negative audio measures only false alert rate ($\text{FA/hr}$), providing zero statistical power regarding event recall.

### 6.1 Negative Audio Exposure Requirement ($\ge 100\text{ hours}$)
- **Target Exposure:** At least 100 hours of continuous, unattended platform recordings containing station environmental sounds without target events.
- **Target False Alert Rate:** $\le 0.05\text{ false alert episodes per hour}$ (fewer than 1 false alarm per 20 operating hours).
- **Poisson Rule of Three Bound:** If $k = 0$ false alerts are observed across $T = 100$ hours, the exact Poisson 95% upper confidence bound is:
  $$\lambda_{\text{upper}} = \frac{-\ln(0.05)}{T} = \frac{2.9957}{100} \approx 0.0300\text{ FA/hr}$$
  This mathematically certifies that the true platform false alert rate is below $0.05\text{ FA/hr}$ at 95% confidence.

### 6.2 Positive Event Sample Sizing Criteria
To measure event detection rate (Recall, $p$) with a meaningful margin of error, sample size $N$ must satisfy the binomial sampling criterion:
$$N \ge \frac{z_{1-\alpha/2}^2 \cdot p (1 - p)}{E^2}$$
For 95% confidence ($z = 1.96$), an expected detection rate $p = 0.85$, and a desired margin of error $E \le 0.07$ (confidence interval width $\le 14\%$):
$$N \ge \frac{(1.96)^2 \cdot (0.85)(0.15)}{(0.07)^2} = \frac{3.8416 \cdot 0.1275}{0.0049} \approx 100\text{ positive events}$$

### 6.3 Condition-Level Sizing Quotas
Platform acoustics vary drastically across environmental states. Positive events must be distributed across all five operational conditions to compute condition-specific Wilson score 95% confidence intervals:

| Acoustic Operating Condition | Target Negative Exposure | Target Verified Shake Events ($N_c$) | Justification / Acoustic Challenge |
| :--- | :--- | :--- | :--- |
| **Quiet Platform** (off-peak ambient) | 30.0 hours | $\ge 25$ events | Baseline sensitivity and clean rattle transients |
| **Train Passby & Braking** (85–95 dBA) | 25.0 hours | $\ge 30$ events | Acoustic masking by high-amplitude low-frequency rumble |
| **Station PA Announcements** (chimes/speech) | 15.0 hours | $\ge 25$ events | Reverberant speech syllable and chime transient rejection |
| **Rain & Environmental** (canopy drip/wind) | 20.0 hours | $\ge 25$ events | Broadband stochastic rain masking vs kinematic cadence |
| **Passenger Crowd Chatter** (rush hour footsteps) | 10.0 hours | $\ge 25$ events | Footwear impact bursts and vocal murmur overlap |
| **TOTAL FIELD EVALUATION** | **$\ge 100.0$ hours** | **$\ge 130$ events** | **Statistically robust coverage of platform operating envelope** |

---

## 7. Data Partitioning & Split Isolation (Requirement 6)

To eliminate data leakage and optimistic evaluation bias:

1. **Chronological Splitting Strategy (`split_strategy="chronological"`):**
   - Sessions are grouped strictly by `session_id`.
   - Continuous sessions are ordered by timestamp. Earlier continuous sessions are assigned to `train` (50%) and `val` (25%).
   - The latest continuous days or time blocks are reserved strictly for the untouched `final_test` (25%).
2. **Session Preservation:**
   - All clips and ground-truth annotations from a continuous recording session remain intact within the same partition. Chunks from the same continuous file **never** cross partition boundaries.
3. **Cross-Split Site Audit:**
   - The partition manifest automatically logs `site_cross_split_audit` reporting whether sessions from the same physical site appear across splits (`same_site_in_multiple_splits`).
4. **Site-Held-Out Strategy (`split_strategy="site_held_out"`):**
   - When evaluating model transfer to new stations, entire sites are held out separately. No site in `final_test` may appear in `train` or `val`.

---

## 8. Data Retention, Access Controls & Secure Erasure (Requirement 7)

### 8.1 Default Ephemeral Retention (24-Hour TTL)
- Continuous recordings (`data/raw/field_sessions/`) and candidate audio clips (`data/review_queue/`) are stored temporarily for empirical analysis.
- Audio is automatically deleted after **24 hours (`AUDIO_RETENTION_HOURS = 24`)**.
- Deletion is triggered by `POST /api/field_session/purge_expired` and before review queue retrieval.

### 8.2 File Deletion vs. Secure Cryptographic Erasure
- **Technical Clarification:** Default deletion uses standard operating system unlinking (`Path.unlink()` and `shutil.rmtree()`). Standard unlinking removes filesystem directory entries and marks storage blocks as available for reuse. **It does NOT perform cryptographic wiping or physical block shredding.**
- **High-Security Option:** If transit authority governance mandates secure erasure (e.g. NIST SP 800-88 Rev 1 Guidelines for Media Sanitization), storage volumes must be provisioned with Full Disk Encryption (LUKS/BitLocker) with cryptographic key shredding upon decommission, or wiped using secure multi-pass overwrite tools (`srm -z`).

### 8.3 Locked Final-Test Benchmark Subset
If railway governance approves archiving a locked benchmark test subset for ongoing detector regression testing:
- **Authorization Reference:** Formally documented under Governance Authorization ID (e.g. `RAIL-GOV-2026-TESTSET-01`).
- **Access Restrictions:** Read-only access restricted exclusively to the Lead Model Evaluator. Stored on an encrypted volume separated from live streaming networks.
- **Mandatory Destruction Date:** Fixed calendar deletion date specified at time of authorization (maximum 90 days from collection completion).

---

## 9. Streaming Evaluation Protocol & Reporting (Requirement 8)

The continuous evaluation must exercise the **complete streaming engine**, not isolated 1.0-second files:

### 9.1 Evaluation Pipeline Execution
```bash
# 1. Partition field sessions into chronological train/val/test splits
python partition_field_data.py --strategy chronological

# 2. Run streaming evaluation on continuous field recordings
python evaluate_streaming_field.py
```

### 9.2 Metric Reporting Standards
The evaluation report must report:
1. **Total Negative Audio Hours ($T$):** Total duration excluding target event intervals $[t_{\text{start}}-0.25\text{s}, t_{\text{end}}+1.5\text{s}]$.
2. **Verified Positive Events ($N$):** Count of ground-truth shake intervals.
3. **Event-Level Detection Rate (Recall):** $\text{Detected Events} / \text{Total Target Events}$ with **Wilson Score 95% Confidence Interval**.
4. **Missed Target Events:** Count of target intervals with zero matching alert episodes.
5. **Detection Delay:** Latency from physical event onset to alert firing ($t_{\text{alert}} - t_{\text{start}}$ in seconds, reporting mean, median, min, max).
6. **False Alert Episodes Per Hour:** Count of false alert episodes divided by negative hours, with **Poisson Exact 95% Confidence Interval** ($\chi^2$ quantiles) and Rule of Three ($3.0 / T$).
7. **Condition Breakdown:** Independent metric rows for Quiet Platform, Train Passby, Announcements, Rain/Weather, and Crowd Chatter.
8. **Synthetic vs. Real Isolation:** Results from synthetic noise mixtures must be reported separately from real-site platform recordings. Never combine synthetic benchmark scores with installed-microphone metrics.

---

## 10. Circuit Breakers: Pause & Stop Procedures (Requirement 9)

Operators must immediately pause or terminate field audio collection if any of the following failure triggers occur:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   FIELD COLLECTION CIRCUIT BREAKERS                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
 [Trigger A: Sensor]         [Trigger B: Boundary]       [Trigger C: Privacy]
 Mic clipping >= 10%        Capture outside approved    Intelligible private speech
 DC offset drift > 0.1      zone or time window         or passenger distress call
 Signal loss > 30s          Mounting bracket shift      Security tactical voice
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    ▼
                       [IMMEDIATE CIRCUIT BREAKER]
                   1. Stop recording stream immediately
                   2. Quarantine and purge affected audio
                   3. File Incident Log to Rail Governance
                   4. Do NOT resume without recertification
```

### Circuit Breaker Incident Actions:
1. **Trigger A (Microphone Hardware Failure / Clipping):**
   - *Condition:* Digital clipping ratio $\ge 10\%$ over 5 consecutive windows, DC offset drift $> 0.1$, or continuous flatline/silence $> 30\text{s}$.
   - *Action:* Stop recording immediately. Check microphone cable, pre-amplifier gain, and phantom power. Inspect microphone diaphragm for water ingress.
2. **Trigger B (Boundary or Authorization Breach):**
   - *Condition:* Audio captured outside authorized test area, outside permitted calendar hours, or microphone shifted from approved mounting bracket.
   - *Action:* Halt capture instantly. Delete unauthorized audio buffer via `POST /api/field_session/purge_expired`. Log incident to Governance Registry.
3. **Trigger C (Unexpected Sensitive Acoustic Content):**
   - *Condition:* Intelligible private passenger conversation near microphone, distress/medical emergency vocalizations, or security staff tactical communications.
   - *Action:* Immediately stop recording. Quarantine session folder. Notify Railway Data Protection Officer. Delete audio file without running model evaluation.
4. **Trigger D (Retention or Access Control Failure):**
   - *Condition:* Unpurged audio older than 24 hours discovered, or unauthorized read access detected.
   - *Action:* Terminate field collection. Execute emergency manual purge. Recertify filesystem permissions before resuming.

---

## 11. Pre-Field Checklist & Operator Workflow (Requirement 10)

### 11.1 Pre-Field Readiness Checklist
Before traveling to a station or powering on collection equipment, verify each gate:

- [ ] **Gate 1: Written Governance Sign-off:** Railway site authorization and DPO privacy approvals signed and archived.
- [ ] **Gate 2: Shadow Mode Confirmed:** `SHADOW_MODE_ENABLED = True`, `dispatch_authorized = False`, `automatic_enforcement = False` in `config.py`.
- [ ] **Gate 3: Threshold Locked:** Production threshold confirmed at $\tau = 0.80$ (`CONSERVATIVE_THRESHOLD = 0.80`).
- [ ] **Gate 4: Checkpoint Hash Verified:** Model checkpoint matches SHA-256 `ecac9edfa5012e4770460ccc740d495256afe54c0893f7273f226ca59682bec5`.
- [ ] **Gate 5: Resampling Validated:** Polyphase sinc anti-aliasing confirmed with Kaiser window filter down to 22,050 Hz.
- [ ] **Gate 6: Retention Purge Functional:** Verified automated deletion of files exceeding 24-hour TTL.
- [ ] **Gate 7: Safe Test Canisters Prepared:** Controlled shake canisters checked: agitator balls intact, zero paint discharge, no graffiti creation.

### 11.2 Step-by-Step Operator Workflow
1. **Initialize System:** Start web dashboard (`python app.py`) or connect capture terminal.
2. **Enter Session Metadata:** In the dashboard, enter `session_id`, `site_id`, `microphone_id`, `operator_id`, and `condition`.
3. **Verify Signal Quality:** Verify live audio input level. Ensure background RMS is between $0.005$ and $0.05$ without digital clipping.
4. **Start Continuous Recording:** Click **Start Continuous Recording**. Note start time in operator physical logbook.
5. **Conduct Controlled Shake Trials (if scheduled):**
   - In approved test zone, perform verified can shake (e.g., 2.0s vigorous shake).
   - Record exact trial epoch timestamp and distance in test log.
   - Ensure a clean quiet interval ($\ge 3\text{s}$) between trials.
6. **Stop & Save Session:** Click **Stop & Save Session**. Confirm session folder created under `data/raw/field_sessions/<session_id>/`.
7. **Perform Interval Ground-Truth Annotation:** Open annotation panel, review audio, and mark exact start/end seconds and class labels.
8. **Execute Partitioning & Evaluation:** Run `partition_field_data.py --strategy chronological` followed by `evaluate_streaming_field.py`.
9. **Verify Automatic Purge:** Verify that recordings older than 24 hours are removed by retention cleanup.

---

## 12. Deployment Readiness Gates (Go / No-Go Milestones)

The detector **CANNOT** be declared field-ready for automated alerting until all four readiness gates are certified:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     FIELD READINESS GATES (GO / NO-GO)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
  Gate 1: Exposure ───► >= 100 continuous negative hours on installed mic
  Gate 2: Weather ────► >= 20 continuous hours of adverse rain/wind
  Gate 3: Precision ──► <= 0.05 False Alerts/hr at Poisson 95% upper bound
  Gate 4: Recall ─────► >= 85% event recall across all 5 conditions (Wilson CI)
  Gate 5: Review ─────► Human visual verification SOP certified by station team
```

*If any gate fails, the system remains strictly in **SHADOW MODE (NOT FIELD-VALIDATED)**.*
