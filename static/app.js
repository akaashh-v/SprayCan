/**
 * Front-end Controller for Precision Aerosol Spray Can Sound Detector.
 * Manages Web Audio API microphone streaming, canvas visualizer,
 * real-time two-stage telemetry meters, sandbox testing, and file diagnostics.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let isListening = false;
    let audioContext = null;
    let micStream = null;
    let scriptProcessor = null;
    let analyserNode = null;
    let micAlertCount = 0;
    let pollInterval = null;

    // Audio parameters & buffers
    const TARGET_SAMPLE_RATE = 22050;
    const WINDOW_SAMPLES = 22050;
    let nativeSampleRate = 22050;
    let nativeWindowSamples = 22050;
    let nativeBuffer = new Float32Array(22050);
    let nativeBufferWriteIdx = 0;

    let systemInfo = {
        conservative_threshold: 0.80,
        required_consecutive_windows: 2,
        cooldown_seconds: 0.5,
        enable_noise_cancellation: true,
    };

    // DOM Elements
    const elements = {
        systemStatusBadge: document.getElementById("system-status-badge"),
        systemStatusDot: document.getElementById("system-status-dot"),
        systemStatusText: document.getElementById("system-status-text"),
        badgeThresholdVal: document.getElementById("badge-threshold-val"),
        badgeCooldownVal: document.getElementById("badge-cooldown-val"),
        ncBadge: document.getElementById("nc-badge"),
        badgeNcVal: document.getElementById("badge-nc-val"),
        btnToggleNc: document.getElementById("btn-toggle-nc"),

        meterSnrVal: document.getElementById("meter-snr-val"),
        meterSnrFill: document.getElementById("meter-snr-fill"),
        meterNrVal: document.getElementById("meter-nr-val"),
        meterNrFill: document.getElementById("meter-nr-fill"),
        meterNoiseFloorVal: document.getElementById("meter-noise-floor-val"),

        mainStatusBanner: document.getElementById("main-status-banner"),
        bannerStatusEmoji: document.getElementById("banner-status-emoji"),
        bannerHeadline: document.getElementById("banner-headline"),
        bannerDescription: document.getElementById("banner-description"),
        persistenceSubText: document.getElementById("persistence-sub-text"),
        persistenceDotsContainer: document.getElementById("persistence-dots-container"),

        // Sensitivity controls
        currentThresholdLabel: document.getElementById("current-threshold-label"),
        presetSensitive: document.getElementById("preset-sensitive"),
        presetBalanced: document.getElementById("preset-balanced"),
        presetConservative: document.getElementById("preset-conservative"),
        thresholdSlider: document.getElementById("threshold-slider"),
        canvasRateTag: document.getElementById("canvas-rate-tag"),
        micDeviceName: document.getElementById("mic-device-name"),

        btnToggleMic: document.getElementById("btn-toggle-mic"),
        micBtnIcon: document.getElementById("mic-btn-icon"),
        micBtnText: document.getElementById("mic-btn-text"),
        micAlertCount: document.getElementById("mic-alert-count"),
        waveformCanvas: document.getElementById("waveform-canvas"),
        canvasRmsTag: document.getElementById("canvas-rms-tag"),

        // Stage 1 meters
        meterRmsVal: document.getElementById("meter-rms-val"),
        meterRmsFill: document.getElementById("meter-rms-fill"),
        meterCadenceVal: document.getElementById("meter-cadence-val"),
        meterCadenceFill: document.getElementById("meter-cadence-fill"),
        meterValleyVal: document.getElementById("meter-valley-val"),
        meterValleyFill: document.getElementById("meter-valley-fill"),
        meterCvVal: document.getElementById("meter-cv-val"),
        meterCvFill: document.getElementById("meter-cv-fill"),
        meterMetalVal: document.getElementById("meter-metal-val"),
        meterMetalFill: document.getElementById("meter-metal-fill"),
        stage1VerdictText: document.getElementById("stage1-verdict-text"),

        // Stage 2 meters
        meterSprayVal: document.getElementById("meter-spray-val"),
        meterSprayFill: document.getElementById("meter-spray-fill"),
        conservativeMarker: document.getElementById("conservative-threshold-marker"),
        lblConservativeVal: document.getElementById("lbl-conservative-val"),
        meterHissVal: document.getElementById("meter-hiss-val"),
        meterHissFill: document.getElementById("meter-hiss-fill"),
        meterHardnegVal: document.getElementById("meter-hardneg-val"),
        meterHardnegFill: document.getElementById("meter-hardneg-fill"),
        meterImpactVal: document.getElementById("meter-impact-val"),
        meterImpactFill: document.getElementById("meter-impact-fill"),
        meterEnvVal: document.getElementById("meter-env-val"),
        meterEnvFill: document.getElementById("meter-env-fill"),
        meterAmbientVal: document.getElementById("meter-ambient-val"),
        meterAmbientFill: document.getElementById("meter-ambient-fill"),
        meterEntropyVal: document.getElementById("meter-entropy-val"),
        meterEntropyFill: document.getElementById("meter-entropy-fill"),

        // Log
        eventStreamContainer: document.getElementById("event-stream-container"),
        btnClearLog: document.getElementById("btn-clear-log"),

        // Sandbox
        sandboxGrid: document.getElementById("sandbox-grid"),
        sandboxPlayerCard: document.getElementById("sandbox-player-card"),
        playerSampleTitle: document.getElementById("player-sample-title"),
        sandboxAudioPlayer: document.getElementById("sandbox-audio-player"),
        sandboxEvalOutput: document.getElementById("sandbox-eval-output"),

        // Upload
        uploadDropzone: document.getElementById("upload-dropzone"),
        fileInput: document.getElementById("file-input"),
        btnBrowseFile: document.getElementById("btn-browse-file"),
        fileEvalResult: document.getElementById("file-eval-result"),
        resDuration: document.getElementById("res-duration"),
        resWindows: document.getElementById("res-windows"),
        resCandidates: document.getElementById("res-candidates"),
        resAlerts: document.getElementById("res-alerts"),
        spectrogramImage: document.getElementById("spectrogram-image"),
        timelineList: document.getElementById("timeline-list"),

        // Calibration
        btnRefreshCalibration: document.getElementById("btn-refresh-calibration"),
        calibSelectedThreshold: document.getElementById("calib-selected-threshold"),
        calibrationTableBody: document.getElementById("calibration-table-body"),
    };

    // -------------------------------------------------------------------------
    // Tab Navigation
    // -------------------------------------------------------------------------
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const targetPaneId = tab.getAttribute("data-tab");
            navTabs.forEach((t) => t.classList.remove("active"));
            tabPanes.forEach((p) => p.classList.remove("active"));
            tab.classList.add("active");
            const targetPane = document.getElementById(targetPaneId);
            if (targetPane) targetPane.classList.add("active");

            if (targetPaneId === "tab-sandbox") loadDemoSamples();
            if (targetPaneId === "tab-calibration") loadCalibrationReport();
        });
    });

    // -------------------------------------------------------------------------
    // Logging Utility
    // -------------------------------------------------------------------------
    function addLogEntry(message, type = "info") {
        if (!elements.eventStreamContainer) return;
        const now = new Date();
        const timeStr = now.toTimeString().split(" ")[0];

        const row = document.createElement("div");
        row.className = `log-entry log-entry-${type}`;
        row.innerHTML = `<span class="log-time">${timeStr}</span> <span class="log-msg">${message}</span>`;

        elements.eventStreamContainer.prepend(row);
        while (elements.eventStreamContainer.children.length > 60) {
            elements.eventStreamContainer.removeChild(elements.eventStreamContainer.lastChild);
        }
    }

    if (elements.btnClearLog) {
        elements.btnClearLog.addEventListener("click", () => {
            elements.eventStreamContainer.innerHTML = "";
            addLogEntry("Log stream cleared.", "info");
        });
    }

    // -------------------------------------------------------------------------
    // System Status Initialization
    // -------------------------------------------------------------------------
    async function initSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            if (data.model_loaded) {
                elements.systemStatusDot.style.backgroundColor = "var(--accent-green)";
                elements.systemStatusText.innerText = "SYSTEM ACTIVE";
            } else {
                elements.systemStatusDot.style.backgroundColor = "var(--accent-amber)";
                elements.systemStatusText.innerText = "MODEL STANDBY";
            }

            systemInfo.conservative_threshold = data.conservative_threshold || 0.80;
            systemInfo.required_consecutive_windows = data.required_consecutive_windows || 2;
            systemInfo.cooldown_seconds = data.cooldown_seconds !== undefined ? data.cooldown_seconds : 0.5;
            systemInfo.enable_noise_cancellation = data.enable_noise_cancellation !== undefined ? data.enable_noise_cancellation : true;

            const threshPct = Math.round(systemInfo.conservative_threshold * 100);
            updateSensitivityUI(systemInfo.conservative_threshold);
            updateNoiseCancellationUI(systemInfo.enable_noise_cancellation);
            renderPersistenceDots(0, systemInfo.required_consecutive_windows);
            elements.badgeCooldownVal.innerText = `${systemInfo.cooldown_seconds}s`;

            addLogEntry(
                `Detector engine ready: Noise Cancellation=${systemInfo.enable_noise_cancellation ? "ON" : "OFF"}, Threshold=${threshPct}%, Persistence=${systemInfo.required_consecutive_windows} windows`,
                "info"
            );
        } catch (err) {
            elements.systemStatusText.innerText = "OFFLINE";
            elements.systemStatusDot.style.backgroundColor = "var(--accent-red)";
            console.error("Failed to fetch system status:", err);
        }
    }

    // Dynamic Persistence Dots Renderer
    function renderPersistenceDots(count, maxRequired) {
        if (!elements.persistenceDotsContainer) return;
        elements.persistenceDotsContainer.innerHTML = "";
        for (let i = 0; i < maxRequired; i++) {
            const span = document.createElement("span");
            span.className = `p-dot ${i < count ? "active" : ""}`;
            elements.persistenceDotsContainer.appendChild(span);
        }
    }

    // High quality linear interpolation resampler for browser audio
    function resampleLinear(inputBuffer, inputSampleRate, targetSampleRate, targetLength) {
        if (inputSampleRate === targetSampleRate && inputBuffer.length === targetLength) {
            return inputBuffer;
        }
        const output = new Float32Array(targetLength);
        const ratio = (inputBuffer.length - 1) / (targetLength - 1);
        for (let i = 0; i < targetLength; i++) {
            const srcPos = i * ratio;
            const srcIdx = Math.floor(srcPos);
            const frac = srcPos - srcIdx;
            const s0 = inputBuffer[srcIdx] || 0;
            const s1 = inputBuffer[srcIdx + 1] !== undefined ? inputBuffer[srcIdx + 1] : s0;
            output[i] = s0 + frac * (s1 - s0);
        }
        return output;
    }

    // Update sensitivity UI and sync with backend
    function updateSensitivityUI(thresh) {
        const pct = Math.round(thresh * 100);
        if (elements.badgeThresholdVal) elements.badgeThresholdVal.innerText = `${pct}%`;
        if (elements.lblConservativeVal) elements.lblConservativeVal.innerText = `${pct}%`;
        if (elements.conservativeMarker) elements.conservativeMarker.style.left = `${pct}%`;
        if (elements.thresholdSlider) elements.thresholdSlider.value = pct;
        if (elements.currentThresholdLabel) {
            let labelText = `${pct}%`;
            if (pct >= 92) labelText += " (Ultra-Conservative)";
            else if (pct >= 80) labelText += " (Balanced Precision)";
            else labelText += " (High Sensitivity)";
            elements.currentThresholdLabel.innerText = labelText;
        }

        [elements.presetSensitive, elements.presetBalanced, elements.presetConservative].forEach((btn) => {
            if (!btn) return;
            const btnVal = Math.round(parseFloat(btn.getAttribute("data-thresh")) * 100);
            if (Math.abs(btnVal - pct) <= 2) btn.classList.add("active");
            else btn.classList.remove("active");
        });
    }

    async function setSystemThreshold(newVal) {
        newVal = Math.max(0.50, Math.min(0.98, parseFloat(newVal)));
        systemInfo.conservative_threshold = newVal;
        updateSensitivityUI(newVal);

        try {
            await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ conservative_threshold: newVal }),
            });
            addLogEntry(`Sensitivity threshold updated to ${Math.round(newVal * 100)}%`, "info");
        } catch (e) {
            console.error("Failed to update threshold:", e);
        }
    }

    function updateNoiseCancellationUI(isActive) {
        systemInfo.enable_noise_cancellation = !!isActive;
        if (elements.badgeNcVal) {
            elements.badgeNcVal.innerText = isActive ? "ACTIVE 🟢" : "OFF ⚪";
            elements.badgeNcVal.style.color = isActive ? "var(--accent-green)" : "var(--text-muted)";
        }
        if (elements.btnToggleNc) {
            elements.btnToggleNc.innerText = isActive ? "ACTIVE [ON]" : "BYPASSED [OFF]";
            elements.btnToggleNc.style.color = isActive ? "var(--accent-green)" : "var(--text-muted)";
        }
    }

    async function toggleNoiseCancellation() {
        const nextVal = !systemInfo.enable_noise_cancellation;
        updateNoiseCancellationUI(nextVal);
        try {
            await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enable_noise_cancellation: nextVal }),
            });
            addLogEntry(`Noise cancellation ${nextVal ? "enabled (Active)" : "bypassed (Off)"}`, "info");
        } catch (e) {
            console.error("Failed to toggle noise cancellation:", e);
        }
    }

    // -------------------------------------------------------------------------
    // Telemetry & Decision Meters Update
    // -------------------------------------------------------------------------
    function updateDecisionUI(res) {
        const status = res.status || "STATUS: SILENT";
        const alertFired = res.alert_fired || false;
        const pSpray = res.p_spray_can || 0.0;
        const pHiss = res.p_aerosol_spray_hiss || 0.0;
        const pHardNeg = res.p_hard_negative || 0.0;
        const pImpact = res.p_loud_impact || 0.0;
        const pEnv = res.p_continuous_environmental || 0.0;
        const pAmbient = res.p_ambient || 0.0;
        const entropy = res.entropy || 0.0;
        const rms = res.rms || 0.0;
        const cadence = res.cadence_hz || 0.0;
        const burstCount = res.burst_count || 0;
        const valleyDb = res.valley_contrast_db || 0.0;
        const intervalCv = res.interval_cv || 0.0;
        const metalRatio = res.canister_spectral_ratio || 0.0;
        const consecutive = res.consecutive_windows || 0;
        const required = res.required_windows || systemInfo.required_consecutive_windows;
        const explanation = res.explanation || "";

        // 0. Noise Cancellation Telemetry
        const snr = res.snr_db !== undefined ? res.snr_db : 0.0;
        const nr = res.noise_reduction_db !== undefined ? res.noise_reduction_db : 0.0;
        const noiseFloor = res.noise_floor_rms !== undefined ? res.noise_floor_rms : 0.0;
        const ncActive = res.noise_cancellation_active !== undefined ? res.noise_cancellation_active : systemInfo.enable_noise_cancellation;

        if (elements.meterSnrVal) {
            elements.meterSnrVal.innerText = `${snr.toFixed(1)} dB`;
            elements.meterSnrFill.style.width = `${Math.min(100, Math.max(0, (snr / 25.0) * 100))}%`;
        }
        if (elements.meterNrVal) {
            elements.meterNrVal.innerText = `${nr.toFixed(1)} dB`;
            elements.meterNrFill.style.width = `${Math.min(100, Math.max(0, (nr / 20.0) * 100))}%`;
        }
        if (elements.meterNoiseFloorVal) {
            elements.meterNoiseFloorVal.innerText = noiseFloor.toFixed(4);
        }
        updateNoiseCancellationUI(ncActive);

        // 1. Stage 1 Meters
        elements.meterRmsVal.innerText = rms.toFixed(4);
        elements.meterRmsFill.style.width = `${Math.min(100, (rms / 0.1) * 100)}%`;

        elements.meterCadenceVal.innerText = `${cadence.toFixed(1)} Hz (${burstCount} bursts)`;
        elements.meterCadenceFill.style.width = `${Math.min(100, (cadence / 15.0) * 100)}%`;

        if (elements.meterValleyVal) {
            elements.meterValleyVal.innerText = `${valleyDb.toFixed(1)} dB`;
            elements.meterValleyFill.style.width = `${Math.min(100, (valleyDb / 20.0) * 100)}%`;
        }

        if (elements.meterCvVal) {
            elements.meterCvVal.innerText = intervalCv.toFixed(2);
            elements.meterCvFill.style.width = `${Math.min(100, (intervalCv / 1.0) * 100)}%`;
        }

        if (elements.meterMetalVal) {
            elements.meterMetalVal.innerText = `${metalRatio.toFixed(2)}x`;
            elements.meterMetalFill.style.width = `${Math.min(100, (metalRatio / 5.0) * 100)}%`;
        }

        if (res.is_clipped) {
            elements.stage1VerdictText.innerText = "DIGITAL CLIPPING / UNUSABLE";
            elements.stage1VerdictText.style.color = "var(--alert-red)";
        } else if (res.is_stage1_candidate) {
            elements.stage1VerdictText.innerText = `CANDIDATE SHAKING (${cadence.toFixed(1)} Hz)`;
            elements.stage1VerdictText.style.color = "var(--accent-cyan)";
        } else if (rms < 0.001) {
            elements.stage1VerdictText.innerText = "SILENT / LOW ENERGY";
            elements.stage1VerdictText.style.color = "var(--text-muted)";
        } else {
            elements.stage1VerdictText.innerText = "STAGE 1 REJECTED (NON-SHAKING)";
            elements.stage1VerdictText.style.color = "var(--text-secondary)";
        }

        // 2. Stage 2 Meters
        elements.meterSprayVal.innerText = `${(pSpray * 100).toFixed(1)}%`;
        elements.meterSprayFill.style.width = `${Math.min(100, pSpray * 100)}%`;

        if (elements.meterHissVal) {
            elements.meterHissVal.innerText = `${(pHiss * 100).toFixed(1)}%`;
            elements.meterHissFill.style.width = `${Math.min(100, pHiss * 100)}%`;
        }

        elements.meterHardnegVal.innerText = `${(pHardNeg * 100).toFixed(1)}%`;
        elements.meterHardnegFill.style.width = `${Math.min(100, pHardNeg * 100)}%`;

        if (elements.meterImpactVal) {
            elements.meterImpactVal.innerText = `${(pImpact * 100).toFixed(1)}%`;
            elements.meterImpactFill.style.width = `${Math.min(100, pImpact * 100)}%`;
        }

        if (elements.meterEnvVal) {
            elements.meterEnvVal.innerText = `${(pEnv * 100).toFixed(1)}%`;
            elements.meterEnvFill.style.width = `${Math.min(100, pEnv * 100)}%`;
        }

        elements.meterAmbientVal.innerText = `${(pAmbient * 100).toFixed(1)}%`;
        elements.meterAmbientFill.style.width = `${Math.min(100, pAmbient * 100)}%`;

        elements.meterEntropyVal.innerText = entropy.toFixed(2);
        elements.meterEntropyFill.style.width = `${Math.min(100, entropy * 100)}%`;

        // 3. Persistence Indicator Dots
        elements.persistenceSubText.innerText = `${consecutive} / ${required} Required (${(required * 0.25).toFixed(2)}s)`;
        renderPersistenceDots(consecutive, required);

        // 4. Main Status Banner styling & text
        elements.mainStatusBanner.className = "alert-banner";

        if (alertFired) {
            micAlertCount++;
            if (elements.micAlertCount) elements.micAlertCount.innerText = micAlertCount;

            elements.mainStatusBanner.classList.add("status-alert");
            elements.bannerStatusEmoji.innerText = "🚨";
            elements.bannerHeadline.innerText = "ALERT: SPRAY CAN SHAKING CONFIRMED";
            elements.bannerDescription.innerText = explanation;

            // Trigger browser audible notification beep
            playAlertBeep();
            addLogEntry(`🚨 CONFIRMED ALERT: Spray Can Shaking! Confidence=${(pSpray * 100).toFixed(1)}%`, "alert");
        } else if (res.in_cooldown) {
            elements.mainStatusBanner.classList.add("status-cooldown");
            elements.bannerStatusEmoji.innerText = "⏳";
            elements.bannerHeadline.innerText = "IN COOLDOWN";
            elements.bannerDescription.innerText = `Alert suppressed during cooldown window (${res.cooldown_remaining_sec}s remaining).`;
        } else if (res.is_uncertain) {
            elements.mainStatusBanner.classList.add("status-uncertain");
            elements.bannerStatusEmoji.innerText = "❓";
            elements.bannerHeadline.innerText = "UNCERTAIN / AMBIGUOUS SOUND";
            elements.bannerDescription.innerText = explanation;
            addLogEntry(`Ambiguous sound rejected: Entropy=${entropy.toFixed(2)}`, "info");
        } else if (status.includes("AEROSOL_SPRAY_HISS_REJECTED")) {
            elements.mainStatusBanner.classList.add("status-hardneg");
            elements.bannerStatusEmoji.innerText = "💨";
            elements.bannerHeadline.innerText = "AEROSOL SPRAY HISS (NO ALERT)";
            elements.bannerDescription.innerText = explanation;
            addLogEntry(`Aerosol spray hiss rejected: P(hiss)=${(pHiss * 100).toFixed(1)}%. Spraying is NOT shaking.`, "hardneg");
        } else if (status.includes("CONTINUOUS_ENVIRONMENTAL_REJECTED")) {
            elements.mainStatusBanner.classList.add("status-hardneg");
            elements.bannerStatusEmoji.innerText = "🌧️";
            elements.bannerHeadline.innerText = "ENVIRONMENTAL SOUND (RAIN / WATER DROPS / TRAIN)";
            elements.bannerDescription.innerText = explanation;
            addLogEntry(`Environmental sound rejected: P(env)=${(pEnv * 100).toFixed(1)}%. Rain / water is NOT a spray can.`, "hardneg");
        } else if (status.includes("LOUD_IMPACT_REJECTED")) {
            elements.mainStatusBanner.classList.add("status-hardneg");
            elements.bannerStatusEmoji.innerText = "💥";
            elements.bannerHeadline.innerText = "LOUD SUDDEN IMPACT (IGNORED)";
            elements.bannerDescription.innerText = explanation;
            addLogEntry(`Sudden loud impact rejected: P(impact)=${(pImpact * 100).toFixed(1)}%. Loudness alone cannot alert.`, "hardneg");
        } else if (status.includes("HARD_NEGATIVE_RATTLE")) {
            elements.mainStatusBanner.classList.add("status-hardneg");
            elements.bannerStatusEmoji.innerText = "🛡️";
            elements.bannerHeadline.innerText = "HARD NEGATIVE RATTLE (IGNORED)";
            elements.bannerDescription.innerText = explanation;
            addLogEntry(`Hard negative rattle rejected: P(hard_neg)=${(pHardNeg * 100).toFixed(1)}%`, "hardneg");
        } else if (status.includes("AMBIENT_AND_SPEECH") || status.includes("AMBIENT_OR_SPEECH")) {
            elements.mainStatusBanner.classList.add("status-silent");
            elements.bannerStatusEmoji.innerText = "🗣️";
            elements.bannerHeadline.innerText = "AMBIENT SOUND / SPEECH (IGNORED)";
            elements.bannerDescription.innerText = explanation;
        } else if (res.is_stage1_candidate) {
            elements.mainStatusBanner.classList.add("status-candidate");
            elements.bannerStatusEmoji.innerText = "⚡";
            elements.bannerHeadline.innerText = "STAGE 1 CANDIDATE DETECTED";
            elements.bannerDescription.innerText = explanation;
            addLogEntry(`Stage 1 Candidate (${cadence.toFixed(1)} Hz, Persistence: ${consecutive}/${required})`, "candidate");
        } else if (status.includes("SILENT") || rms < 0.001) {
            elements.mainStatusBanner.classList.add("status-silent");
            elements.bannerStatusEmoji.innerText = "🔇";
            elements.bannerHeadline.innerText = "STATUS: SILENT";
            elements.bannerDescription.innerText = "Quiet background noise or silence. Awaiting acoustic input.";
        } else {
            elements.mainStatusBanner.classList.add("status-silent");
            elements.bannerStatusEmoji.innerText = "🔈";
            elements.bannerHeadline.innerText = "NON-SHAKING SOUND";
            elements.bannerDescription.innerText = explanation;
        }
    }

    // Audible alert notification tone using Web Audio synth
    function playAlertBeep() {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(1760, ctx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.25);
        } catch (e) {
            // Audio context policy
        }
    }

    // -------------------------------------------------------------------------
    // Canvas Oscilloscope Visualizer
    // -------------------------------------------------------------------------
    function drawWaveform() {
        if (!elements.waveformCanvas || !analyserNode) return;
        const canvas = elements.waveformCanvas;
        const ctx = canvas.getContext("2d");
        const width = canvas.width;
        const height = canvas.height;

        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Float32Array(bufferLength);

        function render() {
            if (!isListening) {
                // Clear when stopped
                ctx.fillStyle = "#0c1220";
                ctx.fillRect(0, 0, width, height);
                ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
                ctx.beginPath();
                ctx.moveTo(0, height / 2);
                ctx.lineTo(width, height / 2);
                ctx.stroke();
                return;
            }

            requestAnimationFrame(render);
            analyserNode.getFloatTimeDomainData(dataArray);

            ctx.fillStyle = "#0c1220";
            ctx.fillRect(0, 0, width, height);

            // Draw center grid line
            ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();

            // Calculate instantaneous RMS for canvas tag
            let sumSq = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sumSq += dataArray[i] * dataArray[i];
            }
            const curRms = Math.sqrt(sumSq / dataArray.length);
            if (elements.canvasRmsTag) {
                elements.canvasRmsTag.innerText = `RMS: ${curRms.toFixed(3)}`;
            }

            // Draw animated oscilloscope trace
            ctx.lineWidth = 2;
            ctx.strokeStyle = curRms > 0.01 ? "#06b6d4" : "rgba(6, 182, 212, 0.4)";
            ctx.beginPath();

            const sliceWidth = width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i];
                const y = (v * (height / 2)) + (height / 2);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            ctx.lineTo(width, height / 2);
            ctx.stroke();
        }

        render();
    }

    // -------------------------------------------------------------------------
    // Real-Time Microphone Stream (Web Audio API)
    // -------------------------------------------------------------------------
    function floatTo16BitPCMBase64(floatArray) {
        const buffer = new ArrayBuffer(floatArray.length * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < floatArray.length; i++) {
            const s = Math.max(-1, Math.min(1, floatArray[i]));
            view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        let binary = "";
        const bytes = new Uint8Array(buffer);
        const len = bytes.byteLength;
        const chunkSize = 8192;
        for (let i = 0; i < len; i += chunkSize) {
            binary += String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i + chunkSize, len)));
        }
        return btoa(binary);
    }

    function resampleLinear(input, inRate, outRate, targetLength) {
        if (inRate === outRate && input.length === targetLength) {
            return input;
        }
        const output = new Float32Array(targetLength);
        const ratio = (input.length - 1) / (targetLength - 1);
        for (let i = 0; i < targetLength; i++) {
            const pos = i * ratio;
            const index = Math.floor(pos);
            const frac = pos - index;
            if (index >= input.length - 1) {
                output[i] = input[input.length - 1];
            } else {
                output[i] = input[index] * (1 - frac) + input[index + 1] * frac;
            }
        }
        return output;
    }

    async function startListening() {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === "suspended") {
                await audioContext.resume();
            }
            nativeSampleRate = audioContext.sampleRate || 22050;
            nativeWindowSamples = Math.round(nativeSampleRate * 1.0);
            nativeBuffer = new Float32Array(nativeWindowSamples);
            nativeBufferWriteIdx = 0;

            if (elements.canvasRateTag) {
                elements.canvasRateTag.innerText = `${nativeSampleRate} Hz → 22,050 Hz`;
            }
            if (elements.micDeviceName) {
                elements.micDeviceName.innerText = `Active Mic (${nativeSampleRate} Hz)`;
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: false,
                },
            });
            micStream = stream;

            const source = audioContext.createMediaStreamSource(stream);
            analyserNode = audioContext.createAnalyser();
            analyserNode.fftSize = 1024;
            source.connect(analyserNode);

            // Use zero-gain node to prevent speaker feedback while keeping scriptProcessor alive
            const muteGainNode = audioContext.createGain();
            muteGainNode.gain.value = 0;

            // Process audio frames with ScriptProcessor (bufferSize = 4096)
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            source.connect(scriptProcessor);
            scriptProcessor.connect(muteGainNode);
            muteGainNode.connect(audioContext.destination);

            scriptProcessor.onaudioprocess = (event) => {
                const inputData = event.inputBuffer.getChannelData(0);
                // Copy into native rate circular rolling buffer
                for (let i = 0; i < inputData.length; i++) {
                    nativeBuffer[nativeBufferWriteIdx] = inputData[i];
                    nativeBufferWriteIdx = (nativeBufferWriteIdx + 1) % nativeWindowSamples;
                }
            };

            isListening = true;
            elements.micBtnText.innerText = "Stop Listening";
            elements.micBtnIcon.innerText = "⏹";
            elements.btnToggleMic.classList.replace("btn-primary", "btn-secondary");

            addLogEntry(`Microphone started (${nativeSampleRate} Hz). Auto-resampling to 22,050 Hz every 250ms.`, "info");
            drawWaveform();

            let isRequestInFlight = false;

            // Send rolling 1.0s window every 250ms
            pollInterval = setInterval(async () => {
                if (!isListening || isRequestInFlight) return;

                // Assemble chronologically ordered 1.0s window from circular buffer
                const orderedWindow = new Float32Array(nativeWindowSamples);
                for (let i = 0; i < nativeWindowSamples; i++) {
                    orderedWindow[i] = nativeBuffer[(nativeBufferWriteIdx + i) % nativeWindowSamples];
                }

                // Resample to exactly 22,050 Hz (WINDOW_SAMPLES)
                const resampledWindow = resampleLinear(orderedWindow, nativeSampleRate, TARGET_SAMPLE_RATE, WINDOW_SAMPLES);
                const base64Pcm = floatTo16BitPCMBase64(resampledWindow);

                isRequestInFlight = true;
                try {
                    const resp = await fetch("/api/predict_window", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            audio_base64: base64Pcm,
                            sample_rate: TARGET_SAMPLE_RATE,
                            native_sample_rate: nativeSampleRate,
                        }),
                    });
                    const resData = await resp.json();
                    updateDecisionUI(resData);
                } catch (e) {
                    console.error("Window prediction request failed:", e);
                } finally {
                    isRequestInFlight = false;
                }
            }, 350);

        } catch (err) {
            console.error("Microphone access failed:", err);
            alert(`Unable to access microphone: ${err.message}. Please check browser microphone permissions.`);
        }
    }

    function stopListening() {
        isListening = false;
        if (pollInterval) clearInterval(pollInterval);
        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor = null;
        }
        if (micStream) {
            micStream.getTracks().forEach((track) => track.stop());
            micStream = null;
        }
        if (audioContext && audioContext.state !== "closed") {
            audioContext.close();
        }

        elements.micBtnText.innerText = "Start Listening";
        elements.micBtnIcon.innerText = "▶";
        elements.btnToggleMic.classList.replace("btn-secondary", "btn-primary");
        addLogEntry("Microphone stream stopped.", "info");
    }

    if (elements.btnToggleMic) {
        elements.btnToggleMic.addEventListener("click", () => {
            if (isListening) {
                stopListening();
            } else {
                startListening();
            }
        });
    }

    // -------------------------------------------------------------------------
    // Hard-Negative Sandbox Test Suite
    // -------------------------------------------------------------------------
    async function loadDemoSamples() {
        if (!elements.sandboxGrid) return;
        try {
            const res = await fetch("/api/demo_samples");
            const data = await res.json();
            const samples = data.samples || [];

            elements.sandboxGrid.innerHTML = "";
            samples.forEach((sample) => {
                const card = document.createElement("div");
                card.className = "sandbox-item";
                card.id = `sample-card-${sample.id.replace(/\.[^/.]+$/, "")}`;

                let badgeClass = "badge-ambient";
                if (sample.type === "positive") badgeClass = "badge-target";
                if (sample.type === "hard_negative") badgeClass = "badge-hard-neg";
                if (sample.type === "aerosol_spray_hiss") badgeClass = "badge-hard-neg";

                card.innerHTML = `
                    <div class="sandbox-item-header">
                        <span class="sample-badge ${badgeClass}">${sample.badge}</span>
                    </div>
                    <div class="sample-name">${sample.label}</div>
                    <button class="btn-test-sample" data-id="${sample.id}">
                        ▶ Evaluate Sound
                    </button>
                `;

                const btn = card.querySelector(".btn-test-sample");
                btn.addEventListener("click", () => evaluateDemoSample(sample));

                elements.sandboxGrid.appendChild(card);
            });
        } catch (err) {
            console.error("Failed to load demo samples:", err);
        }
    }

    async function evaluateDemoSample(sample) {
        // Highlight active sample
        document.querySelectorAll(".sandbox-item").forEach((el) => el.classList.remove("active-sample"));
        const activeCard = document.getElementById(`sample-card-${sample.id.replace(/\.[^/.]+$/, "")}`);
        if (activeCard) activeCard.classList.add("active-sample");

        // Show player
        elements.sandboxPlayerCard.style.display = "block";
        elements.playerSampleTitle.innerText = `Evaluating: ${sample.label}`;
        elements.sandboxAudioPlayer.src = `/api/demo_sample/${sample.id}/audio`;
        elements.sandboxAudioPlayer.play().catch(() => {});

        elements.sandboxEvalOutput.innerHTML = `<div style="color: var(--accent-cyan); padding: 10px;">Evaluating acoustic pattern across sliding windows...</div>`;

        try {
            const res = await fetch(`/api/demo_sample/${sample.id}/evaluate`, { method: "POST" });
            const data = await res.json();

            // Render evaluation summary in sandbox player
            const isAlert = data.confirmed_alerts_count > 0;
            const alertClass = isAlert ? "highlight-alert" : "highlight-spray";

            elements.sandboxEvalOutput.innerHTML = `
                <div style="margin-top: 12px; padding: 12px; background: rgba(0,0,0,0.3); border-radius: var(--radius-sm);">
                    <div style="font-weight: 700; font-size: 1rem; color: ${isAlert ? 'var(--accent-red)' : 'var(--text-primary)'}; margin-bottom: 6px;">
                        ${data.final_verdict}
                    </div>
                    <div style="font-size: 0.825rem; color: var(--text-secondary); display: flex; gap: 16px; flex-wrap: wrap;">
                        <span>Duration: <strong>${data.total_duration_sec}s</strong></span>
                        <span>Windows: <strong>${data.total_windows_evaluated}</strong></span>
                        <span>Stage 1 Candidates: <strong>${data.stage1_candidates_count}</strong></span>
                        <span>Max Spray Confidence: <strong class="${alertClass}">${(data.max_spray_confidence * 100).toFixed(1)}%</strong></span>
                        <span>Max Hard-Neg Confidence: <strong>${(data.max_hard_neg_confidence * 100).toFixed(1)}%</strong></span>
                    </div>
                </div>
            `;

            // If alerts fired or windows evaluated, update main decision meters with peak window
            if (data.timeline && data.timeline.length > 0) {
                // Find highest spray window or first alert window
                let peakWindow = data.timeline.find((w) => w.alert_fired) || data.timeline[0];
                for (const w of data.timeline) {
                    if (w.p_spray_can > peakWindow.p_spray_can) peakWindow = w;
                }
                updateDecisionUI(peakWindow);
            }

            addLogEntry(
                `Sandbox [${sample.label}]: ${data.final_verdict} (Max Spray=${(data.max_spray_confidence * 100).toFixed(1)}%)`,
                isAlert ? "alert" : "info"
            );
        } catch (err) {
            console.error("Evaluation failed:", err);
            elements.sandboxEvalOutput.innerText = `Evaluation failed: ${err.message}`;
        }
    }

    // -------------------------------------------------------------------------
    // Audio File Analyzer (Upload & Diagnostics)
    // -------------------------------------------------------------------------
    if (elements.uploadDropzone && elements.fileInput) {
        elements.btnBrowseFile.addEventListener("click", () => elements.fileInput.click());
        elements.uploadDropzone.addEventListener("click", (e) => {
            if (e.target !== elements.btnBrowseFile) elements.fileInput.click();
        });

        elements.uploadDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.add("drag-over");
        });

        elements.uploadDropzone.addEventListener("dragleave", () => {
            elements.uploadDropzone.classList.remove("drag-over");
        });

        elements.uploadDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.remove("drag-over");
            if (e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        elements.fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    async function handleFileUpload(file) {
        addLogEntry(`Uploading & evaluating audio file: ${file.name}...`, "info");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/evaluate_file", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();
            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }

            renderFileEvaluationResults(data);
        } catch (err) {
            console.error("File upload evaluation error:", err);
            alert(`File analysis failed: ${err.message}`);
        }
    }

    function renderFileEvaluationResults(data) {
        elements.fileEvalResult.style.display = "block";
        elements.resDuration.innerText = `${data.total_duration_sec}s`;
        elements.resWindows.innerText = data.total_windows_evaluated;
        elements.resCandidates.innerText = data.stage1_candidates_count;
        elements.resAlerts.innerText = data.confirmed_alerts_count;

        // Render spectrogram
        if (data.spectrogram_image_base64) {
            elements.spectrogramImage.src = `data:image/png;base64,${data.spectrogram_image_base64}`;
            elements.spectrogramImage.style.display = "block";
        }

        // Render timeline
        elements.timelineList.innerHTML = "";
        (data.timeline || []).forEach((item) => {
            const row = document.createElement("div");
            row.className = `timeline-row ${item.alert_fired ? "is-alert" : ""}`;
            const timeTag = `[${item.window_start_sec}s - ${item.window_end_sec}s]`;
            const sprayPct = (item.p_spray_can * 100).toFixed(1);
            const hardNegPct = (item.p_hard_negative * 100).toFixed(1);

            row.innerHTML = `
                <span>${timeTag} ${item.status}</span>
                <span>Spray: <strong>${sprayPct}%</strong> | HardNeg: ${hardNegPct}% | Cadence: ${item.cadence_hz}Hz</span>
            `;
            elements.timelineList.appendChild(row);
        });

        // Update main meters with first alert window or highest spray window
        if (data.timeline && data.timeline.length > 0) {
            let peak = data.timeline.find((w) => w.alert_fired) || data.timeline[0];
            for (const w of data.timeline) {
                if (w.p_spray_can > peak.p_spray_can) peak = w;
            }
            updateDecisionUI(peak);
        }

        addLogEntry(
            `File Evaluation Complete: ${data.final_verdict} (Confirmed Alerts=${data.confirmed_alerts_count})`,
            data.confirmed_alerts_count > 0 ? "alert" : "info"
        );
    }

    // -------------------------------------------------------------------------
    // Threshold Calibration & False Alert Sweep
    // -------------------------------------------------------------------------
    async function loadCalibrationReport() {
        if (!elements.calibrationTableBody) return;
        try {
            const res = await fetch("/api/calibration_report");
            const data = await res.json();
            if (data.error) return;

            const sweep = data.validation_tuning?.validation_sweep || data.calibration?.calibration_sweep || [];
            const selectedThreshold = data.conservative_threshold || 0.75;

            if (elements.calibSelectedThreshold) {
                elements.calibSelectedThreshold.innerText = `${Math.round(selectedThreshold * 100)}%`;
            }

            elements.calibrationTableBody.innerHTML = "";
            sweep.forEach((step) => {
                const tr = document.createElement("tr");
                const isSelected = Math.abs(step.threshold - selectedThreshold) < 0.01;
                if (isSelected) tr.classList.add("selected-row");

                const fpVal = step.fp !== undefined ? step.fp : (step.fp_hard_neg || 0);
                const recallVal = step.recall !== undefined ? step.recall : (step.tpr_spray || 0);
                const precVal = step.precision !== undefined ? step.precision : 1.0;
                const f1Val = step.f1 !== undefined ? step.f1 : 0.0;
                const hardNegZero = fpVal === 0;
                const verdictBadge = hardNegZero
                    ? `<span style="color: var(--accent-green);">✓ Zero False Alerts</span>`
                    : `<span style="color: var(--accent-red);">${fpVal} False Alert(s)</span>`;

                tr.innerHTML = `
                    <td>${step.threshold.toFixed(2)}</td>
                    <td>${(recallVal * 100).toFixed(1)}%</td>
                    <td><strong>${fpVal}</strong></td>
                    <td>0.0%</td>
                    <td>0.0%</td>
                    <td>${(precVal * 100).toFixed(1)}%</td>
                    <td>${f1Val.toFixed(4)}</td>
                    <td>${isSelected ? '★ <strong>SELECTED</strong>' : verdictBadge}</td>
                `;

                elements.calibrationTableBody.appendChild(tr);
            });
        } catch (err) {
            console.error("Failed to load calibration report:", err);
        }
    }

    if (elements.btnRefreshCalibration) {
        elements.btnRefreshCalibration.addEventListener("click", loadCalibrationReport);
    }

    // Wire up sensitivity presets & slider
    if (elements.presetSensitive) {
        elements.presetSensitive.addEventListener("click", () => setSystemThreshold(0.75));
    }
    if (elements.presetBalanced) {
        elements.presetBalanced.addEventListener("click", () => setSystemThreshold(0.85));
    }
    if (elements.presetConservative) {
        elements.presetConservative.addEventListener("click", () => setSystemThreshold(0.95));
    }
    if (elements.thresholdSlider) {
        elements.thresholdSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value) / 100.0;
            updateSensitivityUI(val);
        });
        elements.thresholdSlider.addEventListener("change", (e) => {
            const val = parseFloat(e.target.value) / 100.0;
            setSystemThreshold(val);
        });
    }

    // Wire up noise cancellation toggle
    if (elements.btnToggleNc) {
        elements.btnToggleNc.addEventListener("click", toggleNoiseCancellation);
    }
    if (elements.ncBadge) {
        elements.ncBadge.addEventListener("click", toggleNoiseCancellation);
    }

    // Initialize system state
    initSystemStatus();
});
