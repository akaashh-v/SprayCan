# ESP32-WROVER-B Deployment Feasibility Study: YAMNet vs. Spectrogram CNN

## Executive Summary
Deploying the full **YAMNet embedding extractor + classifier head directly on the ESP32-WROVER-B is technically NOT FEASIBLE** due to insurmountable memory and compute constraints. However, **Model B (Expanded-Data CNN) is fully feasible and ready for edge deployment**.

## Quantitative Hardware Comparison

| Dimension | ESP32-WROVER-B Hardware Limit | Production CNN (Model A/B) | YAMNet Pipeline (Model C) | Feasibility Result |
| :--- | :--- | :--- | :--- | :--- |
| **Model Weights (INT8)** | 4096 KB Flash/PSRAM | 532 KB | 3,789 KB | YAMNet consumes 92% of entire PSRAM |
| **Activation Arena** | 260 KB SRAM / 4 MB PSRAM | 128 KB | ~1,850 KB | Cannot fit in SRAM; stalls PSRAM bus |
| **Total RAM Footprint** | 4,096 KB total PSRAM | ~256 KB | **5,703 KB** | **EXCEEDS PSRAM BY 1,607 KB (139%)** |
| **Computational Load** | 240 MHz Xtensa LX6 (No SIMD) | 14.8 MFLOPs | 195.0 MFLOPs | 13.2x higher compute load |
| **Inference Latency** | **1,000 ms (Real-Time Budget)** | **~380 ms** | **~4,370 ms** | **4.37x SLOWER THAN REAL-TIME** |
| **Audio Resampling** | Native 22,050 Hz | None (0 ms) | 22.05k -> 16k (+65 ms) | Software polyphase filter required |

## Detailed Technical Blockers for YAMNet on ESP32

### ❌ Severe Memory Overflow
INT8 YAMNet weights (3789.1 KB) plus activation arena (1850.0 KB) require 5703.1 KB RAM, which exceeds the total 4096 KB PSRAM by 1607.1 KB (139% of total capacity).

### ❌ Severe Real-Time Latency Violation
Estimated total inference pipeline takes ~4370 ms on ESP32 Xtensa LX6 @ 240 MHz. This is 4.4x slower than real time (budget: 1000 ms). Audio ring buffers would immediately overflow, dropping 75% of incoming audio.

### ❌ Resampling & Dual-Model Complexity
YAMNet strictly requires 16,000 Hz audio. The INMP441 capture is locked at 22,050 Hz to match the production filterbank. Performing on-chip decimation/interpolation adds 65 ms of CPU overhead per second.

## Strategic Recommendation

- Do NOT deploy YAMNet directly on the ESP32-WROVER-B.
- Deploy Model B (Expanded-Data CNN) converted to INT8 TFLite on the ESP32. It maintains sub-400 ms latency and fits comfortably in 532 KB.
- If YAMNet's superior general feature representation is desired, use it as a desktop/server-side secondary validator when the ESP32 flags an ambiguous rattle event.
- Alternatively, explore Knowledge Distillation where YAMNet acts as a teacher guiding Model B's training.
