"""Real-Time Physical Microphone Aerosol Spray Can Sound Detector.

Streams live audio from your microphone and runs the two-stage decision pipeline:
1. Stage 1: Detects kinematic bursts & cadence (2.0 - 8.5 Hz) and broadband rattling energy (2 - 9 kHz).
2. Stage 2: Confirms authentic spray-can resonance vs hard negatives (coins, keys, pill bottles, etc.)
   using the deep CNN over the full 2D Log-Mel spectrogram.
3. Multi-Window Temporal Persistence: Alerts ONLY when matched across >= 2 consecutive windows.
4. Continuous Real-Time Evaluation: Zero cooldown; re-arms immediately via event-state tracking.
"""

import argparse
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from config import (
    AUDIO_SAMPLES,
    CONSERVATIVE_THRESHOLD,
    HOP_SAMPLES,
    SAMPLE_RATE,
    STREAM_HOP_DURATION,
)
from detector_engine import SprayCanDetectorEngine

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[90m"


def print_banner(engine: SprayCanDetectorEngine, device_info: dict) -> None:
    """Print terminal header banner."""
    print("=" * 80)
    print(f"{BOLD}{CYAN}   PRECISION AEROSOL SPRAY CAN SHAKING SOUND DETECTOR (REAL-TIME){RESET}")
    print("=" * 80)
    print(f" • Input Device        : {device_info.get('name', 'Default')} (Index {device_info.get('index', 'default')})")
    print(f" • Sampling Rate       : {SAMPLE_RATE} Hz | Window: 1.0s | Hop: {STREAM_HOP_DURATION*1000:.0f} ms")
    print(f" • Stage 1 Pre-Filter  : Kinematic cadence (2.0-8.5 Hz) + Broadband rattle (2-9 kHz)")
    print(f" • Stage 2 Neural CNN  : Full 2D Log-Mel Spectrogram (64x44) with Hard-Negative Rejection")
    print(f" • Conservative Cutoff : {engine.conservative_threshold*100:.0f}% confidence")
    print(f" • Multi-Window Gating : Required {engine.required_consecutive_windows} consecutive windows (>= 0.50s)")
    print(f" • Alert Cooldown      : NONE (0.0s - Continuous Real-Time)")
    print("=" * 80)
    print(f"{DIM}Press Ctrl+C to stop real-time monitoring.{RESET}\n")


def format_bar(val: float, max_val: float = 1.0, length: int = 20, fill_char: str = "■") -> str:
    """Render a colored ASCII progress bar."""
    ratio = max(0.0, min(1.0, val / max_val))
    filled = int(ratio * length)
    empty = length - filled
    return f"[{fill_char * filled}{' ' * empty}]"


def run_live_detector(device_index: Optional[int] = None) -> None:
    """Run real-time audio listener on microphone."""
    try:
        if device_index is None:
            device_info = sd.query_devices(kind="input")
        else:
            device_info = sd.query_devices(device_index, kind="input")
    except Exception as e:
        print(f"{RED}Error querying audio input device: {e}{RESET}")
        return

    engine = SprayCanDetectorEngine()
    print_banner(engine, device_info)

    buffer_size = AUDIO_SAMPLES
    hop_size = HOP_SAMPLES
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)

    alert_count = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer, alert_count
        if status:
            pass  # Overflow/underflow handled gracefully

        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        # Update rolling buffer
        if len(mono) >= buffer_size:
            audio_buffer = mono[-buffer_size:].astype(np.float32)
        else:
            audio_buffer = np.roll(audio_buffer, -len(mono))
            audio_buffer[-len(mono):] = mono.astype(np.float32)

    # Open microphone stream
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=hop_size,
            device=device_index,
            callback=audio_callback,
        )
    except Exception as e:
        print(f"{RED}Failed to open audio stream on device {device_index}: {e}{RESET}")
        return

    with stream:
        print(f"{GREEN}✓ Microphone stream active. Listening for acoustic signatures...{RESET}\n")
        try:
            while True:
                time.sleep(STREAM_HOP_DURATION)
                res = engine.evaluate_window(audio_buffer)

                status = res["status"]
                alert_fired = res["alert_fired"]
                p_spray = res["p_spray_can"]
                p_hard_neg = res["p_hard_negative"]
                cadence = res["cadence_hz"]
                bursts = res["burst_count"]
                broadband = res["broadband_ratio"]
                persistence = res["consecutive_windows"]
                required = res["required_windows"]

                # Persistence indicator: [● ● ○]
                dots = ""
                for i in range(required):
                    if i < persistence:
                        dots += f"{GREEN}●{RESET} "
                    else:
                        dots += f"{DIM}○{RESET} "

                # Color-coded status line
                if alert_fired:
                    alert_count += 1
                    status_line = f"\n{BOLD}{RED}🚨 >>> [ALERT #{alert_count}] AEROSOL SPRAY CAN SHAKING CONFIRMED! <<< 🚨{RESET}"
                    detail_line = (
                        f"   Confidence: {BOLD}{GREEN}{p_spray*100:5.1f}%{RESET} | "
                        f"Cadence: {cadence:.1f} Hz ({bursts} bursts) | "
                        f"Persistence: {required}/{required} windows | {res['explanation']}\n"
                    )
                    sys.stdout.write(status_line + "\n" + detail_line)
                    sys.stdout.flush()
                else:
                    if status == "UNUSABLE_AUDIO_CLIPPING":
                        badge = f"{RED}[CLIPPING: UNUSABLE]{RESET}"
                    elif status == "STATUS: SILENT":
                        badge = f"{DIM}[SILENT]{RESET}"
                    elif "NO_SHAKING_CADENCE" in status:
                        badge = f"{DIM}[NO CADENCE]{RESET}"
                    elif "STAGE1_CANDIDATE" in status:
                        badge = f"{CYAN}[STAGE 1 CANDIDATE]{RESET}"
                    elif "LOUD_IMPACT" in status:
                        badge = f"{YELLOW}[LOUD IMPACT: IGNORED]{RESET}"
                    elif "CONTINUOUS_ENVIRONMENTAL" in status:
                        badge = f"{MAGENTA}[ENVIRONMENTAL: IGNORED]{RESET}"
                    elif "HARD_NEGATIVE_RATTLE" in status:
                        badge = f"{MAGENTA}[HARD NEGATIVE: IGNORED]{RESET}"
                    elif "UNCERTAIN" in status:
                        badge = f"{YELLOW}[UNCERTAIN SOUND]{RESET}"
                    elif "SUSTAINED_SHAKING" in status:
                        badge = f"{GREEN}[SUSTAINED SHAKING]{RESET}"
                    elif "IN_COOLDOWN" in status:
                        badge = f"{YELLOW}[IN COOLDOWN: {res['cooldown_remaining_sec']}s]{RESET}"
                    else:
                        badge = f"[{status}]"

                    meter = format_bar(p_spray, 1.0, length=12)
                    sys.stdout.write(
                        f"\r{badge:32s} | Spray: {p_spray*100:4.1f}% {meter} | "
                        f"HardNeg: {p_hard_neg*100:4.1f}% | "
                        f"Bursts: {bursts:2d} ({cadence:3.1f} Hz) | "
                        f"Persistence: {dots.strip()}  "
                    )
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Stopped listening. Total alerts detected: {alert_count}{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Spray Can Shaking Sound Detector")
    parser.add_argument("--device", type=int, default=None, help="Input microphone device index (optional)")
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable Audio Input Devices:")
        print(sd.query_devices())
        sys.exit(0)

    run_live_detector(device_index=args.device)
