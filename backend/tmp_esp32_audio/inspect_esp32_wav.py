
from pathlib import Path
import wave
import numpy as np

# Folder where Flask saves ESP32 recordings
AUDIO_DIR = Path("tmp_esp32_audio")

files = sorted(
    AUDIO_DIR.glob("esp32_*.wav"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

if not files:
    raise SystemExit(
        "No WAV files found. Run the Flask receiver first."
    )

wav_path = files[0]
print(f"\nInspecting: {wav_path}\n")

try:
    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        duration = frames / sample_rate
        audio_bytes = wav.readframes(frames)

    print("----- WAV DETAILS -----")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Bits per sample: {sample_width * 8}")
    print(f"Frames: {frames}")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Audio data bytes: {len(audio_bytes)}")
    print(f"File size: {wav_path.stat().st_size} bytes")

    # Convert supported PCM sample formats to numbers
    if sample_width == 1:
        samples = np.frombuffer(
            audio_bytes, dtype=np.uint8
        ).astype(np.float32) - 128
    elif sample_width == 2:
        samples = np.frombuffer(
            audio_bytes, dtype="<i2"
        ).astype(np.float32)
    elif sample_width == 4:
        samples = np.frombuffer(
            audio_bytes, dtype="<i4"
        ).astype(np.float32)
    elif sample_width == 3:
        raw = np.frombuffer(audio_bytes, dtype=np.uint8)
        raw = raw[:len(raw) // 3 * 3].reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        samples = ((values ^ 0x800000) - 0x800000).astype(
            np.float32
        )
    else:
        raise ValueError("Unsupported PCM sample width")

    if samples.size == 0:
        raise ValueError("Audio file contains no samples")

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples ** 2)))

    print("\n----- AUDIO LEVELS -----")
    print(f"Peak amplitude: {peak:.1f}")
    print(f"RMS amplitude: {rms:.1f}")

    if peak == 0:
        print("WARNING: Audio is completely silent.")
    elif rms < 10:
        print("WARNING: Audio level is very low; listen to it.")
    else:
        print("Audio contains non-zero sample values.")

    print("\nInspection complete.")

except Exception as exc:
    print(f"Could not inspect this WAV: {exc}")
    print("The WAV may use a format unsupported by Python's wave module.")
