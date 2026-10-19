#ifndef AUDIO_DSP_H_
#define AUDIO_DSP_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_DSP_SAMPLE_RATE 22050
#define AUDIO_DSP_AUDIO_SAMPLES 22050
#define AUDIO_DSP_N_FFT 1024
#define AUDIO_DSP_HOP_LENGTH 512
#define AUDIO_DSP_N_MELS 64
#define AUDIO_DSP_N_FRAMES 44
#define AUDIO_DSP_SPECTROGRAM_SIZE (AUDIO_DSP_N_MELS * AUDIO_DSP_N_FRAMES) // 2816

/**
 * Initializes DSP tables, FFT twiddle factors, and Hann window.
 * Allocates internal scratch buffers in PSRAM/SRAM.
 * @return true on success, false on allocation/initialization failure.
 */
bool audio_dsp_init(void);

/**
 * Converts 16-bit PCM Mono audio to normalized float32 [-1.0, 1.0].
 * @param src16 Input 16-bit signed PCM buffer.
 * @param dstFloat Output float32 buffer (must hold at least num_samples floats).
 * @param num_samples Number of samples to convert.
 */
void audio_dsp_pcm16_to_float(const int16_t* src16, float* dstFloat, size_t num_samples);

/**
 * Computes the exact 64x44 Log-Mel Spectrogram matching Python librosa / audio_utils.py.
 * Output tensor is row-major of shape (64, 44), exactly matching CNN input (Batch, 64, 44, 1).
 *
 * Pipeline:
 * 1. Constant zero-padding by 512 samples at start & end.
 * 2. 44 frames x 1024-point FFT with periodic Hann window.
 * 3. Power spectrum |X[k]|^2.
 * 4. 64-bin Mel filterbank dot product.
 * 5. dB conversion: 10 * log10(max(val, 1e-10)) - 10 * log10(max(global_max, 1e-10)), clamped to [-80, 0] dB.
 * 6. Global zero-mean, unit-variance normalization: (dB - mean) / (std + 1e-6).
 *
 * @param pcm_float Float32 audio samples (length AUDIO_DSP_AUDIO_SAMPLES).
 * @param out_spectrogram Output buffer of size AUDIO_DSP_SPECTROGRAM_SIZE (2816 floats).
 * @return true on success.
 */
bool audio_dsp_compute_log_mel(const float* pcm_float, float* out_spectrogram);

#ifdef __cplusplus
}
#endif

#endif // AUDIO_DSP_H_
