#include "audio_dsp.h"
#include "mel_filterbank.h"

#include <Arduino.h>
#include <esp_dsp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float* s_hann_window = nullptr;
static float* s_fft_complex = nullptr;
static float* s_power_spectrum = nullptr;
static float* s_mel_matrix = nullptr;
static bool s_dsp_initialized = false;

bool audio_dsp_init(void) {
  if (s_dsp_initialized) {
    return true;
  }

  // Initialize esp-dsp Radix-2 FFT tables for 1024 points
  esp_err_t res = dsps_fft2r_init_fc32(NULL, AUDIO_DSP_N_FFT);
  if (res != ESP_OK) {
    Serial.printf("[DSP] ERROR: dsps_fft2r_init_fc32 failed (code %d)!\n", res);
    return false;
  }

  // Allocate scratch buffers in PSRAM if available, or internal heap
  size_t hann_bytes = AUDIO_DSP_N_FFT * sizeof(float);
  size_t fft_bytes = 2 * AUDIO_DSP_N_FFT * sizeof(float);
  size_t power_bytes = FFT_HALF_BINS * sizeof(float);
  size_t mel_bytes = AUDIO_DSP_SPECTROGRAM_SIZE * sizeof(float);

  s_hann_window = (float*)(psramFound() ? ps_malloc(hann_bytes) : malloc(hann_bytes));
  s_fft_complex = (float*)(psramFound() ? ps_malloc(fft_bytes) : malloc(fft_bytes));
  s_power_spectrum = (float*)(psramFound() ? ps_malloc(power_bytes) : malloc(power_bytes));
  s_mel_matrix = (float*)(psramFound() ? ps_malloc(mel_bytes) : malloc(mel_bytes));

  if (!s_hann_window || !s_fft_complex || !s_power_spectrum || !s_mel_matrix) {
    Serial.println("[DSP] ERROR: Failed to allocate scratch buffers for audio DSP!");
    return false;
  }

  // Generate periodic Hann window: 0.5 * (1 - cos(2 * pi * n / N)) matching scipy.signal.windows.hann(1024, sym=False)
  const float pi2_over_n = (2.0f * (float)M_PI) / (float)AUDIO_DSP_N_FFT;
  for (int i = 0; i < AUDIO_DSP_N_FFT; ++i) {
    s_hann_window[i] = 0.5f * (1.0f - cosf(pi2_over_n * (float)i));
  }

  s_dsp_initialized = true;
  Serial.printf("[DSP] Initialized successfully. Scratch memory: %u bytes\n",
                (unsigned int)(hann_bytes + fft_bytes + power_bytes + mel_bytes));
  return true;
}

void audio_dsp_pcm16_to_float(const int16_t* src16, float* dstFloat, size_t num_samples) {
  if (!src16 || !dstFloat) return;
  const float inv_scale = 1.0f / 32768.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    dstFloat[i] = (float)src16[i] * inv_scale;
  }
}

bool audio_dsp_compute_log_mel(const float* pcm_float, float* out_spectrogram) {
  if (!s_dsp_initialized) {
    if (!audio_dsp_init()) return false;
  }
  if (!pcm_float || !out_spectrogram) return false;

  // Process 44 frames with hop 512, centered constant zero-padding of 512 samples
  for (int f = 0; f < AUDIO_DSP_N_FRAMES; ++f) {
    int start_sample = f * AUDIO_DSP_HOP_LENGTH - 512;

    // Windowing with constant zero-padding bounds checking
    for (int i = 0; i < AUDIO_DSP_N_FFT; ++i) {
      int s_idx = start_sample + i;
      float sample = 0.0f;
      if (s_idx >= 0 && s_idx < AUDIO_DSP_AUDIO_SAMPLES) {
        sample = pcm_float[s_idx];
      }
      s_fft_complex[2 * i] = sample * s_hann_window[i];
      s_fft_complex[2 * i + 1] = 0.0f;
    }

    // In-place Radix-2 Complex FFT + Bit Reversal
    dsps_fft2r_fc32(s_fft_complex, AUDIO_DSP_N_FFT);
    dsps_bit_rev2r_fc32(s_fft_complex, AUDIO_DSP_N_FFT);

    // Compute single-sided power spectrum |X[k]|^2 for 513 frequency bins
    for (int k = 0; k < FFT_HALF_BINS; ++k) {
      float re = s_fft_complex[2 * k];
      float im = s_fft_complex[2 * k + 1];
      s_power_spectrum[k] = re * re + im * im;
    }

    // Mel filterbank dot-product for 64 bins (sparse triangular representation)
    for (int m = 0; m < AUDIO_DSP_N_MELS; ++m) {
      const MelFilterDef& filt = g_mel_filters[m];
      float mel_val = 0.0f;
      const float* weights = &g_mel_weights[filt.weight_offset];
      const float* powers = &s_power_spectrum[filt.start_bin];

      for (int b = 0; b < filt.num_bins; ++b) {
        mel_val += powers[b] * weights[b];
      }

      // Store in row-major matrix: [m * 44 + f]
      s_mel_matrix[m * AUDIO_DSP_N_FRAMES + f] = mel_val;
    }
  }

  // Find global maximum for power_to_db reference
  float max_val = 0.0f;
  for (int i = 0; i < AUDIO_DSP_SPECTROGRAM_SIZE; ++i) {
    if (s_mel_matrix[i] > max_val) {
      max_val = s_mel_matrix[i];
    }
  }

  // Convert to dB: 10 * log10(max(val, 1e-10)) - 10 * log10(max(max_val, 1e-10)) clamped to [-80, 0] dB
  const float ref_db = 10.0f * log10f(fmaxf(max_val, 1e-10f));
  for (int i = 0; i < AUDIO_DSP_SPECTROGRAM_SIZE; ++i) {
    float val_db = 10.0f * log10f(fmaxf(s_mel_matrix[i], 1e-10f)) - ref_db;
    if (val_db < -80.0f) val_db = -80.0f;
    s_mel_matrix[i] = val_db;
  }

  // Global zero-mean, unit-variance normalization: (dB - mean) / (std + 1e-6)
  double sum = 0.0;
  for (int i = 0; i < AUDIO_DSP_SPECTROGRAM_SIZE; ++i) {
    sum += s_mel_matrix[i];
  }
  float mean = (float)(sum / (double)AUDIO_DSP_SPECTROGRAM_SIZE);

  double sum_sq = 0.0;
  for (int i = 0; i < AUDIO_DSP_SPECTROGRAM_SIZE; ++i) {
    float diff = s_mel_matrix[i] - mean;
    sum_sq += (double)(diff * diff);
  }
  float std_dev = sqrtf((float)(sum_sq / (double)AUDIO_DSP_SPECTROGRAM_SIZE));
  float inv_std = 1.0f / (std_dev + 1e-6f);

  for (int i = 0; i < AUDIO_DSP_SPECTROGRAM_SIZE; ++i) {
    out_spectrogram[i] = (s_mel_matrix[i] - mean) * inv_std;
  }

  return true;
}
