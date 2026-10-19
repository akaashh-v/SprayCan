#ifndef STAGE1_FILTER_H_
#define STAGE1_FILTER_H_

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

struct Stage1Result {
  bool is_candidate;
  bool is_clipped;
  float rms;
  float peak;
  int burst_count;
  float cadence_hz;
  float contrast_db;
  const char* reason;
};

/**
 * Fast kinematic pre-filter evaluating back-and-forth shaking burst cadence.
 * Runs in < 0.5 ms on Core 0 to gate execution of the heavy Stage 2 CNN.
 */
inline Stage1Result stage1_analyze_window(const float* audio, size_t num_samples) {
  Stage1Result res = {false, false, 0.0f, 0.0f, 0, 0.0f, 0.0f, "Empty"};
  if (!audio || num_samples == 0) return res;

  // 1. RMS & Peak Calculation
  double sum_sq = 0.0;
  float max_abs = 0.0f;
  size_t clipped_count = 0;

  for (size_t i = 0; i < num_samples; ++i) {
    float s = audio[i];
    float a = fabsf(s);
    if (a > max_abs) max_abs = a;
    if (a >= 0.995f) clipped_count++;
    sum_sq += (double)(s * s);
  }

  res.peak = max_abs;
  res.rms = sqrtf((float)(sum_sq / (double)num_samples));

  // Digital clipping check
  float clip_ratio = (float)clipped_count / (float)num_samples;
  if (clip_ratio > 0.001f || max_abs >= 1.02f) {
    res.is_clipped = true;
    res.reason = "Digital clipping / unusable audio";
    return res;
  }

  // Energy floor check (silent / ambient floor)
  if (res.rms < 0.0010f) {
    res.reason = "Below RMS energy floor";
    return res;
  }

  // 2. Short-Time Envelope & Burst Analysis (20 sub-windows of 50 ms)
  constexpr int NUM_SLICES = 20;
  const size_t slice_len = num_samples / NUM_SLICES;
  float slice_rms[NUM_SLICES];
  float min_slice_rms = 1e9f;
  float max_slice_rms = 0.0f;

  for (int s = 0; s < NUM_SLICES; ++s) {
    double s_sum = 0.0;
    size_t offset = s * slice_len;
    for (size_t i = 0; i < slice_len; ++i) {
      float v = audio[offset + i];
      s_sum += (double)(v * v);
    }
    float r = sqrtf((float)(s_sum / (double)slice_len));
    slice_rms[s] = r;
    if (r < min_slice_rms) min_slice_rms = r;
    if (r > max_slice_rms) max_slice_rms = r;
  }

  // Peak-to-valley contrast
  if (min_slice_rms < 1e-6f) min_slice_rms = 1e-6f;
  float contrast_ratio = max_slice_rms / min_slice_rms;
  res.contrast_db = 20.0f * log10f(contrast_ratio);

  // Count local burst peaks across the 20 slices
  int bursts = 0;
  float burst_thresh = max_slice_rms * 0.45f;
  for (int s = 1; s < NUM_SLICES - 1; ++s) {
    if (slice_rms[s] > burst_thresh &&
        slice_rms[s] > slice_rms[s - 1] &&
        slice_rms[s] >= slice_rms[s + 1]) {
      bursts++;
    }
  }

  res.burst_count = bursts;
  res.cadence_hz = (float)bursts; // in a 1.0s window, bursts = Hz

  // Kinematic Shaking Criteria:
  // Repetitive back-and-forth impacts: 2 to 7 bursts per second + significant valley contrast (> 3.5 dB)
  if (bursts >= 2 && bursts <= 9 && res.contrast_db >= 3.5f) {
    res.is_candidate = true;
    res.reason = "Periodic kinematic shaking candidate";
  } else if (res.rms > 0.0080f && res.contrast_db >= 4.0f) {
    // High energy transient burst
    res.is_candidate = true;
    res.reason = "High energy transient candidate";
  } else {
    res.is_candidate = false;
    res.reason = "Non-candidate (smooth or irregular envelope)";
  }

  return res;
}

#endif // STAGE1_FILTER_H_
