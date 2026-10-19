#ifndef EDGE_DETECTOR_H_
#define EDGE_DETECTOR_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EDGE_DETECTOR_NUM_CLASSES 6
#define EDGE_DETECTOR_CLASS_SPRAY 5
#define EDGE_DETECTOR_CONSERVATIVE_THRESHOLD 0.80f
#define EDGE_DETECTOR_MIN_MARGIN 0.15f
#define EDGE_DETECTOR_MAX_ENTROPY 1.25f

struct EdgePredictionResult {
  float probs[EDGE_DETECTOR_NUM_CLASSES];
  float p_spray;
  float max_competing_prob;
  int predicted_class;
  int max_competing_class;
  float confidence_margin;
  float entropy;
  bool is_uncertain;
  bool alert_confirmed;
  unsigned long preprocess_time_ms;
  unsigned long inference_time_ms;
  const char* verdict;
};

/**
 * Initializes the TFLite Micro runtime, registers required operators,
 * and allocates the working Tensor Arena in PSRAM.
 * @return true on success, false on error.
 */
bool edge_detector_init(void);

/**
 * Checks if the edge detector runtime is successfully initialized and ready.
 */
bool edge_detector_is_ready(void);

/**
 * Executes on-device CNN inference on a 64x44 Log-Mel spectrogram.
 * @param spectrogram Input Log-Mel Spectrogram array of 2816 floats (row-major 64x44).
 * @param result Output prediction structure.
 * @return true if inference succeeded, false on error.
 */
bool edge_detector_run_inference(const float* spectrogram, EdgePredictionResult* result);

/**
 * Returns the human-readable class name for a given class index (0-5).
 */
const char* edge_detector_get_class_name(int class_index);

#ifdef __cplusplus
}
#endif

#endif // EDGE_DETECTOR_H_
