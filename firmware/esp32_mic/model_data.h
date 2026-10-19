#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <stddef.h>

#ifdef __has_attribute
#define TFLM_ALIGNED(x) __attribute__((aligned(x)))
#else
#define TFLM_ALIGNED(x)
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_best_detector_model_data[] TFLM_ALIGNED(16);
extern const size_t g_best_detector_model_data_len;

#ifdef __cplusplus
}
#endif

#endif  // MODEL_DATA_H_
