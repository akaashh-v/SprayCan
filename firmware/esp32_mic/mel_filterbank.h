#ifndef MEL_FILTERBANK_H_
#define MEL_FILTERBANK_H_

#include <stdint.h>

#define MEL_NUM_BINS 64
#define FFT_SIZE 1024
#define FFT_HALF_BINS 513
#define HOP_SIZE 512
#define NUM_TIME_FRAMES 44
#define TOTAL_MEL_WEIGHTS 985

struct MelFilterDef {
  uint16_t start_bin;
  uint16_t num_bins;
  uint16_t weight_offset;
};

extern const MelFilterDef g_mel_filters[MEL_NUM_BINS];
extern const float g_mel_weights[TOTAL_MEL_WEIGHTS];

#endif // MEL_FILTERBANK_H_
