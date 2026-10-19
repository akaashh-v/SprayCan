#include "edge_detector.h"
#include "model_data.h"

#include <Arduino.h>
#include <math.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Tensor arena size: 260 KB allocated in PSRAM
constexpr size_t TENSOR_ARENA_SIZE = 260 * 1024;

static uint8_t* s_tensor_arena = nullptr;
static const tflite::Model* s_model = nullptr;
static tflite::MicroInterpreter* s_interpreter = nullptr;
static bool s_edge_ready = false;

static const char* const s_class_names[EDGE_DETECTOR_NUM_CLASSES] = {
  "ambient_and_speech",
  "loud_impact",
  "continuous_environmental",
  "hard_negative_rattle",
  "aerosol_spray_hiss",
  "spray_can_shaking"
};

const char* edge_detector_get_class_name(int class_index) {
  if (class_index >= 0 && class_index < EDGE_DETECTOR_NUM_CLASSES) {
    return s_class_names[class_index];
  }
  return "unknown";
}

bool edge_detector_is_ready(void) {
  return s_edge_ready;
}

bool edge_detector_init(void) {
  if (s_edge_ready) return true;

  Serial.println("[TFLM] Initializing Edge TensorFlow Lite Micro Runtime...");

  // 1. Verify model buffer
  s_model = tflite::GetModel(g_best_detector_model_data);
  if (s_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[TFLM] ERROR: Model schema mismatch! Got %u, expected %d\n",
                  (unsigned int)s_model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  Serial.printf("[TFLM] Model validated. Binary size: %u bytes\n",
                (unsigned int)g_best_detector_model_data_len);

  // 2. Allocate Tensor Arena in PSRAM
  if (psramFound()) {
    s_tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
    if (s_tensor_arena) {
      Serial.printf("[TFLM] Allocated %u KB Tensor Arena in PSRAM\n",
                    (unsigned int)(TENSOR_ARENA_SIZE / 1024));
    }
  }

  // Fallback to internal heap if PSRAM allocation failed or not present
  if (!s_tensor_arena) {
    s_tensor_arena = (uint8_t*)malloc(TENSOR_ARENA_SIZE);
    if (!s_tensor_arena) {
      Serial.printf("[TFLM] FATAL ERROR: Cannot allocate %u KB Tensor Arena!\n",
                    (unsigned int)(TENSOR_ARENA_SIZE / 1024));
      return false;
    }
    Serial.printf("[TFLM] Allocated %u KB Tensor Arena in internal SRAM\n",
                  (unsigned int)(TENSOR_ARENA_SIZE / 1024));
  }

  // 3. Register exact operators used by the model
  static tflite::MicroMutableOpResolver<15> resolver;
  resolver.AddConcatenation();
  resolver.AddConv2D();
  resolver.AddDequantize();
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddPack();
  resolver.AddQuantize();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddSoftmax();
  resolver.AddStridedSlice();
  resolver.AddTranspose();

  // 4. Construct MicroInterpreter instance
  s_interpreter = new tflite::MicroInterpreter(
    s_model,
    resolver,
    s_tensor_arena,
    TENSOR_ARENA_SIZE
  );

  if (!s_interpreter) {
    Serial.println("[TFLM] ERROR: MicroInterpreter creation failed!");
    return false;
  }

  // 5. Allocate tensors from the arena
  TfLiteStatus alloc_status = s_interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.printf("[TFLM] ERROR: AllocateTensors() failed (status %d)!\n", alloc_status);
    return false;
  }

  // 6. Verify input and output tensor dimensions
  TfLiteTensor* input = s_interpreter->input(0);
  TfLiteTensor* output = s_interpreter->output(0);

  if (!input || !output) {
    Serial.println("[TFLM] ERROR: Input/Output tensor pointers are null!");
    return false;
  }

  Serial.printf("[TFLM] Input Tensor: dims=[%d,%d,%d,%d], type=%d, bytes=%u\n",
                input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3],
                input->type, (unsigned int)input->bytes);
  Serial.printf("[TFLM] Output Tensor: dims=[%d,%d], type=%d, bytes=%u\n",
                output->dims->data[0], output->dims->data[1], output->type, (unsigned int)output->bytes);
  Serial.printf("[TFLM] Arena Used: %u / %u bytes (%u%%)\n",
                (unsigned int)s_interpreter->arena_used_bytes(), (unsigned int)TENSOR_ARENA_SIZE,
                (unsigned int)((s_interpreter->arena_used_bytes() * 100) / TENSOR_ARENA_SIZE));

  s_edge_ready = true;
  Serial.println("[TFLM] Edge CNN Inference Engine Ready on ESP32-WROVER-B.");
  return true;
}

bool edge_detector_run_inference(const float* spectrogram, EdgePredictionResult* result) {
  if (!s_edge_ready || !spectrogram || !result) {
    return false;
  }

  TfLiteTensor* input = s_interpreter->input(0);
  if (!input || !input->data.f) {
    return false;
  }

  // Copy 2816 float32 spectrogram values into model input tensor
  memcpy(input->data.f, spectrogram, 64 * 44 * sizeof(float));

  // Run model inference and measure exact execution time
  unsigned long t0 = millis();
  TfLiteStatus invoke_status = s_interpreter->Invoke();
  unsigned long t1 = millis();
  result->inference_time_ms = (t1 >= t0) ? (t1 - t0) : 0;

  if (invoke_status != kTfLiteOk) {
    Serial.printf("[TFLM] ERROR: Invoke() failed with code %d\n", invoke_status);
    return false;
  }

  // Read output probabilities
  TfLiteTensor* output = s_interpreter->output(0);
  if (!output || !output->data.f) {
    return false;
  }

  int pred_idx = 0;
  float max_prob = -1.0f;
  float max_comp_prob = -1.0f;
  int max_comp_idx = 0;
  float entropy = 0.0f;

  for (int i = 0; i < EDGE_DETECTOR_NUM_CLASSES; ++i) {
    float p = output->data.f[i];
    if (p < 0.0f) p = 0.0f;
    if (p > 1.0f) p = 1.0f;
    result->probs[i] = p;

    if (p > max_prob) {
      max_prob = p;
      pred_idx = i;
    }

    if (i != EDGE_DETECTOR_CLASS_SPRAY) {
      if (p > max_comp_prob) {
        max_comp_prob = p;
        max_comp_idx = i;
      }
    }

    if (p > 1e-7f) {
      entropy -= p * logf(p);
    }
  }

  result->p_spray = result->probs[EDGE_DETECTOR_CLASS_SPRAY];
  result->predicted_class = pred_idx;
  result->max_competing_prob = max_comp_prob;
  result->max_competing_class = max_comp_idx;
  result->confidence_margin = result->p_spray - max_comp_prob;
  result->entropy = entropy;

  // Decision rule matching Python evaluate_candidate_prediction:
  // 1. Conservative threshold check (p_spray >= 0.80)
  // 2. Clear dominance margin (margin >= 0.15)
  // 3. Low ambiguity entropy (entropy <= 1.25)
  bool threshold_met = (result->p_spray >= EDGE_DETECTOR_CONSERVATIVE_THRESHOLD);
  bool margin_met = (result->confidence_margin >= EDGE_DETECTOR_MIN_MARGIN);
  bool entropy_met = (result->entropy <= EDGE_DETECTOR_MAX_ENTROPY);

  result->is_uncertain = (!entropy_met || (result->p_spray > 0.40f && !threshold_met));

  if (threshold_met && margin_met && entropy_met) {
    result->alert_confirmed = true;
    result->verdict = "CONFIRMED_ALERT";
  } else if (result->is_uncertain) {
    result->alert_confirmed = false;
    result->verdict = "UNCERTAIN_AMBIGUOUS_SOUND";
  } else {
    result->alert_confirmed = false;
    result->verdict = "REJECTED_NON_TARGET";
  }

  return true;
}
