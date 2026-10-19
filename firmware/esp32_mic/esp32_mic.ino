/**
 * Acoustic Spray-Can Shaking Detector - ESP32 Microphone Firmware
 * Hardware: ESP32-WROVER-R + INMP441 I2S Microphone
 *
 * Operational Mode:
 * - Autonomous Wi-Fi Command Polling: Starts & stops recording directly from web dashboard.
 * - Authenticated audio uploads (22050 Hz 1.0s WAV) to /esp32/test-audio.
 * - Low-latency state acknowledgement to /esp32/status.
 * - Non-blocking Wi-Fi loss recovery and network timeout protection.
 */

#include <WiFi.h>
#include <ESP_I2S.h>
#include <HTTPClient.h>
#include <math.h>
#include "audio_dsp.h"
#include "edge_detector.h"
#include "stage1_filter.h"

// =========================================================================
// Network & Server Configuration
// =========================================================================

// Wi-Fi Access Point Credentials
const char* WIFI_SSID = "JHF IT Innovations_2.4";
const char* WIFI_PASSWORD = "JHFHiLite@AskJHFIT#2.4";

// Backend Receiver Server Host & Port (5001 default)
const char* SERVER_HOST = "192.168.1.14";
const int   SERVER_PORT = 5001;

// Device Authentication Pre-Shared Key (matches ESP32_TEST_TOKEN)
const char* API_TOKEN = "SprayDetectorTest_ChangeThis_8392";

// Timing & Polling Settings
const unsigned long COMMAND_POLL_INTERVAL_MS = 600;   // Poll backend command every 600ms when idle
const unsigned long HTTP_TIMEOUT_MS          = 3000;  // 3.0s HTTP timeout prevents hanging
const unsigned long WIFI_RECONNECT_RETRY_MS  = 3000;  // Retry interval on Wi-Fi loss

// INMP441 I2S Microphone Pins
constexpr int BCLK_PIN = 23;
constexpr int WS_PIN   = 33;
constexpr int SD_PIN   = 32;

// =========================================================================
// Global State & Dual-Core Audio Transport Handles
// =========================================================================

struct AudioChunk {
  uint8_t* data;
  size_t size;
  unsigned long captureStart;
  unsigned long captureEnd;
  unsigned long enqueueTime;
  unsigned long captureSeq;
  bool edgeEvaluated;
  EdgePredictionResult edgeResult;
};

I2SClass I2S;
volatile bool recordingEnabled = false;
volatile bool diagnosticMode = false;
volatile bool manualTestActive = false;
unsigned long manualTestStartTime = 0;
const unsigned long MANUAL_TEST_DURATION_MS = 5000;
unsigned long lastCommandPollTime = 0;
unsigned long lastWifiAttemptTime = 0;
int consecutiveUploadFailures = 0;

// Explicit sequence & telemetry counters
unsigned long captureSeqCounter = 0;
unsigned long uploadAttemptCounter = 0;
unsigned long droppedChunkCounter = 0;
unsigned long lastDropLogTime = 0;

QueueHandle_t audioQueue = nullptr;
QueueHandle_t uploadQueue = nullptr;
TaskHandle_t captureTaskHandle = nullptr;
TaskHandle_t uploadTaskHandle = nullptr;

String serverBaseUrl;
String uploadUrl;
String commandUrl;
String statusUrl;
String alertUrl;

// Forward declaration for background upload task
void backgroundUploadTask(void* parameter);

// =========================================================================
// Diagnostic Telemetry & Audio Metric Analysis
// =========================================================================

struct AudioStats {
  int32_t minVal;
  int32_t maxVal;
  double meanVal;
  double rms;
  double peakNorm;
  double peakDbFS;
  double rmsDbFS;
  double clippingPct;
  size_t numSamples;
  uint16_t bitsPerSample;
  bool isSilent;
};

/**
 * Analyzes a 1.0-second WAV audio buffer and computes statistical telemetry.
 * Parses the standard 44-byte RIFF/WAV header and analyzes raw PCM samples.
 */
AudioStats analyzeWavBuffer(const uint8_t* wavData, size_t wavSize) {
  AudioStats stats = {0, 0, 0.0, 0.0, 0.0, -96.0, -96.0, 0.0, 0, 16, true};
  if (wavData == nullptr || wavSize < 44) return stats;

  uint16_t bitsPerSample = *(const uint16_t*)&wavData[34];
  stats.bitsPerSample = bitsPerSample;

  if (bitsPerSample == 16) {
    const int16_t* samples = (const int16_t*)(wavData + 44);
    size_t count = (wavSize - 44) / sizeof(int16_t);
    if (count == 0) return stats;
    stats.numSamples = count;

    int32_t minS = 32767;
    int32_t maxS = -32768;
    int64_t sumS = 0;
    double sumSq = 0.0;
    int32_t peakAbs = 0;
    size_t clipped = 0;

    for (size_t i = 0; i < count; ++i) {
      int16_t s = samples[i];
      if (s < minS) minS = s;
      if (s > maxS) maxS = s;
      sumS += s;
      int32_t absS = abs((int32_t)s);
      if (absS > peakAbs) peakAbs = absS;
      if (absS >= 32700) clipped++;
      double norm = (double)s / 32768.0;
      sumSq += norm * norm;
    }

    stats.minVal = minS;
    stats.maxVal = maxS;
    stats.meanVal = (double)sumS / (double)count;
    stats.rms = sqrt(sumSq / (double)count);
    stats.peakNorm = (double)peakAbs / 32768.0;
    stats.peakDbFS = (stats.peakNorm > 1e-6) ? (20.0 * log10(stats.peakNorm)) : -96.0;
    stats.rmsDbFS = (stats.rms > 1e-6) ? (20.0 * log10(stats.rms)) : -96.0;
    stats.clippingPct = ((double)clipped * 100.0) / (double)count;
    stats.isSilent = (stats.rms < 0.0004);
  } else if (bitsPerSample == 32) {
    const int32_t* samples = (const int32_t*)(wavData + 44);
    size_t count = (wavSize - 44) / sizeof(int32_t);
    if (count == 0) return stats;
    stats.numSamples = count;

    int32_t minS = 2147483647;
    int32_t maxS = -2147483648;
    double sumS = 0.0;
    double sumSq = 0.0;
    int64_t peakAbs = 0;
    size_t clipped = 0;

    for (size_t i = 0; i < count; ++i) {
      int32_t s = samples[i];
      if (s < minS) minS = s;
      if (s > maxS) maxS = s;
      sumS += (double)s;
      int64_t absS = (s < 0) ? -(int64_t)s : (int64_t)s;
      if (absS > peakAbs) peakAbs = absS;
      if (absS >= 2140000000) clipped++;
      double norm = (double)s / 2147483648.0;
      sumSq += norm * norm;
    }

    stats.minVal = minS;
    stats.maxVal = maxS;
    stats.meanVal = sumS / (double)count;
    stats.rms = sqrt(sumSq / (double)count);
    stats.peakNorm = (double)peakAbs / 2147483648.0;
    stats.peakDbFS = (stats.peakNorm > 1e-6) ? (20.0 * log10(stats.peakNorm)) : -96.0;
    stats.rmsDbFS = (stats.rms > 1e-6) ? (20.0 * log10(stats.rms)) : -96.0;
    stats.clippingPct = ((double)clipped * 100.0) / (double)count;
    stats.isSilent = (stats.rms < 0.0004);
  }
  return stats;
}

/**
 * Prints comprehensive microphone statistics and system telemetry to Serial Monitor.
 */
void printMicrophoneDiagnostics(const AudioStats& stats, unsigned long captureDurMs) {
  Serial.println("\n================ [I2S MIC DIAGNOSTIC - 1 SEC SUMMARY] ================");
  Serial.printf("  Hardware: ESP32 (%s rev %d @ %u MHz) | Core: 3.3.12\n",
                ESP.getChipModel(), ESP.getChipRevision(), ESP.getCpuFreqMHz());
  Serial.printf("  Interface: I2S_MODE_STD | BCLK: %d, WS: %d, SD: %d | Slot: Mono Left\n",
                BCLK_PIN, WS_PIN, SD_PIN);
  Serial.printf("  Audio Format: 22,050 Hz | %u-bit WAV PCM | Window: 1.0s (%lu ms actual)\n",
                stats.bitsPerSample, captureDurMs);

  if (psramFound()) {
    Serial.printf("  PSRAM: DETECTED & ACTIVE | Total: %u KB | Free: %u KB (Min Free: %u KB)\n",
                  ESP.getPsramSize() / 1024, ESP.getFreePsram() / 1024, ESP.getMinFreePsram() / 1024);
  } else {
    Serial.println("  PSRAM: NOT DETECTED (Using Internal SRAM)");
  }

  Serial.printf("  Internal Heap: Free: %u KB | Min Free: %u KB | Max Alloc: %u KB\n",
                ESP.getFreeHeap() / 1024, ESP.getMinFreeHeap() / 1024, ESP.getMaxAllocHeap() / 1024);
  Serial.printf("  Samples Captured: %u | Dropped Stale Chunks: %lu\n",
                (unsigned int)stats.numSamples, droppedChunkCounter);
  Serial.printf("  Sample Min: %+ld | Sample Max: %+ld | DC Offset (Mean): %+.2f\n",
                (long)stats.minVal, (long)stats.maxVal, stats.meanVal);
  Serial.printf("  RMS Amplitude: %.5f (%+.1f dBFS) | Peak: %.5f (%+.1f dBFS)\n",
                stats.rms, stats.rmsDbFS, stats.peakNorm, stats.peakDbFS);
  
  const char* sigStatus = stats.isSilent ? "[QUIET / SILENCE]" :
                          (stats.rms > 0.0080 ? "[HIGH ENERGY / SHAKING CANDIDATE]" : "[ACTIVE SOUND / SPEECH]");
  Serial.printf("  Clipping: %.2f%% | Signal Assessment: %s\n",
                stats.clippingPct, sigStatus);
  Serial.println("======================================================================");
}

// =========================================================================
// 16-bit PCM WAV Conversion & Buffer Management
// =========================================================================

/**
 * Constructs a canonical 44-byte RIFF/WAV header for 16-bit PCM Mono audio @ 22,050 Hz.
 */
void writeWavHeader16(uint8_t* header, uint32_t numSamples, uint32_t sampleRate) {
  uint32_t dataSize = numSamples * sizeof(int16_t);
  uint32_t riffSize = 36 + dataSize;
  uint32_t byteRate = sampleRate * sizeof(int16_t);
  uint16_t blockAlign = sizeof(int16_t);

  // RIFF chunk
  header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
  *(uint32_t*)&header[4] = riffSize;
  header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';

  // fmt subchunk
  header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
  *(uint32_t*)&header[16] = 16;           // Subchunk1Size = 16 for PCM
  *(uint16_t*)&header[20] = 1;            // AudioFormat = 1 (Linear PCM)
  *(uint16_t*)&header[22] = 1;            // NumChannels = 1 (Mono)
  *(uint32_t*)&header[24] = sampleRate;   // SampleRate = 22050
  *(uint32_t*)&header[28] = byteRate;     // ByteRate = 44100
  *(uint16_t*)&header[32] = blockAlign;   // BlockAlign = 2
  *(uint16_t*)&header[34] = 16;           // BitsPerSample = 16

  // data subchunk
  header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
  *(uint32_t*)&header[40] = dataSize;     // Subchunk2Size
}

/**
 * Converts a 32-bit I2S WAV buffer (from recordWAV) to a compact 16-bit PCM WAV.
 * - Extracts the 24-bit audio aligned in the upper bits of the 32-bit I2S slot.
 * - Scales via arithmetic right shift (>> 16) with two's complement sign extension.
 * - Clamps to [-32768, 32767] for saturation protection.
 * - Allocates output buffer in PSRAM if available, preserving internal SRAM.
 * - Generates valid 44,144-byte mono 22,050 Hz 16-bit WAV.
 */
uint8_t* convert32to16Wav(const uint8_t* rawWav, size_t rawSize, size_t* outSize) {
  *outSize = 0;
  if (rawWav == nullptr || rawSize < 44) return nullptr;

  uint16_t rawBits = *(const uint16_t*)&rawWav[34];
  if (rawBits != 32) {
    // Already 16-bit or other format: copy as-is
    uint8_t* copyBuf = (uint8_t*)(psramFound() ? ps_malloc(rawSize) : malloc(rawSize));
    if (copyBuf != nullptr) {
      memcpy(copyBuf, rawWav, rawSize);
      *outSize = rawSize;
    }
    return copyBuf;
  }

  size_t numSamples = (rawSize - 44) / sizeof(int32_t);
  if (numSamples == 0) return nullptr;

  size_t pcm16Size = 44 + (numSamples * sizeof(int16_t));
  uint8_t* pcm16Buf = (uint8_t*)(psramFound() ? ps_malloc(pcm16Size) : malloc(pcm16Size));
  if (pcm16Buf == nullptr) {
    Serial.println("[!] ERROR: Failed to allocate 16-bit audio buffer in PSRAM/SRAM!");
    return nullptr;
  }

  // Write standard 16-bit WAV header
  writeWavHeader16(pcm16Buf, numSamples, 22050);

  // Convert samples with two's complement arithmetic shift and saturation clamp
  const int32_t* src32 = (const int32_t*)(rawWav + 44);
  int16_t* dst16 = (int16_t*)(pcm16Buf + 44);

  for (size_t i = 0; i < numSamples; ++i) {
    int32_t s = src32[i] >> 16;
    if (s > 32767) s = 32767;
    else if (s < -32768) s = -32768;
    dst16[i] = (int16_t)s;
  }

  *outSize = pcm16Size;
  return pcm16Buf;
}

// =========================================================================
// Wi-Fi Management
// =========================================================================

void buildEndpoints() {
  serverBaseUrl = "http://" + String(SERVER_HOST) + ":" + String(SERVER_PORT);
  uploadUrl  = serverBaseUrl + "/esp32/test-audio";
  commandUrl = serverBaseUrl + "/esp32/command";
  statusUrl  = serverBaseUrl + "/esp32/status";
  alertUrl   = serverBaseUrl + "/esp32/alert";
}

bool ensureWiFiConnected() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }

  unsigned long now = millis();
  if (now - lastWifiAttemptTime >= WIFI_RECONNECT_RETRY_MS) {
    lastWifiAttemptTime = now;
    Serial.println("[Wi-Fi] Connection lost. Attempting non-blocking reconnect...");
    WiFi.reconnect();
    if (WiFi.status() == WL_CONNECTED) {
      WiFi.setSleep(false);
    }
  }
  return false;
}

// =========================================================================
// Backend State Acknowledgement & Command Polling
// =========================================================================

/**
 * Send explicit state change confirmation (ACK) to backend.
 * @param stateString "LISTENING" or "STOPPED"
 */
bool sendStateAck(const char* stateString) {
  if (!ensureWiFiConnected()) return false;

  HTTPClient http;
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.begin(statusUrl);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-Token", API_TOKEN);
  http.addHeader("X-Device-State", stateString);

  String payload = "{\"state\":\"" + String(stateString) + "\"}";
  int httpCode = http.POST(payload);

  if (httpCode > 0) {
    Serial.printf("[STATE-ACK] State '%s' acknowledged (HTTP %d)\n", stateString, httpCode);
  } else {
    Serial.printf("[STATE-ACK] Failed to send state '%s' (%s)\n", stateString, http.errorToString(httpCode).c_str());
  }

  http.end();
  return (httpCode >= 200 && httpCode < 300);
}

/**
 * Poll backend /esp32/command to check desired operational state.
 */
void checkBackendCommand() {
  if (!ensureWiFiConnected()) return;

  HTTPClient http;
  http.setTimeout(HTTP_TIMEOUT_MS); 

  String pollUrl = commandUrl + "?state=" + (recordingEnabled ? "LISTENING" : "STOPPED");
  http.begin(pollUrl);
  http.addHeader("X-Device-Token", API_TOKEN);
  http.addHeader("X-Device-State", recordingEnabled ? "LISTENING" : "STOPPED");

  int httpCode = http.GET();

  if (httpCode == 200) {
    String body = http.getString();
    
    // Check command from JSON response without heavy JSON parser
    if (body.indexOf("\"command\":\"START\"") >= 0 || body.indexOf("\"command\": \"START\"") >= 0) {
      if (!recordingEnabled) {
        Serial.println("\n[COMMAND-POLL] >>> START LISTENING command received from Web Dashboard <<<");
        recordingEnabled = true;
        consecutiveUploadFailures = 0;
        sendStateAck("LISTENING");
      }
    } else if (body.indexOf("\"command\":\"STOP\"") >= 0 || body.indexOf("\"command\": \"STOP\"") >= 0) {
      if (recordingEnabled) {
        Serial.println("\n[COMMAND-POLL] >>> STOP LISTENING command received from Web Dashboard <<<");
        recordingEnabled = false;
        sendStateAck("STOPPED");
      }
    }

    // When stopped and idle, print periodic poll status (every 10 polls) so Serial monitor remains informative
    if (!recordingEnabled) {
      static unsigned long idlePollCounter = 0;
      idlePollCounter++;
      if (idlePollCounter % 10 == 1) {
        Serial.printf("[COMMAND-POLL] Idle polling backend (state: STOPPED) -> awaiting Start Listening (poll #%lu)\n", idlePollCounter);
      }
    }
  } else if (httpCode > 0) {
    Serial.printf("[COMMAND-POLL] Server returned HTTP %d\n", httpCode);
  } else {
    // Network or connection error (server down or IP unreachable)
  }

  http.end();
}

/**
 * Optional Serial Monitor diagnostic commands (fallback for manual testing).
 */
void checkSerialCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;

    if (c == 'D' || c == 'd') {
      diagnosticMode = !diagnosticMode;
      Serial.println("\n-------------------------------------------------------------");
      Serial.printf("[DIAGNOSTIC] Mode %s.\n",
                    diagnosticMode ? "ENABLED (Printing live 1.0s audio metrics to Serial)" : "DISABLED");
      Serial.println("-------------------------------------------------------------");
    } else if (c == 'M' || c == 'm') {
      if (!recordingEnabled) {
        Serial.println("\n[MANUAL-TEST] Initiating 5-second standalone microphone test...");
        manualTestActive = true;
        manualTestStartTime = millis();
        recordingEnabled = true;
        sendStateAck("LISTENING");
      } else {
        Serial.println("[MANUAL-TEST] Already recording/listening.");
      }
    } else if (c == 'A' || c == 'a') {
      if (!recordingEnabled) {
        Serial.println("\n[MANUAL] Starting recording via Serial command 'A'...");
        recordingEnabled = true;
        sendStateAck("LISTENING");
      } else {
        Serial.println("[MANUAL] Already recording.");
      }
    } else if (c == 'B' || c == 'b') {
      if (recordingEnabled) {
        Serial.println("\n[MANUAL] Stopping recording via Serial command 'B'...");
        recordingEnabled = false;
        manualTestActive = false;
        sendStateAck("STOPPED");
      } else {
        Serial.println("[MANUAL] Already stopped.");
      }
    } else if (c == 'S' || c == 's') {
      Serial.printf("[STATUS] IP: %s | State: %s | DiagMode: %s | URL: %s\n",
                    WiFi.localIP().toString().c_str(),
                    recordingEnabled ? "LISTENING" : "STOPPED",
                    diagnosticMode ? "ON" : "OFF",
                    serverBaseUrl.c_str());
    } else if (c == 'H' || c == 'h' || c == '?') {
      Serial.println("\n=== Serial Commands Menu ===");
      Serial.println("  D : Toggle 1-second microphone diagnostic metrics (RMS, peak, clipping, min/max)");
      Serial.println("  M : Run 5-second manual microphone test capture");
      Serial.println("  A : Start continuous audio capture & streaming");
      Serial.println("  B : Stop continuous audio capture");
      Serial.println("  S : Show status (IP, state, PSRAM, heap)");
      Serial.println("=============================");
    }
  }
}

// =========================================================================
// Setup
// =========================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=======================================================");
  Serial.println("  ESP32-WROVER-R Acoustic Spray-Can Shaking Detector");
  Serial.println("  Autonomous Web Dashboard Wi-Fi Controller Active");
  Serial.println("=======================================================");

  // Hardware & Memory Diagnostics
  Serial.println("\n--- Hardware & Memory Telemetry ---");
  Serial.printf("  Chip Model:    ESP32 (%s rev %d @ %u MHz)\n",
                ESP.getChipModel(), ESP.getChipRevision(), ESP.getCpuFreqMHz());
  Serial.printf("  SDK Version:   %s\n", ESP.getSdkVersion());

  if (psramFound()) {
    Serial.println("  PSRAM:         DETECTED & ACTIVE");
    Serial.printf("  PSRAM Size:    %u KB (Free: %u KB, Min Free: %u KB)\n",
                  ESP.getPsramSize() / 1024, ESP.getFreePsram() / 1024, ESP.getMinFreePsram() / 1024);
  } else {
    Serial.println("  PSRAM:         NOT DETECTED (Using Internal SRAM)");
  }

  Serial.printf("  Internal Heap: %u KB (Free: %u KB, Min Free: %u KB, Max Alloc: %u KB)\n",
                ESP.getHeapSize() / 1024, ESP.getFreeHeap() / 1024, ESP.getMinFreeHeap() / 1024, ESP.getMaxAllocHeap() / 1024);
  Serial.printf("  I2S Pins:      BCLK=%d, WS=%d, SD=%d (Mono Left)\n", BCLK_PIN, WS_PIN, SD_PIN);
  Serial.println("-----------------------------------");

  buildEndpoints();

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.printf("Connecting to Wi-Fi '%s'...", WIFI_SSID);
  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED && (millis() - start < 15000)) {
    delay(400);
    Serial.print(".");
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[!] Wi-Fi initial connection timed out. Will retry in background loop.");
  } else {
    WiFi.setSleep(false);
    Serial.println("\n[+] Wi-Fi connected successfully!");
    Serial.println("[+] Wi-Fi Modem Sleep DISABLED (Full RF performance active)");
    Serial.printf("    ESP32 Local IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("    Target Server:   %s\n", serverBaseUrl.c_str());
  }

  // Initialize INMP441 I2S Microphone: 22,050 Hz, 32-bit Mono
  I2S.setPins(BCLK_PIN, WS_PIN, -1, SD_PIN);

  if (!I2S.begin(
        I2S_MODE_STD,
        22050,
        I2S_DATA_BIT_WIDTH_32BIT,
        I2S_SLOT_MODE_MONO,
        I2S_STD_SLOT_LEFT)) {
    Serial.println("[!] ERROR: I2S microphone initialization failed! Check wiring.");
    return;
  }

  Serial.println("[+] I2S INMP441 microphone initialized (22,050 Hz, Mono, 32-bit)");

  // Initialize bounded audio queue (depth 2 enables pipelined capture while inference runs)
  audioQueue = xQueueCreate(2, sizeof(AudioChunk));

  // Initialize decoupled background upload queue (depth 4 in PSRAM)
  uploadQueue = xQueueCreate(4, sizeof(AudioChunk));

  if (audioQueue != nullptr && uploadQueue != nullptr) {
    Serial.println("[+] Dual FreeRTOS Queues allocated (Capture depth: 2, Upload depth: 4)");
  } else {
    Serial.println("[!] ERROR: FreeRTOS Queue allocation failed!");
  }

  // Initialize on-device Audio DSP & TFLite Micro CNN Edge Detector
  if (!audio_dsp_init()) {
    Serial.println("[!] WARNING: Audio DSP initialization encountered errors!");
  }
  if (!edge_detector_init()) {
    Serial.println("[!] WARNING: Edge CNN Detector initialization failed!");
  }

  // Launch uninterrupted audio capture task on CPU Core 1
  xTaskCreatePinnedToCore(
    audioCaptureTask,
    "AudioCapture",
    8192,
    nullptr,
    2,
    &captureTaskHandle,
    1  // Core 1
  );

  // Launch decoupled background audio upload task on CPU Core 0 (lower priority, non-blocking)
  xTaskCreatePinnedToCore(
    backgroundUploadTask,
    "BgUpload",
    6144,
    nullptr,
    1,
    &uploadTaskHandle,
    0  // Core 0
  );

  Serial.println("[+] System ready. Awaiting 'Start Listening' from Web Dashboard.");
  Serial.println("[+] Tip: Send 'D' in Serial Monitor to toggle live audio metrics.");

  // Confirm stopped state on boot
  sendStateAck("STOPPED");
}

// =========================================================================
// Real-Time Audio Capture Task (FreeRTOS Core 1)
// =========================================================================

/**
 * Safely purge all pending audio chunks in queues and free their memory buffers.
 */
void purgeUploadQueue() {
  if (uploadQueue == nullptr) return;
  AudioChunk stale;
  while (xQueueReceive(uploadQueue, &stale, 0) == pdPASS) {
    if (stale.data != nullptr) {
      free(stale.data);
      stale.data = nullptr;
    }
  }
}

void purgeAudioQueue() {
  if (audioQueue != nullptr) {
    AudioChunk stale;
    while (xQueueReceive(audioQueue, &stale, 0) == pdPASS) {
      if (stale.data != nullptr) {
        free(stale.data);
        stale.data = nullptr;
      }
    }
  }
  purgeUploadQueue();
}

/**
 * Continuous audio acquisition running independently on CPU Core 1.
 * Captures uninterrupted 1.0-second audio windows into a bounded queue (depth 1),
 * ensuring microphone capture is never delayed by network transmission or inference.
 */
void audioCaptureTask(void* parameter) {
  while (true) {
    if (!recordingEnabled) {
      vTaskDelay(pdMS_TO_TICKS(40));
      continue;
    }

    size_t wavSize = 0;
    unsigned long capStart = millis();
    uint8_t* wavData = I2S.recordWAV(1, &wavSize);
    unsigned long capEnd = millis();

    // If stopped while recording, discard buffer immediately
    if (!recordingEnabled) {
      if (wavData != nullptr) {
        free(wavData);
        wavData = nullptr;
      }
      continue;
    }

    if (wavData != nullptr && wavSize >= 44) {
      size_t pcm16Size = 0;
      uint8_t* pcm16Data = convert32to16Wav(wavData, wavSize, &pcm16Size);
      // Immediately free the 88 KB raw 32-bit buffer back to internal heap
      free(wavData);
      wavData = nullptr;

      if (pcm16Data != nullptr && pcm16Size >= 44) {
        captureSeqCounter++;
        AudioChunk chunk;
        chunk.data = pcm16Data;
        chunk.size = pcm16Size;
        chunk.captureStart = capStart;
        chunk.captureEnd = capEnd;
        chunk.enqueueTime = millis();
        chunk.captureSeq = captureSeqCounter;
        chunk.edgeEvaluated = false;
        memset(&chunk.edgeResult, 0, sizeof(EdgePredictionResult));

        // Bounded queue (depth 2): if queue is full, drop oldest chunk to prioritize newest audio
        if (uxQueueSpacesAvailable(audioQueue) == 0) {
          AudioChunk stale;
          if (xQueueReceive(audioQueue, &stale, 0) == pdPASS) {
            if (stale.data != nullptr) {
              free(stale.data);
              stale.data = nullptr;
            }
            droppedChunkCounter++;
            unsigned long now = millis();
            // Rate-limited drop diagnostics: print at most once per 3s to prevent UART flooding
            if (now - lastDropLogTime >= 3000) {
              lastDropLogTime = now;
              Serial.printf("[QUEUE] Bounded audio queue full: dropped %lu stale chunk(s)\n",
                            droppedChunkCounter);
            }
          }
        }
        xQueueSend(audioQueue, &chunk, 0);
      }
    } else {
      if (wavData != nullptr) {
        free(wavData);
        wavData = nullptr;
      }
      vTaskDelay(pdMS_TO_TICKS(20));
    }
  }
}

/**
 * Instantaneous lightweight alert event transmission (< 350 bytes).
 * Bypasses 44 KB audio upload to achieve sub-25ms network dispatch to dashboard.
 */
bool sendEdgeAlert(
    const char* alertType,
    unsigned long captureSeq,
    unsigned long captureDurMs,
    unsigned long s1EvalMs,
    unsigned long dspMs,
    unsigned long cnnMs,
    unsigned long queueWaitMs,
    const EdgePredictionResult* edgeRes = nullptr
) {
  if (WiFi.status() != WL_CONNECTED) return false;

  unsigned long t_tx_start = millis();
  WiFiClient alertClient;
  alertClient.setTimeout(2); // 2-second socket timeout

  if (!alertClient.connect(SERVER_HOST, SERVER_PORT, 1200)) {
    delay(50); // Transient retry
    if (!alertClient.connect(SERVER_HOST, SERVER_PORT, 1200)) {
      Serial.printf("[ALERT-DISPATCH] WARNING: Connection to %s:%d failed!\n", SERVER_HOST, SERVER_PORT);
      return false;
    }
  }
  alertClient.setNoDelay(true);

  char jsonBuf[384];
  char eventId[32];
  snprintf(eventId, sizeof(eventId), "evt_esp32_%lu_%lu", captureSeq, t_tx_start);

  float p_spray = edgeRes ? edgeRes->p_spray : 0.0f;
  float margin = edgeRes ? edgeRes->confidence_margin : 0.0f;
  float entropy = edgeRes ? edgeRes->entropy : 0.0f;
  const char* predClass = edgeRes ? edge_detector_get_class_name(edgeRes->predicted_class) : "candidate";
  const char* verdict = edgeRes ? edgeRes->verdict : "PRELIMINARY_CANDIDATE";
  unsigned long deviceTotalMs = s1EvalMs + dspMs + cnnMs;

  int jsonLen = snprintf(
    jsonBuf, sizeof(jsonBuf),
    "{\"event_id\":\"%s\",\"alert_type\":\"%s\",\"verdict\":\"%s\",\"p_spray\":%.4f,"
    "\"confidence_margin\":%.4f,\"entropy\":%.2f,\"predicted_class\":\"%s\","
    "\"capture_seq\":%lu,\"timings\":{\"capture_dur_ms\":%lu,\"s1_eval_ms\":%lu,"
    "\"dsp_ms\":%lu,\"cnn_ms\":%lu,\"queue_wait_ms\":%lu,\"total_device_ms\":%lu,"
    "\"tx_start_ms\":%lu}}",
    eventId, alertType, verdict, p_spray,
    margin, entropy, predClass,
    captureSeq, captureDurMs, s1EvalMs,
    dspMs, cnnMs, queueWaitMs, deviceTotalMs,
    t_tx_start
  );

  alertClient.printf(
    "POST /esp32/alert HTTP/1.1\r\n"
    "Host: %s:%d\r\n"
    "Connection: close\r\n"
    "Content-Type: application/json\r\n"
    "X-Device-Token: %s\r\n"
    "Content-Length: %d\r\n\r\n",
    SERVER_HOST, SERVER_PORT, API_TOKEN, jsonLen
  );
  alertClient.write((const uint8_t*)jsonBuf, jsonLen);

  unsigned long tResp = millis();
  while (alertClient.connected() && !alertClient.available() && (millis() - tResp < 800)) {
    delay(1);
  }

  int httpCode = -1;
  if (alertClient.available()) {
    String statusLine = alertClient.readStringUntil('\n');
    statusLine.trim();
    int spaceIdx = statusLine.indexOf(' ');
    if (spaceIdx > 0) {
      httpCode = statusLine.substring(spaceIdx + 1, spaceIdx + 4).toInt();
    }
  }
  alertClient.stop();
  unsigned long netTxMs = millis() - t_tx_start;

  Serial.printf(
    "\n🚨 >>> [ALERT-DISPATCH] %s ALERT SENT (EvtId: %s) | HTTP %d in %lu ms [Capture: %lu ms, S1: %lu ms, DSP: %lu ms, CNN: %lu ms, NetTx: %lu ms] <<<\n",
    alertType, eventId, httpCode, deviceTotalMs + netTxMs, captureDurMs, s1EvalMs, dspMs, cnnMs, netTxMs
  );

  return (httpCode >= 200 && httpCode < 300);
}

/**
 * Transmits audio chunk directly over TCP with stage-by-stage timing instrumentation.
 * Uses Connection: close and TCP_NODELAY to prevent socket exhaustion and Nagle delays.
 */
void uploadAudioChunk(const AudioChunk& chunk, const EdgePredictionResult* edgeRes = nullptr, bool edgeEvaluated = false) {
  uploadAttemptCounter++;
  unsigned long uploadAttemptSeq = uploadAttemptCounter;
  unsigned long uploadStart = millis();

  unsigned long captureDurMs = (chunk.captureEnd >= chunk.captureStart) ? (chunk.captureEnd - chunk.captureStart) : 1000;
  unsigned long queueWaitMs = (uploadStart >= chunk.enqueueTime) ? (uploadStart - chunk.enqueueTime) : 0;

  // 1. Connection setup
  unsigned long tConnStart = millis();
  WiFiClient client;
  client.setTimeout((HTTP_TIMEOUT_MS + 999) / 1000);

  if (!client.connect(SERVER_HOST, SERVER_PORT, HTTP_TIMEOUT_MS)) {
    unsigned long connSetupMs = millis() - tConnStart;
    consecutiveUploadFailures++;
    Serial.printf("[AUDIO-UPLOAD] Connect failed after %lu ms to %s:%d (fails: %d)\n",
                  connSetupMs, SERVER_HOST, SERVER_PORT, consecutiveUploadFailures);
    if (consecutiveUploadFailures >= 3) checkBackendCommand();
    if (consecutiveUploadFailures >= 8) {
      Serial.println("[AUDIO-UPLOAD] Excessive network errors. Pausing recording.");
      recordingEnabled = false;
      sendStateAck("STOPPED");
    }
    return;
  }
  unsigned long tConnEnd = millis();
  unsigned long connSetupMs = tConnEnd - tConnStart;

  // 2. Request header transmission with TCP_NODELAY & Connection: close
  client.setNoDelay(true);

  String headers = "POST /esp32/test-audio HTTP/1.1\r\n";
  headers += "Host: " + String(SERVER_HOST) + ":" + String(SERVER_PORT) + "\r\n";
  headers += "User-Agent: ESP32-SprayDetector/2.0\r\n";
  headers += "Connection: close\r\n";
  headers += "Content-Type: audio/wav\r\n";
  headers += "Content-Length: " + String(chunk.size) + "\r\n";
  headers += "X-Device-Token: " + String(API_TOKEN) + "\r\n";
  headers += "X-Device-State: LISTENING\r\n";
  headers += "X-Capture-Start-Ms: " + String(chunk.captureStart) + "\r\n";
  headers += "X-Capture-End-Ms: " + String(chunk.captureEnd) + "\r\n";
  headers += "X-Tx-Start-Ms: " + String(millis()) + "\r\n";
  headers += "X-Capture-Seq: " + String(chunk.captureSeq) + "\r\n";
  headers += "X-Upload-Seq: " + String(uploadAttemptSeq) + "\r\n";

  if (edgeEvaluated && edgeRes != nullptr) {
    headers += "X-Edge-Evaluated: true\r\n";
    headers += "X-Edge-Verdict: " + String(edgeRes->verdict) + "\r\n";
    headers += "X-Edge-P-Spray: " + String(edgeRes->p_spray, 4) + "\r\n";
    headers += "X-Edge-Class: " + String(edge_detector_get_class_name(edgeRes->predicted_class)) + "\r\n";
    headers += "X-Edge-Infer-Ms: " + String(edgeRes->inference_time_ms) + "\r\n";
    headers += "X-Edge-Prep-Ms: " + String(edgeRes->preprocess_time_ms) + "\r\n";
  }
  headers += "\r\n";

  client.print(headers);

  // 3. Request-body transmission (direct high-speed socket stream)
  unsigned long tBodyStart = millis();
  size_t bytesSent = client.write(chunk.data, chunk.size);
  unsigned long tBodyEnd = millis();
  unsigned long bodyTxMs = tBodyEnd - tBodyStart;

  if (bytesSent < chunk.size) {
    consecutiveUploadFailures++;
    Serial.printf("[AUDIO-UPLOAD] Body send failed: only %u / %u bytes sent (Conn: %lu ms, Tx: %lu ms)\n",
                  (unsigned int)bytesSent, (unsigned int)chunk.size, connSetupMs, bodyTxMs);
    client.stop();
    return;
  }

  // 4. Response wait
  unsigned long tRespWaitStart = millis();
  while (client.connected() && !client.available() && (millis() - tRespWaitStart < HTTP_TIMEOUT_MS)) {
    delay(1);
  }
  unsigned long tRespWaitEnd = millis();
  unsigned long respWaitMs = tRespWaitEnd - tRespWaitStart;

  int httpCode = -1;
  String responseBody = "";

  if (client.available()) {
    consecutiveUploadFailures = 0;
    String statusLine = client.readStringUntil('\n');
    statusLine.trim();
    int spaceIdx = statusLine.indexOf(' ');
    if (spaceIdx > 0) {
      httpCode = statusLine.substring(spaceIdx + 1, spaceIdx + 4).toInt();
    }

    bool inBody = false;
    unsigned long readBodyStart = millis();
    while (client.connected() || client.available()) {
      if (millis() - readBodyStart > 1000) break;
      String line = client.readStringUntil('\n');
      if (!inBody && (line == "\r" || line.length() == 0)) {
        inBody = true;
        continue;
      }
      if (inBody) {
        responseBody += line + "\n";
      }
    }
  } else {
    consecutiveUploadFailures++;
    Serial.printf("[AUDIO-UPLOAD] Response timeout after %lu ms\n", respWaitMs);
  }

  // Clean close socket immediately
  client.stop();
  unsigned long totalUploadMs = millis() - uploadStart;

  // Extract server sequence and command from response JSON
  long serverSeq = -1;
  int seqIdx = responseBody.indexOf("\"chunk_seq\":");
  if (seqIdx < 0) seqIdx = responseBody.indexOf("\"chunk_seq\": ");
  if (seqIdx >= 0) {
    int startIdx = responseBody.indexOf(':', seqIdx) + 1;
    serverSeq = responseBody.substring(startIdx, responseBody.indexOf(',', startIdx)).toInt();
  }

  Serial.printf("[AUDIO-UPLOAD] CapSeq=#%lu, UpSeq=#%lu, SrvSeq=#%ld | HTTP %d (%lu ms) [%u bytes, Cap: %lu ms, QWait: %lu ms, Conn: %lu ms, BodyTx: %lu ms, RespWait: %lu ms]\n",
                chunk.captureSeq, uploadAttemptSeq, serverSeq, httpCode, totalUploadMs,
                (unsigned int)chunk.size, captureDurMs, queueWaitMs, connSetupMs, bodyTxMs, respWaitMs);

  // Performance warning if upload exceeded 1.0-second real-time window
  if (totalUploadMs > 1200) {
    Serial.printf("[PERF-WARNING] Upload duration %lu ms exceeds 1.0s window! (Conn: %lu ms, BodyTx: %lu ms, RespWait: %lu ms)\n",
                  totalUploadMs, connSetupMs, bodyTxMs, respWaitMs);
  }

  // Inspect server response for STOP command
  if (responseBody.indexOf("\"command\":\"STOP\"") >= 0 || responseBody.indexOf("\"command\": \"STOP\"") >= 0) {
    Serial.println("\n[COMMAND-POLL] >>> STOP command received in upload response -> Halting capture & upload <<<");
    recordingEnabled = false;
    sendStateAck("STOPPED");
  }
}

/**
 * Decoupled background task: transmits queued audio chunks without blocking detection.
 */
void backgroundUploadTask(void* parameter) {
  while (true) {
    AudioChunk chunk;
    if (xQueueReceive(uploadQueue, &chunk, pdMS_TO_TICKS(100)) == pdPASS) {
      if (recordingEnabled && WiFi.status() == WL_CONNECTED && chunk.data != nullptr) {
        uploadAudioChunk(chunk, chunk.edgeEvaluated ? &chunk.edgeResult : nullptr, chunk.edgeEvaluated);
      }
      if (chunk.data != nullptr) {
        free(chunk.data);
        chunk.data = nullptr;
      }
    }
  }
}

// =========================================================================
// Main Loop (Core 0: Network Transmission, Command Polling & Fallback Control)
// =========================================================================

void loop() {
  // Check optional serial override commands
  checkSerialCommands();

  // Handle manual test timeout if active
  if (manualTestActive) {
    if (millis() - manualTestStartTime >= MANUAL_TEST_DURATION_MS) {
      Serial.printf("\n[MANUAL-TEST] %lu ms test duration completed. Halting capture.\n", MANUAL_TEST_DURATION_MS);
      recordingEnabled = false;
      manualTestActive = false;
      sendStateAck("STOPPED");
    }
  }

  // If currently idle (stopped), poll the backend for a START command
  if (!recordingEnabled) {
    // Purge any stale audio chunks from queue immediately
    purgeAudioQueue();

    unsigned long now = millis();
    if (now - lastCommandPollTime >= COMMAND_POLL_INTERVAL_MS) {
      lastCommandPollTime = now;
      checkBackendCommand();
    }
    delay(20);
    return;
  }

  // If Wi-Fi is lost while recording, pause and reconnect (unless in standalone diagnostic or manual test mode)
  if (!ensureWiFiConnected()) {
    if (!manualTestActive && !diagnosticMode) {
      delay(200);
      return;
    }
  }

  // Dequeue newest audio chunk from Core 1
  AudioChunk chunk;
  chunk.data = nullptr;
  if (xQueueReceive(audioQueue, &chunk, pdMS_TO_TICKS(100)) == pdPASS) {
    if (!recordingEnabled) {
      if (chunk.data != nullptr) {
        free(chunk.data);
        chunk.data = nullptr;
      }
      return;
    }

    unsigned long dequeueTime = millis();
    unsigned long queueWaitMs = (dequeueTime >= chunk.enqueueTime) ? (dequeueTime - chunk.enqueueTime) : 0;
    unsigned long captureDurMs = (chunk.captureEnd >= chunk.captureStart) ? (chunk.captureEnd - chunk.captureStart) : 1000;

    // -------------------------------------------------------------
    // On-Device Edge Neural Network Inference (Stage 1 + Stage 2 CNN)
    // -------------------------------------------------------------
    EdgePredictionResult edgeResult = {};
    bool edgeEvaluated = false;
    unsigned long s1EvalMs = 0;
    unsigned long dspMs = 0;

    if (edge_detector_is_ready() && chunk.data != nullptr && chunk.size >= 44 + (AUDIO_DSP_AUDIO_SAMPLES * sizeof(int16_t))) {
      const int16_t* pcm16 = (const int16_t*)(chunk.data + 44);

      // Allocate temporary PCM float buffer in PSRAM
      float* pcm_float = (float*)(psramFound() ? ps_malloc(AUDIO_DSP_AUDIO_SAMPLES * sizeof(float)) : malloc(AUDIO_DSP_AUDIO_SAMPLES * sizeof(float)));
      if (pcm_float != nullptr) {
        audio_dsp_pcm16_to_float(pcm16, pcm_float, AUDIO_DSP_AUDIO_SAMPLES);

        // Stage 1: Fast kinematic pre-filter (<0.5 ms)
        unsigned long t_s1_start = millis();
        Stage1Result s1 = stage1_analyze_window(pcm_float, AUDIO_DSP_AUDIO_SAMPLES);
        unsigned long t_s1_end = millis();
        s1EvalMs = (t_s1_end >= t_s1_start) ? (t_s1_end - t_s1_start) : 0;

        if (s1.is_candidate) {
          // Candidate detected: emit immediate preliminary warning (< 1 ms after window capture)
          sendEdgeAlert("PRELIMINARY", chunk.captureSeq, captureDurMs, s1EvalMs, 0, 0, queueWaitMs, nullptr);

          // Run Log-Mel preprocessing + Stage 2 CNN inference
          float* log_mel = (float*)(psramFound() ? ps_malloc(AUDIO_DSP_SPECTROGRAM_SIZE * sizeof(float)) : malloc(AUDIO_DSP_SPECTROGRAM_SIZE * sizeof(float)));
          if (log_mel != nullptr) {
            unsigned long t_prep_start = millis();
            audio_dsp_compute_log_mel(pcm_float, log_mel);
            unsigned long t_prep_end = millis();
            dspMs = (t_prep_end >= t_prep_start) ? (t_prep_end - t_prep_start) : 0;

            if (edge_detector_run_inference(log_mel, &edgeResult)) {
              edgeEvaluated = true;
              edgeResult.preprocess_time_ms = dspMs;

              // Print comprehensive on-device prediction telemetry
              const char* predClassName = edge_detector_get_class_name(edgeResult.predicted_class);
              Serial.println("\n----------------- [ON-DEVICE CNN INFERENCE] -----------------");
              Serial.printf("  Verdict:            %s%s%s\n",
                            edgeResult.alert_confirmed ? ">>> " : "",
                            edgeResult.verdict,
                            edgeResult.alert_confirmed ? " <<<" : "");
              Serial.printf("  Spray Shaking Prob: %.4f (Threshold: %.2f) -> %s\n",
                            edgeResult.p_spray, EDGE_DETECTOR_CONSERVATIVE_THRESHOLD,
                            edgeResult.p_spray >= EDGE_DETECTOR_CONSERVATIVE_THRESHOLD ? "MET" : "NOT MET");
              Serial.printf("  Top Predicted:      %s (%.2f%%)\n",
                            predClassName, edgeResult.probs[edgeResult.predicted_class] * 100.0f);
              Serial.printf("  Confidence Margin:  %+.4f (Min: +%.2f)\n",
                            edgeResult.confidence_margin, EDGE_DETECTOR_MIN_MARGIN);
              Serial.printf("  Entropy / Competing: Entropy=%.2f (Max: %.2f) | %s (%.2f%%)\n",
                            edgeResult.entropy, EDGE_DETECTOR_MAX_ENTROPY,
                            edge_detector_get_class_name(edgeResult.max_competing_class), edgeResult.max_competing_prob * 100.0f);
              Serial.printf("  Edge Latency:       Total: %lu ms [S1: %lu ms, DSP: %lu ms, CNN: %lu ms]\n",
                            s1EvalMs + dspMs + edgeResult.inference_time_ms,
                            s1EvalMs, dspMs, edgeResult.inference_time_ms);
              Serial.println("-------------------------------------------------------------");

              // Immediately dispatch confirmed alert or rejection (< 20 ms network transit)
              if (edgeResult.alert_confirmed) {
                sendEdgeAlert("CONFIRMED", chunk.captureSeq, captureDurMs, s1EvalMs, dspMs, edgeResult.inference_time_ms, queueWaitMs, &edgeResult);
              } else {
                sendEdgeAlert("REJECTED", chunk.captureSeq, captureDurMs, s1EvalMs, dspMs, edgeResult.inference_time_ms, queueWaitMs, &edgeResult);
              }
            }
            free(log_mel);
          }
        } else {
          // Non-candidate: Stage 1 bypassed the CNN, keeping latency < 1 ms
          if (diagnosticMode) {
            Serial.printf("[STAGE1] Non-candidate: %s (RMS: %.4f, Bursts: %d, Contrast: %.1f dB) -> CNN Bypassed (%lu ms)\n",
                          s1.reason, s1.rms, s1.burst_count, s1.contrast_db, s1EvalMs);
          }
        }
        free(pcm_float);
      }
    }

    // Attach edge prediction to chunk for background upload
    chunk.edgeEvaluated = edgeEvaluated;
    if (edgeEvaluated) {
      chunk.edgeResult = edgeResult;
    }

    // Diagnostic & Manual Testing: calculate and display 1-second audio frame metrics on Core 0
    if (diagnosticMode || manualTestActive) {
      AudioStats stats = analyzeWavBuffer(chunk.data, chunk.size);
      printMicrophoneDiagnostics(stats, captureDurMs);
    }

    // Hand off audio chunk to decoupled background upload queue (non-blocking)
    if (uploadQueue != nullptr && WiFi.status() == WL_CONNECTED) {
      if (xQueueSend(uploadQueue, &chunk, 0) != pdPASS) {
        // Upload queue full: drop this audio chunk to prevent memory buildup
        if (chunk.data != nullptr) {
          free(chunk.data);
          chunk.data = nullptr;
        }
      }
    } else {
      if (chunk.data != nullptr) {
        free(chunk.data);
        chunk.data = nullptr;
      }
    }
  }
}
