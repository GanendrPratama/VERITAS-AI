/* VERITAS-AI sensor firmware -- ESP32 BLE GATT server.
 *
 * Publishes two channels that services/sensors.py connects to by device
 * name and reads by characteristic UUID (see docs/veritas-ai-design.md
 * Section 7, the `ble_*` config keys -- this firmware must match them, and
 * README.md has the wiring/BLE-profile spec table):
 *   - Heart rate: standard BLE SIG Heart Rate Service/Characteristic,
 *     uint8 format (flags byte 0x00, then BPM) -- from a MAX30102 over I2C.
 *   - GSR: custom notify characteristic, raw 12-bit ADC value as uint16 LE
 *     -- from a resistive GSR sensor's analog output.
 *
 * Two FreeRTOS tasks, connected by a queue, instead of one loop() doing
 * both jobs: SensorTask polls the MAX30102 as fast as beat-detection needs
 * (every ~2ms) and samples GSR every GSR_SAMPLE_MS, pushing each new
 * reading onto sampleQueue; BLETask blocks on that queue and turns each
 * reading into a notify() the moment it arrives. Splitting them means a
 * slow/blocking BLE stack call can never stall sensor polling (and vice
 * versa) -- they only communicate through the queue, never shared state.
 * Pinned to opposite cores (sensor: core 1, BLE: core 0) so they run
 * truly concurrently, not just interleaved by the scheduler.
 *
 * Library: SparkFun MAX3010x Pulse and Proximity Sensor Library (declared
 * in platformio.ini's lib_deps, PlatformIO fetches it automatically) for
 * the MAX30102 + its bundled beat-detection algorithm -- hand-rolling PPG
 * heart-rate detection is a solved problem, no reason to redo it. BLE
 * (BLEDevice.h etc.) and FreeRTOS (freertos/*.h) both ship with the
 * espressif32 platform's arduino framework, no extra lib_deps entry needed
 * for either.
 */
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>

#include <Wire.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <heartRate.h>
#include <MAX30105.h>

#define DEVICE_NAME "VERITAS-SENSOR"          // ble_device_name
#define HR_SERVICE_UUID BLEUUID((uint16_t)0x180D)   // ble_hr_service_uuid
#define HR_CHAR_UUID BLEUUID((uint16_t)0x2A37)       // ble_hr_char_uuid
#define GSR_SERVICE_UUID "bfe582f0-d884-43d4-aa9d-7648f8536e40"  // ble_gsr_service_uuid
#define GSR_CHAR_UUID "6b3a2c00-1e3d-4f6a-9c1e-2a1a2f3b4c5d"     // ble_gsr_char_uuid

#define GSR_PIN 34
#define GSR_SAMPLE_MS 200
#define SENSOR_POLL_MS 2  // MAX30105 beat detection wants frequent IR sampling

#define CHANNEL_HR 0
#define CHANNEL_GSR 1

typedef struct {
  uint8_t channel;  // CHANNEL_HR or CHANNEL_GSR
  uint16_t value;   // BPM for HR, raw 12-bit ADC for GSR
} SensorSample;

MAX30105 particleSensor;
BLECharacteristic *hrCharacteristic;
BLECharacteristic *gsrCharacteristic;
QueueHandle_t sampleQueue;
volatile bool deviceConnected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override { deviceConnected = true; }
  void onDisconnect(BLEServer *server) override {
    deviceConnected = false;
    server->getAdvertising()->start();  // resume advertising after a drop
  }
};

// -- SensorTask: reads hardware, never touches BLE --

void sensorTask(void *pvParameters) {
  const byte RATE_SIZE = 4;  // rolling average window for BPM
  byte rates[RATE_SIZE] = {0};
  byte rateSpot = 0;
  long lastBeat = 0;
  int beatAvg = 0;
  TickType_t lastGsrTick = xTaskGetTickCount();

  for (;;) {
    long irValue = particleSensor.getIR();
    if (checkForBeat(irValue)) {
      long delta = millis() - lastBeat;
      lastBeat = millis();
      float bpm = 60.0 / (delta / 1000.0);
      if (bpm > 20 && bpm < 255) {
        rates[rateSpot++] = (byte)bpm;
        rateSpot %= RATE_SIZE;
        int total = 0;
        for (byte i = 0; i < RATE_SIZE; i++) total += rates[i];
        beatAvg = total / RATE_SIZE;
      }

      // ponytail: 50000 is SparkFun's stock "finger present" IR threshold,
      // not calibrated against this specific sensor/skin -- tune against
      // real readings if BPM stops updating with a finger on the sensor.
      if (irValue > 50000 && beatAvg > 0) {
        SensorSample sample = {CHANNEL_HR, (uint16_t)beatAvg};
        xQueueSend(sampleQueue, &sample, 0);  // drop rather than block; BLETask easily keeps up
      }
    }

    if (xTaskGetTickCount() - lastGsrTick >= pdMS_TO_TICKS(GSR_SAMPLE_MS)) {
      lastGsrTick = xTaskGetTickCount();
      uint16_t raw = analogRead(GSR_PIN);  // ESP32 ADC1: 0-4095 (12-bit)
      SensorSample sample = {CHANNEL_GSR, raw};
      xQueueSend(sampleQueue, &sample, 0);
    }

    vTaskDelay(pdMS_TO_TICKS(SENSOR_POLL_MS));
  }
}

// -- BLETask: only ever consumes the queue and calls notify() --

void bleTask(void *pvParameters) {
  SensorSample sample;
  for (;;) {
    if (xQueueReceive(sampleQueue, &sample, portMAX_DELAY) != pdTRUE) continue;
    if (!deviceConnected) continue;

    if (sample.channel == CHANNEL_HR) {
      uint8_t payload[2] = {0x00, (uint8_t)sample.value};  // flags=0x00 (uint8 format), then BPM
      hrCharacteristic->setValue(payload, 2);
      hrCharacteristic->notify();
    } else {
      uint8_t payload[2] = {(uint8_t)(sample.value & 0xFF), (uint8_t)(sample.value >> 8)};  // LE
      gsrCharacteristic->setValue(payload, 2);
      gsrCharacteristic->notify();
    }
  }
}

void setup() {
  Serial.begin(115200);

  Wire.begin();
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found -- check wiring");
  } else {
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
  }

  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *hrService = server->createService(HR_SERVICE_UUID);
  hrCharacteristic =
      hrService->createCharacteristic(HR_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  hrCharacteristic->addDescriptor(new BLE2902());
  hrService->start();

  BLEService *gsrService = server->createService(GSR_SERVICE_UUID);
  gsrCharacteristic =
      gsrService->createCharacteristic(GSR_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  gsrCharacteristic->addDescriptor(new BLE2902());
  gsrService->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(HR_SERVICE_UUID);
  advertising->addServiceUUID(GSR_SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  sampleQueue = xQueueCreate(16, sizeof(SensorSample));
  xTaskCreatePinnedToCore(sensorTask, "SensorTask", 4096, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(bleTask, "BLETask", 4096, NULL, 1, NULL, 0);

  Serial.println("Advertising as " DEVICE_NAME);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));  // all real work happens in the two FreeRTOS tasks above
}
