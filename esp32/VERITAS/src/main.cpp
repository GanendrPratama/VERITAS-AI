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

#define DEVICE_NAME "VERITAS-SENSOR"          
#define HR_SERVICE_UUID BLEUUID((uint16_t)0x180D)   
#define HR_CHAR_UUID BLEUUID((uint16_t)0x2A37)       

// ESP32-C3 SuperMini I2C Pins
#define SUPERMINI_SDA 5
#define SUPERMINI_SCL 6

#define SENSOR_POLL_MS 2  

typedef struct {
  uint16_t bpm;   
} HeartRateSample;

MAX30105 particleSensor;
BLECharacteristic *hrCharacteristic;
QueueHandle_t sampleQueue;
volatile bool deviceConnected = false;
bool sensorFound = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override { 
    deviceConnected = true; 
    Serial.println("[BLE] Client CONNECTED!");
  }
  void onDisconnect(BLEServer *server) override {
    deviceConnected = false;
    Serial.println("[BLE] Client DISCONNECTED! Restarting advertising...");
    server->getAdvertising()->start();  
  }
};

// -- SensorTask: Reads hardware --
void sensorTask(void *pvParameters) {
  Serial.println("[TASK] SensorTask initialized and running.");
  
  const byte RATE_SIZE = 4;  
  byte rates[RATE_SIZE] = {0};
  byte rateSpot = 0;
  long lastBeat = 0;
  int beatAvg = 0;

  for (;;) {
    if (!sensorFound) {
      Serial.println("[TASK] SensorTask waiting... (MAX30102 not detected)");
      vTaskDelay(pdMS_TO_TICKS(2000));
      continue;
    }

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

      if (irValue > 50000 && beatAvg > 0) {
        Serial.printf("[SENSOR] Beat detected! Raw IR: %ld | BPM: %d\n", irValue, beatAvg);
        HeartRateSample sample = {(uint16_t)beatAvg};
        xQueueSend(sampleQueue, &sample, 0);  
      }
    }

    vTaskDelay(pdMS_TO_TICKS(SENSOR_POLL_MS));
  }
}

// -- BLETask: Sends notifications --
void bleTask(void *pvParameters) {
  Serial.println("[TASK] BLETask initialized and waiting for data...");
  HeartRateSample sample;
  
  for (;;) {
    if (xQueueReceive(sampleQueue, &sample, portMAX_DELAY) != pdTRUE) continue;
    
    if (!deviceConnected) {
      Serial.println("[BLE] Sample queued, but no BLE client connected to send notify.");
      continue;
    }

    uint8_t payload[2] = {0x00, (uint8_t)sample.bpm};  
    hrCharacteristic->setValue(payload, 2);
    hrCharacteristic->notify();
    Serial.printf("[BLE] Sent Notify BPM: %d\n", sample.bpm);
  }
}

void setup() {
  // 1. Lower CPU frequency to keep SuperMini cool
  setCpuFrequencyMhz(80);

  // 2. Initialize Serial
  Serial.begin(115200);
  
  // CRITICAL FOR NATIVE USB CDC: Wait 2 seconds for PC USB driver to attach
  delay(2000); 

  Serial.println("\n--- [BOOT] VERITAS-AI ESP32-C3 SuperMini Starting ---");
  Serial.printf("[BOOT] CPU Frequency set to: %d MHz\n", getCpuFrequencyMhz());

  // 3. Initialize I2C Bus
  Serial.printf("[I2C] Initializing Wire on SDA=GPIO%d, SCL=GPIO%d...\n", SUPERMINI_SDA, SUPERMINI_SCL);
  Wire.begin(SUPERMINI_SDA, SUPERMINI_SCL);

  // 4. Test MAX30102 Sensor
  Serial.println("[I2C] Probing MAX30102 sensor...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[ERROR] MAX30102 NOT found on I2C bus! Check power & SDA/SCL wiring.");
    sensorFound = false;
  } else {
    Serial.println("[SUCCESS] MAX30102 detected!");
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
    sensorFound = true;
  }

  // 5. Initialize BLE Stack
  Serial.printf("[BLE] Initializing BLE stack as '%s'...\n", DEVICE_NAME);
  BLEDevice::init(DEVICE_NAME);
  
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  Serial.println("[BLE] Registering Heart Rate Service (0x180D)...");
  BLEService *hrService = server->createService(HR_SERVICE_UUID);
  hrCharacteristic = hrService->createCharacteristic(
      HR_CHAR_UUID, 
      BLECharacteristic::PROPERTY_NOTIFY
  );
  hrCharacteristic->addDescriptor(new BLE2902());
  hrService->start();

  Serial.println("[BLE] Starting BLE Advertising...");
  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(HR_SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  Serial.println("[SUCCESS] BLE Advertising active!");

  // 6. Setup FreeRTOS Queue & Tasks
  Serial.println("[RTOS] Creating sample queue...");
  sampleQueue = xQueueCreate(16, sizeof(HeartRateSample));

  Serial.println("[RTOS] Launching SensorTask and BLETask...");
  xTaskCreate(sensorTask, "SensorTask", 4096, NULL, 1, NULL);
  xTaskCreate(bleTask, "BLETask", 4096, NULL, 1, NULL);

  Serial.println("--- [BOOT] Setup complete! Board ready. ---\n");
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(5000));
  Serial.println("[STATUS] System loop heartbeat tick (Alive)");
}