#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <DFRobot_BMI160.h>

DFRobot_BMI160 bmi160;
const int8_t I2C_ADDRESS = 0x68;

const char* DEVICE_NAME = "ESP32_IMU_TEST";
const char* SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
const char* TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";
const char* RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";
const size_t NOTIFICATION_CHUNK_SIZE = 20;
const uint16_t MIN_RATE_HZ = 1;
const uint16_t MAX_RATE_HZ = 200;

BLECharacteristic* txCharacteristic = nullptr;
volatile bool deviceConnected = false;
volatile uint32_t sampleIntervalUs = 50000;
uint16_t sampleRateHz = 20;

uint32_t sequenceNumber = 0;
uint32_t sensorReadCount = 0;
uint32_t sensorReadErrors = 0;
uint32_t transmittedRecords = 0;
uint32_t notificationChunks = 0;
uint32_t schedulerSkips = 0;
uint32_t lastReportMs = 0;
uint32_t lastSampleUs = 0;
bool haveLastSample = false;
constexpr float GYRO_LSB_TO_RAD_PER_SEC =
    (2000.0f / 32768.0f) * (3.14159265f / 180.0f);

void sendMessage(const char* message, size_t length) {
  for (size_t offset = 0; offset < length; offset += NOTIFICATION_CHUNK_SIZE) {
    const size_t chunkLength = min(NOTIFICATION_CHUNK_SIZE, length - offset);
    txCharacteristic->setValue(
      reinterpret_cast<const uint8_t*>(message + offset),
      chunkLength
    );
    txCharacteristic->notify();
    notificationChunks++;
  }
}

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer*) override {
    deviceConnected = true;
    Serial.println("EVENT,connected");
  }

  void onDisconnect(BLEServer* server) override {
    deviceConnected = false;
    Serial.println("EVENT,disconnected");
    server->startAdvertising();
  }
};

class CommandCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) override {
    String command = characteristic->getValue();
    command.trim();
    if (!command.startsWith("RATE,")) {
      Serial.print("EVENT,unknown_command,");
      Serial.println(command);
      return;
    }

    const long requestedRate = command.substring(5).toInt();
    if (requestedRate < MIN_RATE_HZ || requestedRate > MAX_RATE_HZ) {
      Serial.print("EVENT,invalid_rate,");
      Serial.println(command);
      return;
    }

    sampleRateHz = static_cast<uint16_t>(requestedRate);
    sampleIntervalUs = 1000000UL / sampleRateHz;
    Serial.print("EVENT,rate_hz,");
    Serial.println(sampleRateHz);
  }
};

void setup() {
  Serial.begin(115200);
  delay(100);

  Wire.begin();
  Wire.setClock(400000);

  if (bmi160.softReset() != BMI160_OK) {
    Serial.println("FATAL,BMI160 reset failed");
    while (true) delay(1000);
  }
  if (bmi160.I2cInit(I2C_ADDRESS) != BMI160_OK) {
    Serial.println("FATAL,BMI160 init failed");
    while (true) delay(1000);
  }
  Serial.println("EVENT,BMI160 ready");

  BLEDevice::init(DEVICE_NAME);
  BLEServer* server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService* service = server->createService(SERVICE_UUID);
  txCharacteristic = service->createCharacteristic(
    TX_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  txCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic* rxCharacteristic = service->createCharacteristic(
    RX_UUID,
    BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
  );
  rxCharacteristic->setCallbacks(new CommandCallbacks());

  service->start();
  server->getAdvertising()->start();

  Serial.print("EVENT,ready,");
  Serial.print(DEVICE_NAME);
  Serial.print(",rate_hz,");
  Serial.println(sampleRateHz);
  lastSampleUs = micros();
  lastReportMs = millis();
}

void loop() {
  const uint32_t nowUs = micros();
  const uint32_t intervalUs = sampleIntervalUs;

  if (static_cast<int32_t>(nowUs - lastSampleUs) >= static_cast<int32_t>(intervalUs)) {
    if (haveLastSample) {
      const uint32_t elapsedUs = nowUs - lastSampleUs;
      if (elapsedUs >= intervalUs * 2) {
        schedulerSkips += (elapsedUs / intervalUs) - 1;
      }
    }
    lastSampleUs = nowUs;
    haveLastSample = true;

    int16_t accelGyro[6] = {0};
    if (bmi160.getAccelGyroData(accelGyro) != BMI160_OK) {
      sensorReadErrors++;
    } else {
      sensorReadCount++;
      if (deviceConnected) {
        char record[128];
        const int recordLength = snprintf(
          record,
          sizeof(record),
          "%lu,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
          static_cast<unsigned long>(sequenceNumber),
          static_cast<unsigned long>(millis()),
          accelGyro[3] / 16384.0f,
          accelGyro[4] / 16384.0f,
          accelGyro[5] / 16384.0f,
          accelGyro[0] * GYRO_LSB_TO_RAD_PER_SEC,
          accelGyro[1] * GYRO_LSB_TO_RAD_PER_SEC,
          accelGyro[2] * GYRO_LSB_TO_RAD_PER_SEC
        );
        if (recordLength > 0 && recordLength < sizeof(record)) {
          sendMessage(record, static_cast<size_t>(recordLength));
          transmittedRecords++;
          sequenceNumber++;
        } else {
          Serial.println("ERROR,record_format_failed");
        }
      }
    }
  }

  const uint32_t nowMs = millis();
  if (nowMs - lastReportMs >= 5000) {
    const uint32_t elapsedMs = nowMs - lastReportMs;
    Serial.print("STATS,elapsed_ms,");
    Serial.print(elapsedMs);
    Serial.print(",rate_hz,");
    Serial.print(sampleRateHz);
    Serial.print(",sensor_reads,");
    Serial.print(sensorReadCount);
    Serial.print(",sensor_errors,");
    Serial.print(sensorReadErrors);
    Serial.print(",transmitted_records,");
    Serial.print(transmittedRecords);
    Serial.print(",notify_chunks,");
    Serial.print(notificationChunks);
    Serial.print(",scheduler_skips,");
    Serial.print(schedulerSkips);
    Serial.print(",connected,");
    Serial.println(deviceConnected ? 1 : 0);

    sensorReadCount = 0;
    sensorReadErrors = 0;
    transmittedRecords = 0;
    notificationChunks = 0;
    schedulerSkips = 0;
    lastReportMs = nowMs;
  }
}
