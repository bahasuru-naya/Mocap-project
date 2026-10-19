#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <DFRobot_BMI160.h>
#include <math.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

/*
 * BMI160 + ESP32 BLE IMU node
 * Target sampling rate: 20 Hz (one sample every 50 ms).
 *
 * BLE packet (20 bytes, little-endian):
 *   [0..1]   'I', 'M'
 *   [2..3]   uint16 sequence number
 *   [4..7]   uint32 timestamp in milliseconds, derived from esp_timer_get_time()
 *   [8..13]  int16 ax, ay, az (g * 1000)
 *   [14..19] int16 gx, gy, gz (rad/s * 100)
 *
 * Calibration command: "CALIBRATE"
 * Calibration response: "CALIB_DONE"
 *
 */

DFRobot_BMI160 bmi160;
const int8_t I2C_ADDR = 0x68;

#define NODE_ID "ESP32_IMU_1"  // Change for each node
#define SERVICE_UUID            "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX  "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX  "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

// Sampling target: 20 Hz = 50 ms.
constexpr uint32_t SAMPLE_INTERVAL_US = 50000UL;
constexpr uint32_t SAMPLE_INTERVAL_MS = 50UL;

// DFRobot BMI160 default ranges assumed by these conversions:
// Accelerometer +/-2 g => 16384 LSB/g; gyroscope +/-2000 degrees/s.
constexpr float ACCEL_LSB_PER_G = 16384.0f;
constexpr float GYRO_LSB_TO_RAD_PER_SEC =
    (2000.0f / 32768.0f) * (3.14159265358979323846f / 180.0f);

// Sensor positive X arrow points into the table: expected stationary vector.
// Change EXPECTED_AX_G to +1.0f if your raw stationary X reading is positive.
constexpr float EXPECTED_AX_G = -1.0f;
constexpr float EXPECTED_AY_G =  0.0f;
constexpr float EXPECTED_AZ_G =  0.0f;

constexpr uint16_t CALIBRATION_COUNT = 100;

BLEServer* pServer = nullptr;
BLECharacteristic* pTxCharacteristic = nullptr;
volatile bool deviceConnected = false;
SemaphoreHandle_t notificationMutex = nullptr;

// Shared only for handing a request from the BLE callback to loop().
portMUX_TYPE calibrationRequestMux = portMUX_INITIALIZER_UNLOCKED;
volatile bool calibrationRequestPending = false;

// Calibration state is accessed/modified only from loop().
bool isCalibrating = false;
uint16_t calibrationSamples = 0;
float sumGx = 0.0f, sumGy = 0.0f, sumGz = 0.0f;
float sumAx = 0.0f, sumAy = 0.0f, sumAz = 0.0f;

float gxOffset = 0.0f, gyOffset = 0.0f, gzOffset = 0.0f;
float axOffset = 0.0f, ayOffset = 0.0f, azOffset = 0.0f;

uint16_t sampleSequence = 0;
int64_t lastSampleUs = 0;
uint32_t sampleCount = 0;
uint32_t missedSampleSlots = 0;
int64_t lastStatsUs = 0;
uint32_t samplesAtLastStats = 0;
uint32_t missedAtLastStats = 0;

static inline void writeUint16LE(uint8_t* buffer, size_t offset, uint16_t value) {
  buffer[offset] = static_cast<uint8_t>(value & 0xFF);
  buffer[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
}

static inline void writeUint32LE(uint8_t* buffer, size_t offset, uint32_t value) {
  buffer[offset] = static_cast<uint8_t>(value & 0xFF);
  buffer[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
  buffer[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFF);
  buffer[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFF);
}

static inline void writeInt16LE(uint8_t* buffer, size_t offset, int16_t value) {
  writeUint16LE(buffer, offset, static_cast<uint16_t>(value));
}

void notifyPacket(const uint8_t* packet, size_t length) {
  if (!deviceConnected || pTxCharacteristic == nullptr) return;

  if (notificationMutex != nullptr) {
    xSemaphoreTake(notificationMutex, portMAX_DELAY);
  }

  // Re-check after acquiring the mutex in case the client disconnected.
  if (deviceConnected) {
    pTxCharacteristic->setValue(const_cast<uint8_t*>(packet), length);
    pTxCharacteristic->notify();
  }

  if (notificationMutex != nullptr) {
    xSemaphoreGive(notificationMutex);
  }
}

bool takeCalibrationRequest() {
  bool requested;
  portENTER_CRITICAL(&calibrationRequestMux);
  requested = calibrationRequestPending;
  calibrationRequestPending = false;
  portEXIT_CRITICAL(&calibrationRequestMux);
  return requested;
}

void startCalibration() {
  // Called only by loop(), never by the BLE callback.
  isCalibrating = true;
  calibrationSamples = 0;
  sumGx = sumGy = sumGz = 0.0f;
  sumAx = sumAy = sumAz = 0.0f;
  Serial.println("Calibration started. Keep the sensor completely stationary.");
}

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    deviceConnected = true;
    Serial.println("BLE client connected");
  }

  void onDisconnect(BLEServer* server) override {
    deviceConnected = false;
    Serial.println("BLE client disconnected");
    server->startAdvertising();
  }
};

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) override {
    String rxValue = characteristic->getValue();
    if (rxValue.length() == 0) return;

    // Do not modify calibration counters/sums here. This callback runs in a
    // different task; it only posts a request for loop() to handle safely.
    if (rxValue.indexOf("CALIBRATE") >= 0) {
      portENTER_CRITICAL(&calibrationRequestMux);
      calibrationRequestPending = true;
      portEXIT_CRITICAL(&calibrationRequestMux);
    }
  }
};

void setup() {
  Serial.begin(115200);
  delay(150);

  notificationMutex = xSemaphoreCreateMutex();
  if (notificationMutex == nullptr) {
    Serial.println("ERROR: Could not create BLE notification mutex");
    while (true) delay(1000);
  }

  Wire.begin();
  Wire.setClock(400000);

  if (bmi160.softReset() != BMI160_OK) {
    Serial.println("ERROR: BMI160 reset failed");
    while (true) delay(1000);
  }

  if (bmi160.I2cInit(I2C_ADDR) != BMI160_OK) {
    Serial.println("ERROR: BMI160 initialization failed");
    while (true) delay(1000);
  }

  Serial.println("BMI160 initialized");
  Serial.println("Assumed ranges: accelerometer +/-2 g; gyroscope +/-2000 deg/s");
  Serial.println("Target sample rate: 20 Hz (50 ms interval)");

  BLEDevice::init(NODE_ID);
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* service = pServer->createService(SERVICE_UUID);

  pTxCharacteristic = service->createCharacteristic(
      CHARACTERISTIC_UUID_TX,
      BLECharacteristic::PROPERTY_NOTIFY);
  pTxCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic* pRxCharacteristic = service->createCharacteristic(
      CHARACTERISTIC_UUID_RX,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  pRxCharacteristic->setCallbacks(new MyCallbacks());

  service->start();
  pServer->getAdvertising()->start();

  Serial.print("BLE UART ready. Device name: ");
  Serial.println(NODE_ID);

  // Schedule the first sample one full interval after setup.
  lastSampleUs = esp_timer_get_time();
  lastStatsUs = lastSampleUs;
}

void loop() {
  // Handle calibration requests in the main loop to avoid callback races.
  if (takeCalibrationRequest()) {
    startCalibration();
  }

  const int64_t nowUs = esp_timer_get_time();
  if ((nowUs - lastSampleUs) < static_cast<int64_t>(SAMPLE_INTERVAL_US)) {
    delay(1);  // Yield to BLE/FreeRTOS tasks without busy-spinning.
    return;
  }

  // Do not rapidly take several readings to "catch up" after a delay: those
  // would not represent distinct 50 ms samples. Record late/missed slots.
  const int64_t elapsedUs = nowUs - lastSampleUs;
  if (elapsedUs >= (2LL * SAMPLE_INTERVAL_US)) {
    missedSampleSlots += static_cast<uint32_t>(elapsedUs / SAMPLE_INTERVAL_US) - 1UL;
  }
  lastSampleUs = nowUs;

  int16_t raw[6] = {0};
  if (bmi160.getAccelGyroData(raw) != BMI160_OK) {
    Serial.println("WARNING: BMI160 read error");
    return;
  }

  // DFRobot_BMI160 returns gyro X/Y/Z first, then accel X/Y/Z.
  float gx = static_cast<float>(raw[0]) * GYRO_LSB_TO_RAD_PER_SEC;
  float gy = static_cast<float>(raw[1]) * GYRO_LSB_TO_RAD_PER_SEC;
  float gz = static_cast<float>(raw[2]) * GYRO_LSB_TO_RAD_PER_SEC;
  float ax = static_cast<float>(raw[3]) / ACCEL_LSB_PER_G;
  float ay = static_cast<float>(raw[4]) / ACCEL_LSB_PER_G;
  float az = static_cast<float>(raw[5]) / ACCEL_LSB_PER_G;

  if (isCalibrating) {
    // Average stationary samples to estimate gyro bias and accel offset
    // relative to the expected gravity vector for the known orientation.
    sumGx += gx; sumGy += gy; sumGz += gz;
    sumAx += ax; sumAy += ay; sumAz += az;
    calibrationSamples++;

    if (calibrationSamples >= CALIBRATION_COUNT) {
      gxOffset = sumGx / static_cast<float>(CALIBRATION_COUNT);
      gyOffset = sumGy / static_cast<float>(CALIBRATION_COUNT);
      gzOffset = sumGz / static_cast<float>(CALIBRATION_COUNT);

      const float meanAx = sumAx / static_cast<float>(CALIBRATION_COUNT);
      const float meanAy = sumAy / static_cast<float>(CALIBRATION_COUNT);
      const float meanAz = sumAz / static_cast<float>(CALIBRATION_COUNT);
      axOffset = meanAx - EXPECTED_AX_G;
      ayOffset = meanAy - EXPECTED_AY_G;
      azOffset = meanAz - EXPECTED_AZ_G;

      isCalibrating = false;
      Serial.println("Calibration complete.");
      Serial.printf("Gyro offsets (rad/s): %.5f, %.5f, %.5f\n", gxOffset, gyOffset, gzOffset);
      Serial.printf("Accel offsets (g): %.5f, %.5f, %.5f\n", axOffset, ayOffset, azOffset);
      Serial.println("Expected stationary corrected acceleration: approximately (-1, 0, 0) g.");

      if (deviceConnected) {
        const char message[] = "CALIB_DONE";
        notifyPacket(reinterpret_cast<const uint8_t*>(message), sizeof(message) - 1);
      }
    }
    return;  // Do not transmit calibration samples as motion data.
  }

  // Remove gyro bias; remove accelerometer bias while preserving expected
  // gravity in the calibration orientation.
  gx -= gxOffset; gy -= gyOffset; gz -= gzOffset;
  ax -= axOffset; ay -= ayOffset; az -= azOffset;

  // Use the high-resolution ESP timer as the source; keep milliseconds in
  // the existing uint32 packet field so the current receiver remains compatible.
  const uint32_t timestampMs = static_cast<uint32_t>(esp_timer_get_time() / 1000ULL);
  sampleCount++;

  if (deviceConnected) {
    uint8_t packet[20];
    packet[0] = 'I';
    packet[1] = 'M';
    writeUint16LE(packet, 2, sampleSequence++);
    writeUint32LE(packet, 4, timestampMs);
    writeInt16LE(packet, 8, static_cast<int16_t>(lroundf(ax * 1000.0f)));
    writeInt16LE(packet, 10, static_cast<int16_t>(lroundf(ay * 1000.0f)));
    writeInt16LE(packet, 12, static_cast<int16_t>(lroundf(az * 1000.0f)));
    writeInt16LE(packet, 14, static_cast<int16_t>(lroundf(gx * 100.0f)));
    writeInt16LE(packet, 16, static_cast<int16_t>(lroundf(gy * 100.0f)));
    writeInt16LE(packet, 18, static_cast<int16_t>(lroundf(gz * 100.0f)));
    notifyPacket(packet, sizeof(packet));
  }

  // Report measured rate and late/missed slots once per second.
  const int64_t statsNowUs = esp_timer_get_time();
  const int64_t statsElapsedUs = statsNowUs - lastStatsUs;
  if (statsElapsedUs >= 1000000LL) {
    const uint32_t samplesThisPeriod = sampleCount - samplesAtLastStats;
    const uint32_t missedThisPeriod = missedSampleSlots - missedAtLastStats;
    const float measuredRate = static_cast<float>(samplesThisPeriod) * 1000000.0f /
                               static_cast<float>(statsElapsedUs);
    Serial.printf("Measured rate: %.1f samples/s | total: %lu | missed schedule slots this period: %lu\n",
                  measuredRate,
                  static_cast<unsigned long>(sampleCount),
                  static_cast<unsigned long>(missedThisPeriod));
    lastStatsUs = statsNowUs;
    samplesAtLastStats = sampleCount;
    missedAtLastStats = missedSampleSlots;
  }
}
