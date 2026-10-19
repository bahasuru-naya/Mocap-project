#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <DFRobot_BMI160.h>
#include <math.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

DFRobot_BMI160 bmi160;
const int8_t i2c_addr = 0x68;

BLEServer* pServer = NULL;
BLECharacteristic* pTxCharacteristic = NULL;
bool deviceConnected = false;
SemaphoreHandle_t notificationMutex = NULL;

// CHANGE THIS BEFORE UPLOADING TO EACH SENSOR (e.g., ESP32_IMU_1, ESP32_IMU_2, ...)
#define NODE_ID "ESP32_IMU_3" 

#define SERVICE_UUID            "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX  "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX  "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
uint16_t sample_sequence = 0;

// Calibration offsets
float gx_offset = 0, gy_offset = 0, gz_offset = 0;
float ax_offset = 0, ay_offset = 0, az_offset = 0;
constexpr float GYRO_LSB_TO_RAD_PER_SEC =
    (2000.0f / 32768.0f) * (3.14159265f / 180.0f);
volatile bool is_calibrating = false;
int calibration_samples = 0;
const int CALIBRATION_COUNT = 100;
float sum_gx = 0, sum_gy = 0, sum_gz = 0;
float sum_ax = 0, sum_ay = 0, sum_az = 0;

// --- Timer Interrupt Variables ---
hw_timer_t * timer = NULL;
volatile bool read_data = false;

// The Interrupt Service Routine (ISR)
void IRAM_ATTR onTimer() {
  read_data = true;
}

void writeUint16LE(uint8_t* buffer, size_t offset, uint16_t value) {
  buffer[offset] = value & 0xFF;
  buffer[offset + 1] = (value >> 8) & 0xFF;
}

void writeUint32LE(uint8_t* buffer, size_t offset, uint32_t value) {
  buffer[offset] = value & 0xFF;
  buffer[offset + 1] = (value >> 8) & 0xFF;
  buffer[offset + 2] = (value >> 16) & 0xFF;
  buffer[offset + 3] = (value >> 24) & 0xFF;
}

void writeInt16LE(uint8_t* buffer, size_t offset, int16_t value) {
  writeUint16LE(buffer, offset, static_cast<uint16_t>(value));
}

void notifyPacket(const uint8_t* packet, size_t length) {
  if (notificationMutex) {
    xSemaphoreTake(notificationMutex, portMAX_DELAY);
  }
  pTxCharacteristic->setValue(packet, length);
  pTxCharacteristic->notify();
  if (notificationMutex) {
    xSemaphoreGive(notificationMutex);
  }
}

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("BLE Client Connected");
  }

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("BLE Client Disconnected");
    pServer->startAdvertising();
  }
};

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String rxValue = pCharacteristic->getValue();

    if (rxValue.length() > 0) {
      Serial.print("Received Command: ");
      Serial.println(rxValue);
      if (rxValue.indexOf("CALIBRATE") >= 0) {
        Serial.println("Starting calibration...");
        is_calibrating = true;
        calibration_samples = 0;
        sum_gx = 0; sum_gy = 0; sum_gz = 0;
        sum_ax = 0; sum_ay = 0; sum_az = 0;
      }
    }
  }
};

void setup() {
  Serial.begin(115200);
  delay(100);
  notificationMutex = xSemaphoreCreateMutex();

  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C for fast reading

  if (bmi160.softReset() != BMI160_OK) {
    Serial.println("BMI160 reset failed!");
    while (1);
  }

  if (bmi160.I2cInit(i2c_addr) != BMI160_OK) {
    Serial.println("BMI160 init failed!");
    while (1);
  }

  Serial.println("BMI160 Ready");

  BLEDevice::init(NODE_ID);
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pTxCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_TX,
                        BLECharacteristic::PROPERTY_NOTIFY
                      );
  pTxCharacteristic->addDescriptor(new BLE2902());

  // Added PROPERTY_WRITE_NR to support "Write Without Response". 
  // This helps prevent BLE bottlenecks and "Unreachable" errors.
  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
                                           CHARACTERISTIC_UUID_RX,
                                           BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
                                         );
  pRxCharacteristic->setCallbacks(new MyCallbacks());

  pService->start();
  pServer->getAdvertising()->start();

  Serial.println("BLE UART Ready");
  Serial.print("Device Name: ");
  Serial.println(NODE_ID);

  // --- Setup Hardware Timer Interrupt ---
  // ESP32 Arduino Core 3.x Timer API
  timer = timerBegin(1000000); // 1MHz frequency (1 tick = 1 microsecond)
  timerAttachInterrupt(timer, &onTimer); // attach ISR
  timerAlarm(timer, 20000, true, 0); // alarm every 20,000 ticks (20ms), autoreload
}

void loop() {
  // Only execute when the timer interrupt flag is set
  if (read_data) {
    read_data = false; // Reset the flag
    
    int16_t accelGyro[6] = {0};

    if (bmi160.getAccelGyroData(accelGyro) == BMI160_OK) {
      // Convert to meaningful units
      float gx = accelGyro[0] * GYRO_LSB_TO_RAD_PER_SEC;
      float gy = accelGyro[1] * GYRO_LSB_TO_RAD_PER_SEC;
      float gz = accelGyro[2] * GYRO_LSB_TO_RAD_PER_SEC;

      float ax = accelGyro[3] / 16384.0;
      float ay = accelGyro[4] / 16384.0;
      float az = accelGyro[5] / 16384.0;

      if (is_calibrating) {
        sum_gx += gx; sum_gy += gy; sum_gz += gz;
        sum_ax += ax; sum_ay += ay; sum_az += az;
        calibration_samples++;
        
        if (calibration_samples >= CALIBRATION_COUNT) {
          gx_offset = sum_gx / CALIBRATION_COUNT;
          gy_offset = sum_gy / CALIBRATION_COUNT;
          gz_offset = sum_gz / CALIBRATION_COUNT;
          ax_offset = sum_ax / CALIBRATION_COUNT;
          ay_offset = sum_ay / CALIBRATION_COUNT;
          az_offset = sum_az / CALIBRATION_COUNT;
          is_calibrating = false;
          Serial.println("Calibration complete.");
          if (deviceConnected) {
              const char calib_msg[] = "CALIB_DONE";
              notifyPacket(
                reinterpret_cast<const uint8_t*>(calib_msg),
                sizeof(calib_msg) - 1
              );
          }
        }
      } else {
        // Apply calibration offsets
        gx -= gx_offset; gy -= gy_offset; gz -= gz_offset;
        ax -= ax_offset; ay -= ay_offset; az -= az_offset;

        unsigned long timestamp = millis();
        
        if (deviceConnected) {
          // One MTU-23-safe notification per sample avoids multi-packet frame loss.
          uint8_t packet[20];
          packet[0] = 'I';
          packet[1] = 'M';
          writeUint16LE(packet, 2, sample_sequence++);
          writeUint32LE(packet, 4, timestamp);
          writeInt16LE(packet, 8, static_cast<int16_t>(lroundf(ax * 1000.0f)));
          writeInt16LE(packet, 10, static_cast<int16_t>(lroundf(ay * 1000.0f)));
          writeInt16LE(packet, 12, static_cast<int16_t>(lroundf(az * 1000.0f)));
          writeInt16LE(packet, 14, static_cast<int16_t>(lroundf(gx * 100.0f)));
          writeInt16LE(packet, 16, static_cast<int16_t>(lroundf(gy * 100.0f)));
          writeInt16LE(packet, 18, static_cast<int16_t>(lroundf(gz * 100.0f)));
          notifyPacket(packet, sizeof(packet));
        }
      }
    } else {
      Serial.println("BMI160 Read Error");
    }
  }
}