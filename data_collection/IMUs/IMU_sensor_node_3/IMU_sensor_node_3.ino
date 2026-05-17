#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <DFRobot_BMI160.h>

DFRobot_BMI160 bmi160;
const int8_t i2c_addr = 0x68;

BLEServer* pServer = NULL;
BLECharacteristic* pTxCharacteristic = NULL;
bool deviceConnected = false;

// CHANGE THIS BEFORE UPLOADING TO EACH SENSOR (e.g., ESP32_IMU_1, ESP32_IMU_2, ...)
#define NODE_ID "ESP32_IMU_3" 

#define SERVICE_UUID            "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX  "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX  "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

// Calibration offsets
float gx_offset = 0, gy_offset = 0, gz_offset = 0;
float ax_offset = 0, ay_offset = 0, az_offset = 0;
bool is_calibrating = false;
int calibration_samples = 0;
const int CALIBRATION_COUNT = 100;
float sum_gx = 0, sum_gy = 0, sum_gz = 0;
float sum_ax = 0, sum_ay = 0, sum_az = 0;

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

  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
                                           CHARACTERISTIC_UUID_RX,
                                           BLECharacteristic::PROPERTY_WRITE
                                         );
  pRxCharacteristic->setCallbacks(new MyCallbacks());

  pService->start();
  pServer->getAdvertising()->start();

  Serial.println("BLE UART Ready");
  Serial.print("Device Name: ");
  Serial.println(NODE_ID);
}

void loop() {
  int16_t accelGyro[6] = {0};

  if (bmi160.getAccelGyroData(accelGyro) == BMI160_OK) {
    // Convert to meaningful units
    float gx = accelGyro[0] * 3.14159 / 180.0;
    float gy = accelGyro[1] * 3.14159 / 180.0;
    float gz = accelGyro[2] * 3.14159 / 180.0;

    float ax = accelGyro[3] / 16384.0;
    float ay = accelGyro[4] / 16384.0;
    float az = accelGyro[5] / 16384.0;

    if (is_calibrating) {
      sum_gx += gx; sum_gy += gy; sum_gz += gz;
      sum_ax += ax; sum_ay += ay; sum_az += (az - 1.0); // Assuming sensor is flat on table, Z should be 1g
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
            String calib_msg = "CALIB_DONE";
            pTxCharacteristic->setValue(calib_msg.c_str());
            pTxCharacteristic->notify();
        }
      }
    } else {
      // Apply calibration offsets
      gx -= gx_offset; gy -= gy_offset; gz -= gz_offset;
      ax -= ax_offset; ay -= ay_offset; az -= az_offset;

      unsigned long timestamp = millis();
      
      // Send comma separated values: timestamp,ax,ay,az,gx,gy,gz
      String data = String(timestamp) + "," +
                    String(ax, 3) + "," +
                    String(ay, 3) + "," +
                    String(az, 3) + "," +
                    String(gx, 3) + "," +
                    String(gy, 3) + "," +
                    String(gz, 3);

      if (deviceConnected) {
        pTxCharacteristic->setValue(data.c_str());
        pTxCharacteristic->notify();
      }
    }
  } else {
    Serial.println("BMI160 Read Error");
  }

  // 20ms delay achieves ~50Hz sampling rate
  delay(20);
}