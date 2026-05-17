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

#define SERVICE_UUID            "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX  "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX  "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

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
    String rxValue = pCharacteristic->getValue();   // FIXED

    if (rxValue.length() > 0) {
      Serial.print("Received: ");
      Serial.println(rxValue);
    }
  }
};

void setup() {
  Serial.begin(115200);
  delay(100);

  Wire.begin();
  Wire.setClock(400000);

  if (bmi160.softReset() != BMI160_OK) {
    Serial.println("BMI160 reset failed!");
    while (1);
  }

  if (bmi160.I2cInit(i2c_addr) != BMI160_OK) {
    Serial.println("BMI160 init failed!");
    while (1);
  }

  Serial.println("BMI160 Ready");

  BLEDevice::init("ESP32_IMU_1");
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
  Serial.println("Device Name: ESP32_IMU_1");
}

void loop() {
  int16_t accelGyro[6] = {0};

  if (bmi160.getAccelGyroData(accelGyro) == BMI160_OK) {
    float gx = accelGyro[0] * 3.14159 / 180.0;
    float gy = accelGyro[1] * 3.14159 / 180.0;
    float gz = accelGyro[2] * 3.14159 / 180.0;

    float ax = accelGyro[3] / 16384.0;
    float ay = accelGyro[4] / 16384.0;
    float az = accelGyro[5] / 16384.0;

    String data = String(ax, 3) + "," +
                  String(ay, 3) + "," +
                  String(az, 3) + "," +
                  String(gx, 3) + "," +
                  String(gy, 3) + "," +
                  String(gz, 3);

    Serial.println(data);

    if (deviceConnected) {
      pTxCharacteristic->setValue(data.c_str());
      pTxCharacteristic->notify();
    }
  } else {
    Serial.println("BMI160 Read Error");
  }

  delay(50);
}