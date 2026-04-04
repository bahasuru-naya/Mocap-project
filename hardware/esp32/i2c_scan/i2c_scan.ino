#include <Wire.h>

void setup() {
  Wire.begin(21, 22);
  Serial.begin(115200);
  Serial.println("Scanning I2C...");

  for (byte i = 1; i < 127; i++) {
    Wire.beginTransmission(i);
    if (Wire.endTransmission() == 0) {
      Serial.print("Device found at: 0x");
      Serial.println(i, HEX);
    }
  }
}

void loop() {}