#include "mpu6500.h"
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;
bfs::Mpu6500 mpu6500(&Wire, bfs::Mpu6500::I2C_ADDR_PRIM);

void setup() {
  Serial.begin(115200);
  
  // Initialize I2C bus
  Wire.begin();
  Wire.setClock(400000);
  
  // Initialize Bluetooth
  SerialBT.begin("ESP32_IMU");  // Bluetooth name
  Serial.println("Bluetooth Started: ESP32_IMU");
  
  // Initialize sensor
  if (!mpu6500.Begin()) {
    Serial.println("MPU-6500 initialization failed!");
    SerialBT.println("ERROR: Sensor failed!");
    while(1) {}
  }
  Serial.println("MPU-6500 initialized successfully!");
  SerialBT.println("READY");
}

void loop() {
  if (mpu6500.Read()) {
    if (mpu6500.new_imu_data()) {
      float ax = mpu6500.accel_x_mps2();
      float ay = mpu6500.accel_y_mps2();
      float az = mpu6500.accel_z_mps2();
      
      float gx = mpu6500.gyro_x_radps();
      float gy = mpu6500.gyro_y_radps();
      float gz = mpu6500.gyro_z_radps();
      
      float temp = mpu6500.die_temp_c();
      
      // Send data as CSV format: ax,ay,az,gx,gy,gz,temp
      String data = String(ax, 3) + "," + 
                    String(ay, 3) + "," + 
                    String(az, 3) + "," + 
                    String(gx, 3) + "," + 
                    String(gy, 3) + "," + 
                    String(gz, 3) + "," + 
                    String(temp, 2);
      
      SerialBT.println(data);
      Serial.println(data);  // Also print to Serial Monitor
    }
  }
  delay(50);  // 20 Hz update rate
}