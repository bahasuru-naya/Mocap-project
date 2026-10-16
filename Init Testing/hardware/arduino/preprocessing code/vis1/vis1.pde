import processing.serial.*;

Serial myPort;

float pitch = 0;
float roll = 0;
float temperature = 0;

void setup() {
  size(900, 600, P3D);
  smooth(8);
  
  println(Serial.list());

  // Arduino COM port
  myPort = new Serial(this, Serial.list()[0], 115200);
  myPort.bufferUntil('\n');

  textSize(16);
}

void draw() {
  background(180);
  lights();

  // ===== 2D TEXT DISPLAY =====
  camera();
  hint(DISABLE_DEPTH_TEST);
  fill(0);

  text("MPU9250 Real-Time Data", 20, 30);
  text("Temperature : " + nf(temperature, 1, 2) + " °C", 20, 60);
  text("Pitch        : " + nf(pitch, 1, 2) + " °", 20, 90);
  text("Roll         : " + nf(roll, 1, 2) + " °", 20, 120);

  hint(ENABLE_DEPTH_TEST);

  // ===== 3D MULTI-COLOR CUBE =====
  pushMatrix();
  translate(width / 2, height / 2 + 50, 0);

  rotateX(radians(pitch));
  rotateZ(radians(roll));

  float s = 200; // cube size
  // Front face
  fill(255, 0, 0); // red
  beginShape();
  vertex(-s/2, -s/2, s/2);
  vertex(s/2, -s/2, s/2);
  vertex(s/2, s/2, s/2);
  vertex(-s/2, s/2, s/2);
  endShape(CLOSE);

  // Back face
  fill(0, 255, 0); // green
  beginShape();
  vertex(-s/2, -s/2, -s/2);
  vertex(s/2, -s/2, -s/2);
  vertex(s/2, s/2, -s/2);
  vertex(-s/2, s/2, -s/2);
  endShape(CLOSE);

  // Right face
  fill(0, 0, 255); // blue
  beginShape();
  vertex(s/2, -s/2, -s/2);
  vertex(s/2, -s/2, s/2);
  vertex(s/2, s/2, s/2);
  vertex(s/2, s/2, -s/2);
  endShape(CLOSE);

  // Left face
  fill(255, 255, 0); // yellow
  beginShape();
  vertex(-s/2, -s/2, -s/2);
  vertex(-s/2, -s/2, s/2);
  vertex(-s/2, s/2, s/2);
  vertex(-s/2, s/2, -s/2);
  endShape(CLOSE);

  // Top face
  fill(0, 255, 255); // cyan
  beginShape();
  vertex(-s/2, -s/2, -s/2);
  vertex(s/2, -s/2, -s/2);
  vertex(s/2, -s/2, s/2);
  vertex(-s/2, -s/2, s/2);
  endShape(CLOSE);

  // Bottom face
  fill(255, 0, 255); // magenta
  beginShape();
  vertex(-s/2, s/2, -s/2);
  vertex(s/2, s/2, -s/2);
  vertex(s/2, s/2, s/2);
  vertex(-s/2, s/2, s/2);
  endShape(CLOSE);

  popMatrix();
}

void serialEvent(Serial myPort) {
  String data = myPort.readStringUntil('\n');
  if (data == null) return;

  data = trim(data);
  data = data.replace("Â", "");
  
  println(data);

  // TEMP:  31.95°C
  if (data.startsWith("TEMP:")) {
    data = data.replace("TEMP:", "")
               .replace("°C", "")
               .trim();
    temperature = parseFloat(data);
    println(temperature);
  }

  // Pitch:  12.83°
  else if (data.startsWith("Pitch:")) {
    data = data.replace("Pitch:", "")
               .replace("°", "")
               .trim();
    pitch = parseFloat(data);
    println(pitch);
  }

  // Roll:  50.23°
  else if (data.startsWith("Roll:")) {
    data = data.replace("Roll:", "")
               .replace("°", "")
               .trim();
    roll = parseFloat(data);
    println(roll);
  }
}
