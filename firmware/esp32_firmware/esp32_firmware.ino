#include <ESP32Servo.h>

// Minimal UART stream of raw accelerometer (ax, ay, az)
// and gyroscope (gx, gy, gz) data from an Adafruit MPU‑6050.
// Format: ax,ay,az,gx,gy,gz\n  (comma‑separated, newline‑terminated)
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

const int SERVO_PIN = 4;
bool poke_state = false;
char receive = 's';
const unsigned long POKE_INTERVAL = 1000; // ms between pokes
unsigned long lastPokeTime = 0;

const int FUNKY_GND_PIN_IO = 5;

const int POKE_ANGLE = 140;
const int REST_ANGLE = 180;

int count_consecutive_state = 0;

Servo actuator_servo;
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);          // faster throughput; adjust to taste
  pinMode(FUNKY_GND_PIN_IO, OUTPUT);
  digitalWrite(FUNKY_GND_PIN_IO, LOW);

  // set up pin for servo
  pinMode(SERVO_PIN, OUTPUT);
  actuator_servo.attach(SERVO_PIN);
  actuator_servo.write(180);
  while (!Serial) { delay(10); } // wait for USB‑CDC port (native USB boards)

  if (!mpu.begin()) {
    // stay here if the sensor is not detected
    while (true) { delay(100); }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100); // give the sensor a moment to settle
}

void loop() {
  if (Serial.available()){
    receive = Serial.read();
    if (receive == 'p') {
      poke_state = true;
    } else if (receive == 's') {
      poke_state = false;
      actuator_servo.write(REST_ANGLE);
    }
  }
  if (poke_state){
    unsigned long currentTime = millis();
    if (currentTime - lastPokeTime >= POKE_INTERVAL) {
      lastPokeTime = currentTime;
      poke();
    }
  }
  // fetch fresh sensor data
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // transmit six values as CSV on one line
  Serial.print(a.acceleration.x); Serial.print(',');
  Serial.print(a.acceleration.y); Serial.print(',');
  Serial.print(a.acceleration.z); Serial.print(',');
  Serial.print(g.gyro.x);        Serial.print(',');
  Serial.print(g.gyro.y);        Serial.print(',');
  Serial.print(g.gyro.z);
  Serial.println();              // record separator ("\n")

  // delay(10); // ~100 Hz update rate; modify or remove as desired
}

void poke(){
  // one poke
  actuator_servo.write(POKE_ANGLE);
  delay(500);
  actuator_servo.write(REST_ANGLE); // angle 0 deg
}
