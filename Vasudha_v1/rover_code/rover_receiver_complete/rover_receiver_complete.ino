#include <DHT.h>
#include <DHT_U.h>

#include <esp_now.h>
#include <WiFi.h>
#include <TinyGPSPlus.h>
#include <HardwareSerial.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_HMC5883_U.h>

// ========== PIN DEFINITIONS ==========
#define ENA 21
#define IN1 19
#define IN2 18
#define IN3 22
#define IN4 23
#define ENB 32

#define SERVO_BASE        26
#define SERVO_SHOULDER    14
#define SERVO_ELBOW       27
#define SERVO_WRIST_PITCH 25
#define SERVO_WRIST_ROLL  33

#define GPS_RX_PIN  16
#define GPS_TX_PIN  17
#define I2C_SDA_PIN 13
#define I2C_SCL_PIN 15

#define SERVO_SENSOR  2
#define DHT11_PIN     4
#define SOIL_PIN     34

// ========== PACKET TYPE IDs ==========
#define PACKET_TYPE_GPS     0x01
#define PACKET_TYPE_SENSOR  0x02

// ========== COMPASS CALIBRATION ==========
#define OFFSET_X  24.05
#define OFFSET_Y -13.36
#define NORM_X     0.97
#define NORM_Y     1.03

// ========== PWM SETTINGS ==========
#define MOTOR_PWM_FREQ       1000
#define MOTOR_PWM_RESOLUTION 8
#define SERVO_PWM_FREQ       50
#define SERVO_PWM_RESOLUTION 16

// ========== TRANSMITTER MAC ==========
uint8_t transmitterAddress[] = {0xB4, 0x8A, 0x0A, 0xB2, 0xA0, 0x6C};

// ========== ESP-NOW STRUCTURES ==========

typedef struct {
  char command[32];
} CommandData;

// packetType MUST be first field in both outgoing structs
typedef struct {
  uint8_t packetType;   // PACKET_TYPE_GPS (0x01)
  float   lat;
  float   lng;
  float   altitude;
  float   speed;
  float   heading;
  float   hdop;
  int     satellites;
  bool    fix;
  char    direction[4];
  char    utc_time[10];
  char    utc_date[12];
} GPSData;

typedef struct {
  uint8_t packetType;   // PACKET_TYPE_SENSOR (0x02)
  char    json[128];
} SensorFeedback;

CommandData         receivedCommand;
GPSData             gpsPacket;
SensorFeedback      sensorPacket;
esp_now_peer_info_t peerInfo;

// ========== OBJECTS ==========
TinyGPSPlus              gps;
HardwareSerial           GPSSerial(2);
Adafruit_HMC5883_Unified mag = Adafruit_HMC5883_Unified(12345);
DHT                      dht(DHT11_PIN, DHT11);

// ========== TIMING ==========
unsigned long lastGPSSend   = 0;
unsigned long lastServoStep = 0;
#define GPS_SEND_INTERVAL 500

// ========== SERVO TEST STATE ==========
bool  servoTestRunning = false;
bool  motorTestRunning = false;
int   servoTestPin     = 0;
int   servoTestAngle   = 90;
int   servoTestPhase   = 0;
int   motorTestStep    = 0;
unsigned long motorTestTimer = 0;
int   motorTestSpeed   = 30;

int servoPins[] = {SERVO_BASE, SERVO_SHOULDER, SERVO_ELBOW,
                   SERVO_WRIST_PITCH, SERVO_WRIST_ROLL};

// ========== SENSOR SERVO STATE MACHINE ==========
enum SensorServoState {
  SS_IDLE,
  SS_SWEEP_TO_0,
  SS_READ_AND_SEND,
  SS_WAIT_AT_0,
  SS_RETURN_TO_90
};
SensorServoState sensorServoState   = SS_IDLE;
bool             sensorServoPending = false;
unsigned long    sensorSweepStart   = 0;

// ========== ARM PRESET STATE MACHINE ==========
enum ArmPresetState {
  AP_IDLE,
  AP_HOMING,             // Move all 5 servos to home angles before preset 1
  AP_EXECUTING_PRESET_1,
  AP_WAIT_2SEC_1,
  AP_EXECUTING_PRESET_2,
  AP_WAIT_2SEC_2,
  AP_EXECUTING_PRESET_3,
  AP_WAIT_2SEC_3,
  AP_EXECUTING_PRESET_4,
  AP_COMPLETE
};
ArmPresetState armPresetState   = AP_IDLE;
bool           armPresetPending = false;
unsigned long  armPresetTimer   = 0;

struct ServoMove {
  int servoIndex;   // 0=s1  1=s2  2=s3  3=s4  4=s5
  int targetAngle;
};

// Home position — run once before preset 1: s1=90 s2=91 s3=92 s4=93 s5=94
ServoMove homePosition[] = {
  {0, 180}, {1, 60}, {2, 13}, {3, 114}, {4, 117}
};
int homePosition_size = 5;

// Preset 1
ServoMove preset1[] = {
  {0, 115}, {1, 60}, {2, 13}, {3, 114}, {4, 117}
};
int preset1_size = 5;

// Preset 2
ServoMove preset2[] = {
  {0, 65}, {2, 120}, {1, 150}, {3, 180}, {4, 117}
};
int preset2_size = 5;

// Preset 3
ServoMove preset3[] = {
  {1, 118}, {2, 102}, {0, 104}, {4, 33}, {3, 169}
};
int preset3_size = 5;

// Preset 4
ServoMove preset4[] = {
  {0, 180}, {1, 60}, {2, 13}, {3, 114}, {4, 117}
};
int preset4_size = 5;

// Physical pins indexed 0-4, matching servoIndex in presets
int armPins[] = {SERVO_BASE, SERVO_SHOULDER, SERVO_ELBOW, SERVO_WRIST_PITCH, SERVO_WRIST_ROLL};

// Per-move tracking — one servo moves one degree every 10ms
int           currentMoveIndex         = 0;
int           currentServoCurrentAngle = 0;
int           currentServoTargetAngle  = 0;
unsigned long servoMoveTimer           = 0;
ServoMove*    currentPreset            = NULL;
int           currentPresetSize        = 0;

// ========== SERVO HELPERS ==========
int angleToDutyCycle(int angle) {
  int pulseWidth = map(angle, 0, 180, 500, 2500);
  return (pulseWidth * 65536) / 20000;
}

void setServoAngle(int pin, int angle) {
  angle = constrain(angle, 0, 180);
  ledcWrite(pin, angleToDutyCycle(angle));
}

// ========== LOAD PRESET — sets up state for executing a new preset ==========
// startAngle is the actual current angle of the first servo in the preset
void loadPreset(ServoMove* preset, int size) {
  currentPreset            = preset;
  currentPresetSize        = size;
  currentMoveIndex         = 0;
  currentServoTargetAngle  = preset[0].targetAngle;
  // Read the actual current angle from the first servo so sweep starts correctly.
  // We track it via currentServoCurrentAngle which was left at the end of the
  // previous move — for the very first move after homing we set it explicitly.
  servoMoveTimer           = millis();
  Serial.print("[ARM] S"); Serial.print(preset[0].servoIndex + 1);
  Serial.print(" -> ");    Serial.println(preset[0].targetAngle);
}

// ========== SENSOR READ + JSON SEND ==========
void readSensorsAndSend() {
  delay(1400);
  float temp     = dht.readTemperature();
  float humidity = dht.readHumidity();

  int soilRaw = analogRead(SOIL_PIN);
  int soilPct = map(soilRaw, 4095, 0, 0, 100);
  soilPct     = constrain(soilPct, 0, 100);

  char tempStr[10];
  char humStr[10];
  char soilStr[10];

  if (isnan(temp))     strncpy(tempStr, "null", sizeof(tempStr));
  else                 snprintf(tempStr, sizeof(tempStr), "%.1f", temp);

  if (isnan(humidity)) strncpy(humStr, "null", sizeof(humStr));
  else                 snprintf(humStr, sizeof(humStr), "%.1f", humidity);

  snprintf(soilStr, sizeof(soilStr), "%d", soilPct);

  sensorPacket.packetType = PACKET_TYPE_SENSOR;
  snprintf(sensorPacket.json, sizeof(sensorPacket.json),
    "{\"temp\":%s,\"humidity\":%s,\"soil\":%s,\"servo\":180}",
    tempStr, humStr, soilStr);

  Serial.print("[SENSOR TX] ");
  Serial.println(sensorPacket.json);

  esp_now_send(transmitterAddress, (uint8_t *)&sensorPacket, sizeof(sensorPacket));
}

// ========== SENSOR SERVO RUNNER (non-blocking) ==========
void runSensorServo() {
  if (sensorServoState == SS_IDLE) {
    if (sensorServoPending) {
      sensorServoPending = false;
      sensorServoState   = SS_SWEEP_TO_0;
      sensorSweepStart   = millis();
      setServoAngle(SERVO_SENSOR, 0);
      Serial.println("[SENSOR] Commanded 0deg, waiting 200ms...");
    }
    return;
  }

  switch (sensorServoState) {

    case SS_SWEEP_TO_0:
      if (millis() - sensorSweepStart >= 200) {
        sensorServoState = SS_READ_AND_SEND;
        Serial.println("[SENSOR] At 0deg, reading sensors...");
      }
      break;

    case SS_READ_AND_SEND:
      readSensorsAndSend();
      sensorServoState = SS_WAIT_AT_0;
      sensorSweepStart = millis();
      Serial.println("[SENSOR] Waiting 2 seconds at 0deg...");
      break;

    case SS_WAIT_AT_0:
      if (millis() - sensorSweepStart >= 2000) {
        sensorServoState = SS_RETURN_TO_90;
        sensorSweepStart = millis();
        setServoAngle(SERVO_SENSOR, 90);
        Serial.println("[SENSOR] Commanded 90deg, waiting 200ms...");
      }
      break;

    case SS_RETURN_TO_90:
      if (millis() - sensorSweepStart >= 200) {
        sensorServoState = SS_IDLE;
        Serial.println("[SENSOR] Back at 90deg - Idle.");
      }
      break;

    default:
      break;
  }
}

// ========== ARM PRESET EXECUTOR (non-blocking, 10ms per degree) ==========
//
// Flow:
//   get_data received
//     -> AP_HOMING        : sweep s1-s5 to home angles (90,91,92,93,94) one degree per 10ms
//     -> AP_EXECUTING_PRESET_1 : sweep each servo in preset1 to its target, one degree per 10ms
//     -> AP_WAIT_2SEC_1   : hold 2000ms
//     -> AP_EXECUTING_PRESET_2 : same for preset2
//     -> AP_WAIT_2SEC_2   : hold 2000ms
//     -> AP_EXECUTING_PRESET_3 : same for preset3
//     -> AP_COMPLETE -> AP_IDLE
//
// Each servo in the current preset is swept fully before moving to the next one.
// Any manual command (s1= s2= etc.) aborts immediately and resets to AP_IDLE.

void runArmPresets() {

  // ---- IDLE: check if a new sequence was requested ----
  if (armPresetState == AP_IDLE) {
    if (armPresetPending) {
      armPresetPending         = false;
      armPresetState           = AP_HOMING;
      currentPreset            = homePosition;
      currentPresetSize        = homePosition_size;
      currentMoveIndex         = 0;
      // Start from wherever each servo physically is — we don't know, so we read
      // the target of the first home move and assume worst case (far side).
      // The 10ms-per-degree loop handles it correctly regardless.
      currentServoTargetAngle  = homePosition[0].targetAngle;
      currentServoCurrentAngle = 0;   // assume worst case — will sweep from 0
      servoMoveTimer           = millis();
      Serial.println("[ARM] Homing servos: s1=90 s2=91 s3=92 s4=93 s5=94");
      Serial.print("[ARM] S1 -> "); Serial.println(homePosition[0].targetAngle);
    }
    return;
  }

  // ---- COMPLETE: return to idle ----
  if (armPresetState == AP_COMPLETE) {
    armPresetState = AP_IDLE;
    Serial.println("[ARM] All presets done. Arm control returned.");
    return;
  }

  // ---- WAIT STATES ----
  if (armPresetState == AP_WAIT_2SEC_1 || armPresetState == AP_WAIT_2SEC_2 || armPresetState == AP_WAIT_2SEC_3) {
    if (millis() - armPresetTimer >= 2000) {
      if (armPresetState == AP_WAIT_2SEC_1) {
        armPresetState           = AP_EXECUTING_PRESET_2;
        currentPreset            = preset2;
        currentPresetSize        = preset2_size;
        Serial.println("[ARM] Executing Preset 2");
      } else if (armPresetState == AP_WAIT_2SEC_2) {
        armPresetState           = AP_EXECUTING_PRESET_3;
        currentPreset            = preset3;
        currentPresetSize        = preset3_size;
        Serial.println("[ARM] Executing Preset 3");
      } else {
        armPresetState           = AP_EXECUTING_PRESET_4;
        currentPreset            = preset4;
        currentPresetSize        = preset4_size;
        Serial.println("[ARM] Executing Preset 4");
      }
      currentMoveIndex         = 0;
      currentServoTargetAngle  = currentPreset[0].targetAngle;
      currentServoCurrentAngle = currentServoTargetAngle;  // start from target = no sweep needed if already there; real position unknown so sweep from 90
      currentServoCurrentAngle = 90;
      servoMoveTimer           = millis();
      Serial.print("[ARM] S"); Serial.print(currentPreset[0].servoIndex + 1);
      Serial.print(" -> ");    Serial.println(currentPreset[0].targetAngle);
    }
    return;
  }

  // ---- EXECUTING (homing or any preset) — 10ms per degree ----
  if (armPresetState == AP_HOMING             ||
      armPresetState == AP_EXECUTING_PRESET_1 ||
      armPresetState == AP_EXECUTING_PRESET_2 ||
      armPresetState == AP_EXECUTING_PRESET_3 ||
      armPresetState == AP_EXECUTING_PRESET_4) {

    // All moves done in this preset?
    if (currentMoveIndex >= currentPresetSize) {
      if (armPresetState == AP_HOMING) {
        // Homing done — start preset 1
        armPresetState           = AP_EXECUTING_PRESET_1;
        currentPreset            = preset1;
        currentPresetSize        = preset1_size;
        currentMoveIndex         = 0;
        currentServoTargetAngle  = preset1[0].targetAngle;
        currentServoCurrentAngle = 90; // servos were just homed to ~90
        servoMoveTimer           = millis();
        Serial.println("[ARM] Homing done. Executing Preset 1");
        Serial.print("[ARM] S"); Serial.print(preset1[0].servoIndex + 1);
        Serial.print(" -> ");    Serial.println(preset1[0].targetAngle);
      } else if (armPresetState == AP_EXECUTING_PRESET_1) {
        armPresetState = AP_WAIT_2SEC_1;
        armPresetTimer = millis();
        Serial.println("[ARM] Preset 1 done. Waiting 2 seconds...");
      } else if (armPresetState == AP_EXECUTING_PRESET_2) {
        armPresetState = AP_WAIT_2SEC_2;
        armPresetTimer = millis();
        Serial.println("[ARM] Preset 2 done. Waiting 2 seconds...");
      } else if (armPresetState == AP_EXECUTING_PRESET_3) {
        armPresetState = AP_WAIT_2SEC_3;
        armPresetTimer = millis();
        Serial.println("[ARM] Preset 3 done. Waiting 2 seconds...");
      } else {
        armPresetState = AP_COMPLETE;
        Serial.println("[ARM] Preset 4 done.");
      }
      return;
    }

    // Time for another degree step?
    if (millis() - servoMoveTimer < 10) return;
    servoMoveTimer = millis();

    int pin = armPins[currentPreset[currentMoveIndex].servoIndex];

    if (currentServoCurrentAngle != currentServoTargetAngle) {
      // Move one degree toward target
      int step = (currentServoTargetAngle > currentServoCurrentAngle) ? 1 : -1;
      currentServoCurrentAngle += step;
      setServoAngle(pin, currentServoCurrentAngle);
    } else {
      // This servo reached its target — advance to next move in preset
      currentMoveIndex++;
      if (currentMoveIndex < currentPresetSize) {
        currentServoTargetAngle  = currentPreset[currentMoveIndex].targetAngle;
        // Use the angle we just finished as the starting point for the next servo.
        // Since we don't track each servo's current angle separately, start from
        // 90 (safe home) so the sweep is always correct.
        currentServoCurrentAngle = 90;
        servoMoveTimer           = millis();
        Serial.print("[ARM] S"); Serial.print(currentPreset[currentMoveIndex].servoIndex + 1);
        Serial.print(" -> ");    Serial.println(currentServoTargetAngle);
      }
      // if currentMoveIndex >= currentPresetSize the top-of-function check handles it next tick
    }
  }
}

// ========== SETUP ==========
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== AgriVision Rover v2 ===\n");

  // Motors
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  ledcAttach(ENA, MOTOR_PWM_FREQ, MOTOR_PWM_RESOLUTION);
  ledcAttach(ENB, MOTOR_PWM_FREQ, MOTOR_PWM_RESOLUTION);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  ledcWrite(ENA, 0); ledcWrite(ENB, 0);
  Serial.println("[OK] Motors");

  // Arm servos
  ledcAttach(SERVO_BASE,        SERVO_PWM_FREQ, SERVO_PWM_RESOLUTION);
  ledcAttach(SERVO_SHOULDER,    SERVO_PWM_FREQ, SERVO_PWM_RESOLUTION);
  ledcAttach(SERVO_ELBOW,       SERVO_PWM_FREQ, SERVO_PWM_RESOLUTION);
  ledcAttach(SERVO_WRIST_PITCH, SERVO_PWM_FREQ, SERVO_PWM_RESOLUTION);
  ledcAttach(SERVO_WRIST_ROLL,  SERVO_PWM_FREQ, SERVO_PWM_RESOLUTION);
  setServoAngle(SERVO_BASE,        180);
  setServoAngle(SERVO_SHOULDER,    60);
  setServoAngle(SERVO_ELBOW,       13);
  setServoAngle(SERVO_WRIST_PITCH, 114);
  setServoAngle(SERVO_WRIST_ROLL, 117);
  Serial.println("[OK] Arm servos");

  // Sensor servo
  ledcAttach(SERVO_SENSOR, SERVO_PWM_FREQ, SERVO_PWM_RESOLUTION);
  setServoAngle(SERVO_SENSOR, 90);
  Serial.println("[OK] Sensor servo GPIO 2");

  // Pre-fill packetType fields once
  gpsPacket.packetType    = PACKET_TYPE_GPS;
  sensorPacket.packetType = PACKET_TYPE_SENSOR;

  // DHT11
  dht.begin();
  Serial.println("[OK] DHT11 GPIO 4");

  // Soil moisture
  pinMode(SOIL_PIN, INPUT);
  Serial.println("[OK] Soil moisture GPIO 34");

  // GPS
  GPSSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("[OK] GPS UART RX=16 TX=17");

  // Compass
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  if (!mag.begin()) {
    Serial.println("[WARN] Compass not found!");
  } else {
    Serial.println("[OK] Compass HMC5883L");
  }

  // WiFi + ESP-NOW
  WiFi.mode(WIFI_STA);
  delay(100);
  Serial.print("Rover MAC: "); Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) {
    Serial.println("[ERROR] ESP-NOW init failed"); return;
  }
  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));
  esp_now_register_send_cb(esp_now_send_cb_t(OnDataSent));

  memcpy(peerInfo.peer_addr, transmitterAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("[ERROR] Failed to add peer — check MAC!");
  } else {
    Serial.println("[OK] Transmitter peer registered");
  }

  Serial.println("\nSystem ready!\n");
}

// ========== ESP-NOW CALLBACKS ==========
void OnDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len) {
  memcpy(&receivedCommand, data, sizeof(receivedCommand));
  String cmd = String(receivedCommand.command);
  cmd.trim();
  Serial.print("CMD RX: "); Serial.println(cmd);
  processCommand(cmd);
}

void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "TX OK" : "TX FAIL");
}

// ========== GPS FEED ==========
void feedGPS() {
  while (GPSSerial.available()) gps.encode(GPSSerial.read());
}

// ========== SEND GPS PACKET ==========
void sendGPSPacket() {
  if (sensorServoState != SS_IDLE) return;
  if (millis() - lastGPSSend < GPS_SEND_INTERVAL) return;
  lastGPSSend = millis();

  gpsPacket.fix        = gps.location.isValid();
  gpsPacket.lat        = gps.location.isValid()   ? gps.location.lat()    : 0.0;
  gpsPacket.lng        = gps.location.isValid()   ? gps.location.lng()    : 0.0;
  gpsPacket.altitude   = gps.altitude.isValid()   ? gps.altitude.meters() : 0.0;
  gpsPacket.speed      = gps.speed.isValid()      ? gps.speed.kmph()      : 0.0;
  gpsPacket.satellites = gps.satellites.isValid() ? gps.satellites.value(): 0;
  gpsPacket.hdop       = gps.hdop.isValid()       ? gps.hdop.hdop()       : 99.9;

  if (gps.time.isValid())
    snprintf(gpsPacket.utc_time, sizeof(gpsPacket.utc_time),
             "%02d:%02d:%02d", gps.time.hour(), gps.time.minute(), gps.time.second());
  else
    strncpy(gpsPacket.utc_time, "--:--:--", sizeof(gpsPacket.utc_time));

  if (gps.date.isValid())
    snprintf(gpsPacket.utc_date, sizeof(gpsPacket.utc_date),
             "%02d/%02d/%04d", gps.date.day(), gps.date.month(), gps.date.year());
  else
    strncpy(gpsPacket.utc_date, "--/--/----", sizeof(gpsPacket.utc_date));

  sensors_event_t event;
  mag.getEvent(&event);
  float cx = (event.magnetic.x - OFFSET_X) * NORM_X;
  float cy = (event.magnetic.y - OFFSET_Y) * NORM_Y;
  float heading = atan2(cx, -cy) * 180.0 / PI - 80.0;
  if (heading < 0)    heading += 360;
  if (heading >= 360) heading -= 360;
  gpsPacket.heading = heading;

  if      (heading >= 337.5 || heading < 22.5)  strncpy(gpsPacket.direction, "N",  4);
  else if (heading >= 22.5  && heading < 67.5)  strncpy(gpsPacket.direction, "NE", 4);
  else if (heading >= 67.5  && heading < 112.5) strncpy(gpsPacket.direction, "E",  4);
  else if (heading >= 112.5 && heading < 157.5) strncpy(gpsPacket.direction, "SE", 4);
  else if (heading >= 157.5 && heading < 202.5) strncpy(gpsPacket.direction, "S",  4);
  else if (heading >= 202.5 && heading < 247.5) strncpy(gpsPacket.direction, "SW", 4);
  else if (heading >= 247.5 && heading < 292.5) strncpy(gpsPacket.direction, "W",  4);
  else if (heading >= 292.5 && heading < 337.5) strncpy(gpsPacket.direction, "NW", 4);

  esp_now_send(transmitterAddress, (uint8_t *)&gpsPacket, sizeof(gpsPacket));
}

// ========== SERVO TEST ==========
void runServoTest() {
  if (!servoTestRunning) return;
  if (millis() - lastServoStep < 30) return;
  lastServoStep = millis();

  if (servoTestPin >= 5) {
    servoTestRunning = false;
    Serial.println("Servo test complete!");
    return;
  }
  setServoAngle(servoPins[servoTestPin], servoTestAngle);

  if (servoTestPhase == 0) {
    servoTestAngle -= 5;
    if (servoTestAngle <= 0) { servoTestAngle = 0; servoTestPhase = 1; }
  } else if (servoTestPhase == 1) {
    servoTestAngle += 5;
    if (servoTestAngle >= 180) { servoTestAngle = 180; servoTestPhase = 2; }
  } else {
    servoTestAngle -= 5;
    if (servoTestAngle <= 90) {
      servoTestAngle = 90; servoTestPhase = 0; servoTestPin++;
    }
  }
}

// ========== MOTOR TEST ==========
void runMotorTest() {
  if (!motorTestRunning) return;
  unsigned long now = millis();
  switch (motorTestStep) {
    case 0: moveForward(motorTestSpeed);       motorTestTimer = now; motorTestStep = 1; break;
    case 1: if (now - motorTestTimer > 2000) { stopMotors();              motorTestTimer = now; motorTestStep = 2; } break;
    case 2: if (now - motorTestTimer > 500)  { moveBackward(motorTestSpeed); motorTestTimer = now; motorTestStep = 3; } break;
    case 3: if (now - motorTestTimer > 2000) { stopMotors();              motorTestTimer = now; motorTestStep = 4; } break;
    case 4: if (now - motorTestTimer > 500)  { turnLeft(motorTestSpeed);   motorTestTimer = now; motorTestStep = 5; } break;
    case 5: if (now - motorTestTimer > 1000) { stopMotors();              motorTestTimer = now; motorTestStep = 6; } break;
    case 6: if (now - motorTestTimer > 500)  { turnRight(motorTestSpeed);  motorTestTimer = now; motorTestStep = 7; } break;
    case 7: if (now - motorTestTimer > 1000) {
      stopMotors(); motorTestRunning = false; motorTestStep = 0;
      Serial.println("Motor test complete!");
    } break;
  }
}

// ========== COMMAND PROCESSOR ==========
void processCommand(String cmd) {
  cmd.toLowerCase();

  // Any manual servo or motor command immediately aborts arm presets
  bool isManualControl = cmd.startsWith("s1=") || cmd.startsWith("s2=") ||
                         cmd.startsWith("s3=") || cmd.startsWith("s4=") ||
                         cmd.startsWith("s5=") || cmd.startsWith("w")   ||
                         cmd.startsWith("a")   || cmd.startsWith("d")   ||
                         cmd == "stop";
  if (isManualControl) {
    armPresetState   = AP_IDLE;
    armPresetPending = false;
  }

  if      (cmd.startsWith("s1=")) { setServoAngle(SERVO_BASE,        constrain(cmd.substring(3).toInt(), 0, 180)); }
  else if (cmd.startsWith("s2=")) { setServoAngle(SERVO_SHOULDER,    constrain(cmd.substring(3).toInt(), 0, 180)); }
  else if (cmd.startsWith("s3=")) { setServoAngle(SERVO_ELBOW,       constrain(cmd.substring(3).toInt(), 0, 180)); }
  else if (cmd.startsWith("s4=")) { setServoAngle(SERVO_WRIST_PITCH, constrain(cmd.substring(3).toInt(), 0, 180)); }
  else if (cmd.startsWith("s5=")) { setServoAngle(SERVO_WRIST_ROLL,  constrain(cmd.substring(3).toInt(), 0, 180)); }
  else if (cmd.startsWith("w"))   { moveForward(cmd.substring(1).toInt()); }
  else if (cmd.startsWith("s") && !cmd.startsWith("s1") && !cmd.startsWith("s2")
                                 && !cmd.startsWith("s3") && !cmd.startsWith("s4")
                                 && !cmd.startsWith("s5")) {
    moveBackward(cmd.substring(1).toInt());
  }
  else if (cmd.startsWith("a"))   { turnRight(cmd.substring(1).toInt()); }
  else if (cmd.startsWith("d"))   { turnLeft(cmd.substring(1).toInt()); }
  else if (cmd == "stop")         { stopMotors(); }
  else if (cmd == "test") {
    servoTestRunning = true; servoTestPin = 0;
    servoTestAngle = 90; servoTestPhase = 0;
    Serial.println("Servo test started...");
  }
  else if (cmd == "testmotors") {
    motorTestRunning = true; motorTestStep = 0;
    Serial.println("Motor test started...");
  }
  else if (cmd == "readsensors" || cmd == "get_data") {
    if (sensorServoState == SS_IDLE && armPresetState == AP_IDLE) {
      sensorServoPending = true;   // sensor servo: GPIO 2
      armPresetPending   = true;   // arm presets:  GPIO 26/14/27/25/33
      Serial.println("[GET_DATA] Sensor servo + arm presets triggered simultaneously.");
    } else {
      Serial.println("[GET_DATA] Busy — ignored.");
    }
  }
  else { Serial.println("Unknown command"); }
}

// ========== MOTOR FUNCTIONS ==========
void moveForward(int sp) {
  sp = constrain(sp, 0, 100);
  int pwm = map(sp, 0, 100, 0, 255);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  ledcWrite(ENA, pwm); ledcWrite(ENB, pwm);
  Serial.print("Forward @ "); Serial.print(sp); Serial.println("%");
}

void moveBackward(int sp) {
  sp = constrain(sp, 0, 100);
  int pwm = map(sp, 0, 100, 0, 255);
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  ledcWrite(ENA, pwm); ledcWrite(ENB, pwm);
  Serial.print("Backward @ "); Serial.print(sp); Serial.println("%");
}

void turnLeft(int sp) {
  sp = constrain(sp, 0, 100);
  int pwm = map(sp, 0, 100, 0, 255);
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  ledcWrite(ENA, pwm); ledcWrite(ENB, pwm);
  Serial.print("Left @ "); Serial.print(sp); Serial.println("%");
}

void turnRight(int sp) {
  sp = constrain(sp, 0, 100);
  int pwm = map(sp, 0, 100, 0, 255);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  ledcWrite(ENA, pwm); ledcWrite(ENB, pwm);
  Serial.print("Right @ "); Serial.print(sp); Serial.println("%");
}

void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  ledcWrite(ENA, 0); ledcWrite(ENB, 0);
  Serial.println("STOP");
}

// ========== MAIN LOOP (fully non-blocking) ==========
void loop() {
  feedGPS();
  sendGPSPacket();
  runSensorServo();
  runArmPresets();
  runServoTest();
  runMotorTest();
}
