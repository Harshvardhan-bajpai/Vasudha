#include <esp_now.h>
#include <WiFi.h>

// ========== ROVER MAC ADDRESS ==========
uint8_t roverAddress[] = {0xD0, 0xEF, 0x76, 0x45, 0x0D, 0xE4};

// ========== PACKET TYPE IDs ==========
// Must match rover exactly
#define PACKET_TYPE_GPS     0x01
#define PACKET_TYPE_SENSOR  0x02

// ========== ESP-NOW STRUCTURES ==========
// Must match rover exactly — packetType MUST be the first field in both structs

// Outgoing: commands to rover
typedef struct {
  char command[32];
} CommandData;

// Incoming: GPS + compass data from rover
typedef struct {
  uint8_t packetType;   // byte[0] = 0x01
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

// Incoming: sensor JSON from rover
typedef struct {
  uint8_t packetType;   // byte[0] = 0x02
  char    json[128];
} SensorFeedback;

CommandData         commandToSend;
GPSData             receivedGPS;
SensorFeedback      receivedSensor;
esp_now_peer_info_t peerInfo;

// ========== SETUP ==========
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== AgriVision Transmitter ===");

  WiFi.mode(WIFI_STA);
  delay(100);
  Serial.print("Transmitter MAC: ");
  Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) {
    Serial.println("[ERROR] ESP-NOW init failed");
    return;
  }

  esp_now_register_send_cb(esp_now_send_cb_t(OnDataSent));
  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));

  memcpy(peerInfo.peer_addr, roverAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("[ERROR] Failed to add rover peer — check roverAddress[]");
    return;
  }

  Serial.println("[OK] Transmitter ready!");
  Serial.println("\n-- Supported Commands --");
  Serial.println("  Movement : w50 / s30 / a40 / d60 / stop");
  Serial.println("  Servos   : s1=90 / s2=45 / s3=120 / s4=60 / s5=75");
  Serial.println("  Sensor   : get_data");
  Serial.println("  Tests    : test / testmotors");
  Serial.println("----------------------------------------\n");
}

// ========== CALLBACKS ==========
void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  if (status == ESP_NOW_SEND_SUCCESS) {
    Serial.println("  Delivered to rover");
  } else {
    Serial.println("  Delivery failed - is rover powered on?");
  }
}

void OnDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len) {
  // Read byte[0] first to identify packet type before any struct copy
  if (len < 1) return;

  uint8_t pktType = data[0];

  if (pktType == PACKET_TYPE_GPS && len >= (int)sizeof(GPSData)) {
    memcpy(&receivedGPS, data, sizeof(receivedGPS));
    printGPSasJSON();
  }
  else if (pktType == PACKET_TYPE_SENSOR && len >= (int)sizeof(SensorFeedback)) {
    memcpy(&receivedSensor, data, sizeof(receivedSensor));
    // Print sensor JSON directly — Python reads this line
    Serial.println(receivedSensor.json);
  }
  else {
    Serial.print("[WARN] Unknown packet type: 0x");
    Serial.println(pktType, HEX);
  }
}

// ========== PRINT GPS AS JSON (Python reads this) ==========
void printGPSasJSON() {
  Serial.print("{");
  Serial.print("\"fix\":");        Serial.print(receivedGPS.fix ? "true" : "false"); Serial.print(",");
  Serial.print("\"lat\":");        Serial.print(receivedGPS.lat, 6);                 Serial.print(",");
  Serial.print("\"lng\":");        Serial.print(receivedGPS.lng, 6);                 Serial.print(",");
  Serial.print("\"altitude\":");   Serial.print(receivedGPS.altitude, 1);            Serial.print(",");
  Serial.print("\"speed\":");      Serial.print(receivedGPS.speed, 2);               Serial.print(",");
  Serial.print("\"heading\":");    Serial.print(receivedGPS.heading, 1);             Serial.print(",");
  Serial.print("\"hdop\":");       Serial.print(receivedGPS.hdop, 2);                Serial.print(",");
  Serial.print("\"satellites\":"); Serial.print(receivedGPS.satellites);             Serial.print(",");
  Serial.print("\"direction\":\"");Serial.print(receivedGPS.direction);              Serial.print("\",");
  Serial.print("\"time\":\"");     Serial.print(receivedGPS.utc_time);               Serial.print("\",");
  Serial.print("\"date\":\"");     Serial.print(receivedGPS.utc_date);               Serial.print("\"");
  Serial.println("}");
}

// ========== SEND COMMAND TO ROVER ==========
void sendCommand(String cmd) {
  if (cmd.length() == 0) return;

  memset(commandToSend.command, 0, sizeof(commandToSend.command));
  cmd.toCharArray(commandToSend.command, sizeof(commandToSend.command));

  Serial.print("-> Sent: ");
  Serial.println(cmd);

  esp_err_t result = esp_now_send(roverAddress, (uint8_t *)&commandToSend, sizeof(commandToSend));
  if (result != ESP_OK) {
    Serial.println("   ESP-NOW send error");
  }
}

// ========== MAIN LOOP ==========
void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      sendCommand(input);
    }
  }
  delay(10);
}
