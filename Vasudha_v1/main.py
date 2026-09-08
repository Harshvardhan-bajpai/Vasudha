import serial
import threading
import json
import time
import cv2

# =========================
# SERIAL CONFIG
# =========================

SERIAL_PORT = "COM7"      # change if needed
BAUD_RATE = 115200

ser = None

try:

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    print("ESP32 connected on", SERIAL_PORT)

except Exception as e:

    print("⚠ WARNING: ESP32 serial port not found")
    print("Dashboard will run without rover connection")


# =========================
# CAMERA SETUP
# =========================

camera = cv2.VideoCapture("http://192.168.137.196:8080/?action=stream")
# camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_lock = threading.Lock()
latest_frame = None


def camera_loop():

    global latest_frame

    while True:

        ret, frame = camera.read()

        if not ret:
            continue

        with frame_lock:
            latest_frame = frame


threading.Thread(target=camera_loop, daemon=True).start()


# =========================
# MJPEG GENERATOR
# =========================

def mjpeg_generator():

    global latest_frame

    while True:

        if latest_frame is None:
            time.sleep(0.02)
            continue

        with frame_lock:
            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        jpg = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpg +
               b"\r\n")


# =========================
# TELEMETRY STATE
# =========================

telemetry = {

    "fix": True,
    "lat": "28.4073051",
    "lng": "77.1186202",
    "altitude": "223",
    "speed": "--",
    "heading": "--",
    "hdop": "--",
    "satellites": "3",
    "direction": "--",
    "time": "--",
    "date": "--",
    "battery": 94
}


# =========================
# BATTERY SIMULATION
# =========================

def battery_drain_loop():

    while True:

        time.sleep(60)

        if telemetry["battery"] > 0:
            telemetry["battery"] -= 1


threading.Thread(target=battery_drain_loop, daemon=True).start()


# =========================
# SERIAL READER
# =========================

def serial_reader():

    global ser

    print("Serial reader thread started")
    
    while True:

        if ser is None:
            time.sleep(1)
            continue

        try:

            line = ser.readline().decode().strip()

            if not line:
                continue
            
            print(f"📨 RAW DATA FROM ESP: {line}")

            # Check if this is JSON (sensor data or GPS data)
            if line.startswith("{"):

                data = json.loads(line)

                print(f"✅ ESP32 DATA RECEIVED: {data}")

                # Check if this is SENSOR DATA (has temp/humidity/soil fields)
                if "temp" in data or "humidity" in data or "soil" in data:
                    print(f"📊 SENSOR DATA DETECTED: {data}")
                    with sensor_data_lock:
                        global sensor_data_received
                        sensor_data_received = data
                    print(f"✅ Sensor data stored: {data}")
                    continue

                # Otherwise, treat as GPS/telemetry data
                telemetry["fix"] = True  # HARDCODED

                # Always update these fields (they work without GPS fix)
                telemetry["heading"] = data.get("heading", "--")
                telemetry["direction"] = data.get("direction", "--")
                telemetry["speed"] = data.get("speed", "--")
                telemetry["hdop"] = data.get("hdop", "--")
                telemetry["time"] = data.get("time", "--")
                telemetry["date"] = data.get("date", "--")

                # Only update GPS coordinates if we have a fix
                if data.get("fix"):

                    telemetry["lat"] = data.get("lat", "28.4073051")
                    telemetry["lng"] = data.get("lng", "77.1186202")
                    telemetry["altitude"] = str(data.get("altitude", "223"))
                    telemetry["satellites"] = str(data.get("satellites", "3"))
                    
                    print(f"📍 GPS FIX: Lat={telemetry['lat']}, Lng={telemetry['lng']}, Heading={telemetry['heading']}°, Altitude={telemetry['altitude']}m, Satellites={telemetry['satellites']}")

                else:

                    telemetry["lat"] = "28.4073051"
                    telemetry["lng"] = "77.1186202"
                    telemetry["altitude"] = "123"
                    telemetry["satellites"] = "3"
                    
                    print(f"⚠ GPS NO FIX (using defaults: heading={telemetry['heading']}°, direction={telemetry['direction']}, altitude={telemetry['altitude']}m, satellites={telemetry['satellites']})")
            else:
                print(f"⚠ Received non-JSON data: {line}")

        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            
        except Exception as e:

            print("Serial read error:", e)


if ser is not None:
    threading.Thread(target=serial_reader, daemon=True).start()
else:
    print("Serial reader disabled (ESP not connected)")


# =========================
# COMMAND SENDER
# =========================

def send_command(cmd):

    global ser

    if ser is None:

        print("⚠ Serial not connected. Command ignored:", cmd)

        return False, "serial_not_connected"

    try:

        ser.write((cmd + "\n").encode())

        print("ESP CMD →", cmd)

        return True, None

    except Exception as e:

        print("Serial write error:", e)

        return False, str(e)


# =========================
# ROVER MOVEMENT
# =========================

def send_rover_command(cmd):

    # Handle stop
    if cmd == "stop":
        return send_command("stop")

    # Dashboard sends direction + speed as one string e.g. "w72", "a55"
    # Parse direction (first char) and speed (remaining digits)
    if len(cmd) >= 2 and cmd[0] in ("w", "s", "a", "d") and cmd[1:].isdigit():
        direction = cmd[0]
        speed     = max(0, min(100, int(cmd[1:])))  # clamp 0-100
        return send_command(f"{direction}{speed}")

    # Fallback: plain single-letter with default speeds
    defaults = {
        "w": "w60",
        "s": "s60",
        "a": "a50",
        "d": "d50"
    }

    if cmd in defaults:
        return send_command(defaults[cmd])

    return False, "invalid command"


# =========================
# SERVO CONTROL
# =========================

def send_servo_command(servo, angle):

    cmd = f"s{servo}={angle}"

    return send_command(cmd)


# =========================
# PLANT DETECTION
# =========================

from tracking import TrackingController

tracker = TrackingController(send_command_callback=send_command)


def set_plant_detection(state):

    print("Plant detection:", state)

    tracker.set_mode(state)


# =========================
# TELEMETRY ACCESS
# =========================

def get_telemetry():

    return telemetry


# =========================
# SENSOR DATA COLLECTION
# =========================

sensor_data_received = None
sensor_data_lock = threading.Lock()


def request_sensor_data():
    """
    Sends get_data command to ESP32 and waits for response.
    Returns sensor data dict or None if timeout.
    """
    global sensor_data_received
    
    sensor_data_received = None
    
    # Send command to ESP32
    print("📊 Requesting sensor data from ESP32...")
    ok, err = send_command("get_data")
    
    if not ok:
        print("❌ Failed to send get_data command")
        return None
    
    # Wait up to 10 seconds for response
    timeout = 10
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        with sensor_data_lock:
            if sensor_data_received is not None:
                print(f"✅ Sensor data received: {sensor_data_received}")
                return sensor_data_received
        
        time.sleep(0.1)
    
    print("⏱ Timeout waiting for sensor data")
    return None