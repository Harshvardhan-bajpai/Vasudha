from flask import Flask, render_template, Response, request, jsonify
import predictor
import os
import json

from main import (
    mjpeg_generator,
    send_rover_command,
    send_servo_command,
    get_telemetry,
    set_plant_detection,
    request_sensor_data
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global logs for dashboard
dashboard_logs = []


def add_log(message):
    """Add message to both terminal and dashboard logs"""
    print(message)
    dashboard_logs.append(message)


# ---------------- DASHBOARD ----------------

@app.route("/")
def dashboard():

    return render_template("dashboard.html")


# ---------------- CAMERA STREAM ----------------

@app.route("/video/rover_feed")
def rover_feed():

    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------- ROVER COMMAND ----------------

@app.route("/api/rover/command", methods=["POST"])
def rover_command():

    data = request.json

    cmd = data.get("command")

    ok, err = send_rover_command(cmd)

    return jsonify({"ok": ok})


# ---------------- SERVO CONTROL ----------------

@app.route("/api/servo", methods=["POST"])
def servo_control():

    data = request.json

    servo = int(data.get("servo"))
    angle = int(data.get("angle"))

    ok, err = send_servo_command(servo, angle)

    return jsonify({"ok": ok})


# ---------------- PLANT DETECTION ----------------

@app.route("/api/plant_detection", methods=["POST"])
def plant_detection():

    data = request.json

    state = data.get("state", False)

    set_plant_detection(state)

    return jsonify({"ok": True})


# ---------------- TELEMETRY ----------------

@app.route("/api/telemetry")
def api_telemetry():

    telemetry_data = get_telemetry()
    
    add_log(f"📡 TELEMETRY: {telemetry_data}")
    
    return jsonify(telemetry_data)


# ---------------- SENSOR DATA ----------------

@app.route("/api/start_sensor_data", methods=["POST"])
def start_sensor_data():
    """Request sensor data from ESP32 and return it to user"""
    add_log("🔄 Sensor data request initiated from dashboard")
    
    sensor_data = request_sensor_data()
    
    if sensor_data is None:
        add_log("❌ Failed to retrieve sensor data (timeout)")
        return jsonify({
            "status": "error",
            "message": "Timeout waiting for sensor data from ESP32 (10 seconds)"
        }), 408
    
    add_log(f"✅ Sensor data retrieved and sent to dashboard: {sensor_data}")
    
    return jsonify({
        "status": "ok",
        "data": sensor_data
    })


# ---------------- DASHBOARD LOGS ----------------
# Add this route inside app.py alongside the other routes

@app.route("/api/plant_detection/status")
def plant_detection_status():
    from main import tracker
    return jsonify({
        "enabled":   tracker.enabled,
        "status":    tracker.status,    # idle | searching | detected | capturing | complete | incomplete
        "completed": tracker.completed
    })


@app.route("/api/logs")
def get_logs():

    return jsonify({"logs": dashboard_logs})


# ---------------- IMAGE UPLOAD ----------------

@app.route("/upload_images", methods=["POST"])
def upload_images():

    files = request.files.getlist("images")

    paths = []

    for f in files:

        path = os.path.join(UPLOAD_FOLDER, f.filename)

        f.save(path)

        paths.append(path)

    return jsonify({
        "success": True,
        "photos": paths
    })


# ---------------- AI PREDICTION ----------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    crop = data.get("name")

    photos = data.get("photos", [])

    sensors = {

        "temp": data.get("temp"),
        "uv_index": data.get("uv_index"),
        "humidity": data.get("humidity"),
        "nitrogen": data.get("nitrogen"),
        "phosphorous": data.get("phosphorous"),
        "potassium": data.get("potassium")

    }

    sensors = {k: v for k, v in sensors.items() if v not in [None, ""]}

    result = predictor.run_pipeline(crop, photos, sensors)

    return jsonify(result)


# ---------------- CONFIRM PREDICTION ----------------

@app.route("/confirm", methods=["POST"])
def confirm():

    data = request.json

    token = data.get("token")

    confirm = data.get("confirm", False)

    if not confirm:

        if token in predictor.pending:
            predictor.pending.pop(token)

        return jsonify({"status": "cancelled"})

    result = predictor.resume_pipeline(token)

    return jsonify(result)

@app.route("/confirm_booking", methods=["POST"])
def confirm_booking():

    data = request.json

    booking_info = json.dumps(data, indent=2)
    
    add_log("=" * 50)
    add_log("BOOKING CONFIRMATION RECEIVED:")
    add_log(booking_info)
    add_log("=" * 50)

    return jsonify({"status": "ok", "message": "Booking data received and logged"})

# ---------------- SERVER START ----------------

if __name__ == "__main__":

    print("VARAH Rover Dashboard + AI System")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )