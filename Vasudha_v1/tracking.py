import cv2
import time
import threading
import requests
import os
from ultralytics import YOLO

# =========================
# CONFIG
# =========================

STREAM_URL           = "http://192.168.137.196:8081/?action=stream"
SNAPSHOT_URL         = "http://192.168.137.196:8080/?action=snapshot"
# STREAM_URL           = 0
# SNAPSHOT_URL         = 0
SAVE_FOLDER          = "rover_samples"
PLANT_CLASS_ID       = 58
CONFIDENCE_THRESHOLD = 0.05
CENTER_TOLERANCE     = 1  # plant center must be within 40% of frame center
MISS_FRAME_TOLERANCE = 25   # allow up to 1 consecutive missed frames before incomplete
DETECTION_COOLDOWN   = 5     # seconds between detection attempts

BASE_POS = {"s1": 109, "s2": 90, "s3": 32, "s4": 28, "s5": 116}

TARGET_POSITIONS = [
    # Position 1 — fire in this exact order
    [("s1", 109), ("s2", 90),  ("s3", 29),  ("s4", 28),  ("s5", 18)],
    # Position 2
    [("s1", 68),  ("s3", 122), ("s2", 147), ("s4", 116), ("s5", 127)],
    # Position 3
    [("s2", 116), ("s1", 109), ("s3", 122), ("s4", 30),  ("s5", 120)],
]


# =========================
# Tracking Controller
# =========================

class TrackingController:

    def __init__(self, send_command_callback=None):
        self.send_command = send_command_callback
        self.enabled      = False
        self.running      = False
        self.completed    = False
        self.status       = "idle"  # idle | searching | detected | capturing | complete | incomplete

        self.model = YOLO('yolov8n.pt')
        self.last_detection = 0

        os.makedirs(SAVE_FOLDER, exist_ok=True)
        self.cap = cv2.VideoCapture(STREAM_URL)

    # =========================
    # Enable / Disable
    # =========================

    def set_mode(self, state):
        self.enabled = state
        if state and not self.running:
            self.completed = False  # reset on re-enable
            self.status    = "searching"
            threading.Thread(target=self.loop, daemon=True).start()
        if not state:
            self.status = "idle"

    # =========================
    # Main Detection Loop
    # =========================

    def loop(self):
        self.running = True
        print("🌿 Plant detection loop started (monitoring only)")

        while self.enabled:
            # Just monitor the enabled state, don't execute arm movement
            time.sleep(0.5)

        self.running = False
        self.status  = "idle"
        print("🛑 Plant detection loop stopped")

    # =========================
    # Execute Arm Sequence (continuous movement)
    # =========================

    def execute_arm_sequence(self):
        """
        Continuously move arm through all target positions.
        Stop immediately if self.enabled becomes False.
        """
        self.status = "capturing"

        # Stop rover
        print("🛑 Stopping rover...")
        if self.send_command:
            self.send_command("stop")
        time.sleep(1)

        for idx, target in enumerate(TARGET_POSITIONS):
            if not self.enabled:
                print("⚠ Detection disabled — stopping arm movement")
                self.return_to_base()
                return

            position_num = idx + 1
            print(f"\n📐 Moving to target position {position_num}...")

            # Move servos to target position
            self.move_servos_smooth(target)

            # Wait before next position
            print(f"⏳ Position {position_num} complete, waiting before next...")
            time.sleep(1)

            # Take snapshot
            print(f"📸 Taking snapshot at position {position_num}...")
            self.take_snapshot(position_num)

        # Sequence complete
        print("\n✅ ARM SEQUENCE COMPLETE")
        self.return_to_base()
        self.status = "searching"

    # =========================
    # YOLO — Check Plant is Centered
    # =========================

    def detect_plant_centered(self, frame):
        h, w     = frame.shape[:2]
        frame_cx = w / 2
        frame_cy = h / 2

        results = self.model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id != PLANT_CLASS_ID or confidence < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                plant_cx = (x1 + x2) / 2
                plant_cy = (y1 + y2) / 2

                x_ok = abs(plant_cx - frame_cx) < (w * CENTER_TOLERANCE)
                y_ok = abs(plant_cy - frame_cy) < (h * CENTER_TOLERANCE)

                if x_ok and y_ok:
                    return True

        return False

    # =========================
    # Check Plant Still Visible
    # (used during servo movement)
    # =========================

    def is_plant_visible(self):
        """
        Reads up to MISS_FRAME_TOLERANCE frames.
        Returns True if plant found in any of them.
        """
        for _ in range(MISS_FRAME_TOLERANCE):
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            results = self.model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    if (int(box.cls[0]) == PLANT_CLASS_ID and
                            float(box.conf[0]) >= CONFIDENCE_THRESHOLD):
                        return True

            time.sleep(0.03)

        return False  # plant not found in any tolerance frame

    # =========================
    # Smooth Servo Movement
    # =========================

    def move_servos_smooth(self, target_list):
        """
        Moves each servo one degree at a time with 10ms delay.
        Fires servos in exact order as written in target_list.
        """
        for servo_name, target_angle in target_list:
            if not self.enabled:
                print("⚠ Detection disabled — stopping servo movement")
                return

            base_angle = BASE_POS.get(servo_name, 90)
            servo_num  = int(servo_name[1])  # "s1" → 1, "s2" → 2 etc

            print(f"  ➡ {servo_name}: {base_angle}° → {target_angle}°")

            step = 1 if target_angle > base_angle else -1

            for angle in range(base_angle, target_angle + step, step):
                if not self.enabled:
                    print("⚠ Detection disabled — stopping servo movement")
                    return

                if self.send_command:
                    self.send_command(f"s{servo_num}={angle}")
                time.sleep(0.01)  # 10ms per degree

    # =========================
    # Return to Base Position
    # =========================

    def return_to_base(self):
        print("🏠 Returning servos to base position...")
        for servo_name, base_angle in BASE_POS.items():
            servo_num = int(servo_name[1])
            if self.send_command:
                self.send_command(f"s{servo_num}={base_angle}")
            time.sleep(0.05)
        print("✅ Servos at base position")

    # =========================
    # Take Snapshot
    # =========================

    def take_snapshot(self, position_num):
        try:
            response = requests.get(SNAPSHOT_URL, timeout=5)
            if response.status_code == 200:
                filename = f"{SAVE_FOLDER}/plant_pos{position_num}_{int(time.time())}.jpg"
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"  ✅ Saved: {filename}")
                return filename
            else:
                print(f"  ❌ Snapshot failed — HTTP {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print("  ❌ Snapshot timed out")
            return None
        except Exception as e:
            print(f"  ❌ Snapshot error: {e}")
            return None