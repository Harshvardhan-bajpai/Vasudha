# 🌱 Vasudha

**AI-Powered Precision Agriculture System for Crop Disease Detection and Prediction**

---

## 🚀 Overview

Vasudha is an AI-driven agricultural intelligence system designed to help farmers detect crop diseases early and monitor farm health using AI-assisted technology.

The system combines:
- 📱 Mobile-based crop diagnosis  
- 🤖 Autonomous rover for on-field monitoring  
- 🧠 AI models for disease detection and prediction  

By leveraging **computer vision + environmental sensing**, Vasudha provides reliable, real-time insights into crop health.

---

## 🧠 Key Features

- 🌿 Crop disease detection using deep learning (ResNet-based models)  
- 📊 Multi-stage AI pipeline (crop classification → disease detection → validation)  
- 🚫 OOD (Out-of-Distribution) filtering for reliable predictions  
- 🌡️ Sensor-based validation (temperature, humidity, UV, NPK)  
- 📸 Multi-image aggregation for stable predictions  
- 📡 Real-time Flask backend APIs  
- 🖥️ Interactive web dashboard for monitoring and control  

---

## 📱 Mobile App

The mobile application allows farmers to:

- Upload crop images  
- Get instant disease predictions  
- Use the platform **completely free**  
- Detect crop issues early and take preventive action  

📈 **Accuracy:** ~91% (real-world conditions)

---

## 🤖 Autonomous Rover System

The rover is designed for **continuous on-field crop monitoring** and high-precision data collection.

### Capabilities

- 📷 Captures plant images from multiple angles  
- 🎯 Detects plants and positions robotic arm automatically  
- 🧠 Runs AI-based disease prediction  
- 🎮 Controlled via web dashboard (WASD + arm control)  
- 📡 Sends real-time telemetry (GPS, direction, speed)  
- 🔄 Enables repeated farm inspection without manual effort  

📈 **Accuracy:** ~97% (close-range monitoring)

---

## 🔩 Hardware Components (Rover)

The rover is built using:

- 🔧 **5-DOF Robotic Arm** – multi-angle image capture  
- ⚙️ **2 DC Motors** – rover movement  
- 🔌 **L298N Motor Driver** – motor control  
- 📡 **ESP32** – main controller and communication  
- 🛰️ **GPS + Compass Module** – location and heading tracking  
- 📷 Camera Module – plant image capture  

---

## 🧪 AI Pipeline
```
Image Input
↓
Crop Classification
↓
OOD Filtering
↓
Disease Detection
↓
Sensor Validation
↓
Multi-Image Aggregation
↓
Final Prediction
```

---

## 📦 Model Downloads

All trained models required for this project can be downloaded from:

👉 https://drive.google.com/drive/folders/1bNhwkvduSpNBdSir8H6QQdqHhzk9l5Bz?usp=drive_link

---

### 📁 Folder Structure (Important)

```
Vasudha/
│
├── 📁 crop_classifier_trained_model/
│   ├── resnet18_best.pth
│   └── ood_stats.json
│
├── 📁 rice_trained_model/
│   └── resnet18_best.pth
│
├── 📁 wheat_trained_model/
│   └── resnet18_best.pth
│
├── 📁 sugarcane_trained_model/
│   └── resnet18_best.pth
│
├── 📁 potato_trained_model/
│   └── resnet18_best.pth
│
├── 📁 templates/
│   └── dashboard.html
│
├── 📁 uploads/
│   └── (empty directory for uploaded images)
│
├── 📄 app.py
├── 📄 main.py
├── 📄 predictor.py
├── 📄 sensor_values.json
├── 📄 requirements.txt
```

---

### ⚠️ Notes

- Do not rename model files  
- Ensure paths match `predictor.py`  
- Missing models will result in `"unknown"` predictions  

---

## ⚙️ Installation & Setup

### 1. Clone Repository
git clone https://github.com/Harshvardhan-bajpai/Vasudha.git

cd Vasudha

---

### 2. Install Dependencies
pip install -r requirements.txt

---

### 3. Download Models

Download from the link above and place them in correct folders.

---

### 4. Run Server
python app.py

---

### 5. Open Dashboard/Mobile APP 
Note: Both devices (pc and mobile) should be on same local network.
Note: Check the local ip of your pc and place it in main.dart file in apk (default: 192.168.137.1).


---

## 🌍 Use Cases

- Early crop disease detection  
- Continuous farm monitoring  
- Precision agriculture  
- Reducing crop losses  
- Data-driven farming decisions  

---

## 🎯 Vision

Vasudha aims to build a **complete agricultural intelligence system** by combining AI, IoT, and robotics to make farming smarter, scalable, and predictive.

---

## 📌 Tech Stack

- Python  
- PyTorch  
- OpenCV  
- Flask  
- ESP32 (Embedded Systems)  
- Computer Vision & Machine Learning  

---

## 👨‍💻 Author

**Harshvardhan Bajpai**

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!
