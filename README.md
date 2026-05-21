# 🇵🇰 Pakistani Currency Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

A professional, real-time Deep Learning application built with **YOLOv8** and **OpenCV** to detect and calculate Pakistani currency notes. This system includes both static image analysis and real-time live webcam detection with built-in total amount calculation.

## ✨ Features

- **Real-Time Detection:** Live webcam inference at high FPS.
- **Smart Amount Calculation:** Automatically detects multiple notes on screen and calculates the total Rs. amount.
- **High Accuracy:** Powered by a custom-trained YOLOv8 model.
- **Logging & Captures:** Automatically saves detection logs (`.txt`) and webcam screenshots (`.jpg`).
- **Confidence Scoring:** Displays the AI's confidence level for every detected note.

## 📁 Project Structure

```text
pakistani-currency-detector/
│
├── app/
│   ├── image_detector.py       # Script for static image testing
│   └── webcam_detector.py      # Script for live webcam detection
│
├── models/
│   └── best.pt                 # Your trained YOLOv8 model weights
│
├── test_images/                # Folder for sample testing images
│
├── outputs/                    # Auto-generated logs and saved screenshots
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation