# 🇵🇰 Pakistani Currency Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

A professional, real-time Deep Learning application built with **YOLOv8** and **OpenCV** to detect and calculate Pakistani currency notes. 

The standout feature of this system is its **Dynamic Live Counting**: simply hold your currency notes in front of the webcam, and the AI will instantly detect the denominations, draw bounding boxes, and calculate the total amount of money in real-time right on your screen.

## ✨ Features

- 🔥 **Dynamic Real-Time Counting:** Put any combination of notes towards the camera, and the system instantly recognizes them and updates the total Rs. amount live.
- ⚡ **High-Speed Inference:** Optimised YOLOv8 webcam detection at high FPS for smooth video performance.
- 🎯 **High Accuracy:** Powered by a custom-trained object detection model specifically built for Pakistani currency.
- 📝 **Smart Logging & Captures:** Automatically saves detailed detection logs (`.txt`) and takes high-quality webcam screenshots (`.jpg`).
- 📊 **Confidence Scoring:** Displays the AI's exact confidence percentage for every single note detected on screen.

## 📥 Download the Pre-Trained Model

To run this project, you need the trained AI weights (`best.pt`). I have provided my custom-trained model for public use. 

👉 **[Download `best.pt` from Google Drive](https://drive.google.com/file/d/1SfNSnKLVLVsgxl7F_0tvuwW1Q-wffxfJ/view?usp=sharing)**

Once downloaded, place the `best.pt` file inside the `models/` folder in your project directory.

## 📁 Project Structure

```text
pakistani-currency-detector/
│
├── app/
│   ├── image_detector.py       # Script for static image testing and logging
│   └── webcam_detector.py      # Script for live webcam detection & live counting
│
├── models/
│   └── best.pt                 # (Download from GDrive and place here)
│
├── test_images/                # Folder for sample testing images
│
├── outputs/                    # Auto-generated logs and saved screenshots
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
