# ==========================================
# Pakistani Currency Detection System
# Image Detection Script
# ==========================================

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "D:\\Langchain\\computer_vision\\Pakistani_Currency_YOLO\\models\\best.pt"

model = YOLO(MODEL_PATH)

print("\n✅ Model loaded successfully!")

# ==========================================
# IMAGE PATH
# ==========================================

# Change this to your test image
IMAGE_PATH = "D:\\Langchain\\computer_vision\\Pakistani_Currency_YOLO\\test_images\\image3.jpg"

    
    

# ==========================================
# CHECK IMAGE EXISTS
# ==========================================

if not os.path.exists(IMAGE_PATH):
    print(f"\n❌ Image not found: {IMAGE_PATH}")
    exit()

# ==========================================
# RUN YOLO PREDICTION
# ==========================================

print("\n🔍 Detecting currency notes...")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.40,          # confidence threshold
    save=True,          # save output image
    imgsz=640,
    line_width=3,
    show_labels=True,
    show_conf=True
)

print("\n✅ Detection completed!")

# ==========================================
# EXTRACT DETECTION INFORMATION
# ==========================================

result = results[0]

boxes = result.boxes

if len(boxes) == 0:
    print("\n⚠️ No currency note detected!")
else:

    print(f"\n💰 Total Notes Detected: {len(boxes)}\n")

    for i, box in enumerate(boxes):

        # Class ID
        class_id = int(box.cls[0])

        # Confidence
        confidence = float(box.conf[0])

        # Class Name
        class_name = model.names[class_id]

        print(f"Note {i+1}")
        print(f"Detected Note : Rs. {class_name}")
        print(f"Confidence    : {confidence:.2f}")
        print("-" * 30)

# ==========================================
# GET SAVED IMAGE PATH
# ==========================================

output_path = result.save_dir

# Find output image automatically
saved_images = os.listdir(output_path)

output_image_path = None

for file in saved_images:

    if file.endswith((".jpg", ".png", ".jpeg", ".webp")):
        output_image_path = os.path.join(output_path, file)
        break

# ==========================================
# DISPLAY OUTPUT IMAGE
# ==========================================

if output_image_path:

    img = cv2.imread(output_image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 12))

    plt.imshow(img)

    plt.title("Pakistani Currency Detection")

    plt.axis("off")

    plt.show()

else:
    print("\n❌ Output image not found!")

# ==========================================
# SAVE DETECTION LOG
# ==========================================

log_folder = "D:\\Langchain\\computer_vision\\Pakistani_Currency_YOLO\\outputs"


os.makedirs(log_folder, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_file = os.path.join(log_folder, f"detection_log_{timestamp}.txt")

with open(log_file, "w") as f:

    f.write("Pakistani Currency Detection Log\n")
    f.write("=" * 40 + "\n\n")

    if len(boxes) == 0:
        f.write("No currency note detected.\n")

    else:

        for i, box in enumerate(boxes):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            f.write(f"Note {i+1}\n")
            f.write(f"Detected Note : Rs. {class_name}\n")
            f.write(f"Confidence    : {confidence:.2f}\n")
            f.write("-" * 30 + "\n")

print(f"\n📝 Detection log saved successfully!")
print(f"📁 Log File: {log_file}")

# ==========================================
# END OF SCRIPT
# ==========================================