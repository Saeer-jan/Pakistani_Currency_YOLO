# ==========================================
# Pakistani Currency Detection System
# Real-Time Webcam Detection
# ==========================================

from ultralytics import YOLO
import cv2
import time
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = r"D:\Langchain\computer_vision\Pakistani_Currency_YOLO\models\best.pt"
OUTPUT_FOLDER = r"D:\Langchain\computer_vision\Pakistani_Currency_YOLO\outputs\webcam_captures"
COUNT_COOLDOWN_SECONDS = 3

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_note_value(class_name: str):
    digits = ''.join(ch for ch in str(class_name) if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def draw_summary_panel(frame, total_amount, note_counts, fps):
    overlay = frame.copy()
    height, width = frame.shape[:2]

    panel_height = 180
    cv2.rectangle(overlay, (10, 10), (width - 10, 10 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"FPS: {int(fps)}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Amount: Rs. {total_amount}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

    y = 125
    cv2.putText(frame, "Calculation:", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += 28

    if note_counts:
        for note_value in sorted(note_counts.keys()):
            count = note_counts[note_value]
            line = f"{note_value} x {count} = {note_value * count}"
            cv2.putText(frame, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 24
            if y > 10 + panel_height - 10:
                break
    else:
        cv2.putText(frame, "No notes counted yet.", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = YOLO(MODEL_PATH)
print("\n✅ YOLO Model Loaded Successfully!")

# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("\n❌ Error: Cannot access webcam!")
    raise SystemExit

print("\n🎥 Webcam Started Successfully!")
print("\nPress:")
print("  Q  → Save final screenshot and quit")
print("  S  → Save screenshot without quitting\n")

# ==========================================
# STATE
# ==========================================

prev_time = time.time()
total_amount = 0
note_counts = {}
last_count_time = {}
last_seen_time = {}

# ==========================================
# MAIN LOOP
# ==========================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("\n❌ Failed to capture frame!")
        break

    current_time = time.time()

    results = model.predict(
        source=frame,
        conf=0.68,          # confidence threshold
        imgsz=640,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()

    # Track detections that are currently visible in this frame.
    seen_this_frame = set()

    if len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            note_value = extract_note_value(class_name)

            if note_value is None:
                continue

            seen_this_frame.add(note_value)
            last_seen_time[note_value] = current_time

            # Count only when the note reappears after being absent for a short time.
            last_counted_at = last_count_time.get(note_value, 0)
            if current_time - last_counted_at >= COUNT_COOLDOWN_SECONDS:
                total_amount += note_value
                note_counts[note_value] = note_counts.get(note_value, 0) + 1
                last_count_time[note_value] = current_time
                print(f"Counted: Rs. {note_value} | Confidence: {confidence:.2f} | Total: Rs. {total_amount}")
            else:
                print(f"Seen again: Rs. {note_value} | Confidence: {confidence:.2f}")

    # Optional cleanup for notes that have disappeared for a while.
    for note_value in list(last_seen_time.keys()):
        if current_time - last_seen_time[note_value] > COUNT_COOLDOWN_SECONDS:
            last_seen_time.pop(note_value, None)

    # FPS calculation
    fps = 1 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    display_frame = annotated_frame.copy()
    draw_summary_panel(display_frame, total_amount, note_counts, fps)

    cv2.imshow("Pakistani Currency Detection System", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_path = os.path.join(OUTPUT_FOLDER, f"final_detection_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"\n📸 Final screenshot saved: {screenshot_path}")
        print(f"✅ Final total amount: Rs. {total_amount}")
        break

    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_path = os.path.join(OUTPUT_FOLDER, f"detection_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"\n📸 Screenshot saved: {screenshot_path}")

# ==========================================
# RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()
print("\n✅ System Closed Successfully!")
