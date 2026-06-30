"""
attendance_system.py
Step 2 of the pipeline: real-time face detection + recognition + attendance logging.

Usage:
    python attendance_system.py

Opens the webcam, detects faces with MTCNN, recognizes them by comparing
FaceNet embeddings against embeddings.pkl, draws a live overlay (box + name),
and logs each recognized person's first appearance of the day to a CSV file
in attendance_logs/.
"""

import os
import csv
import cv2
import torch
import numpy as np
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from utils import load_embeddings, identify_face

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = os.path.join(os.path.dirname(__file__), "attendance_logs")
RECOGNITION_THRESHOLD = 0.4  # lower = stricter match


def get_today_log_path():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"attendance_{today}.csv")


def load_already_logged(log_path):
    """Return the set of names already logged today, to avoid duplicate entries."""
    logged = set()
    if os.path.exists(log_path):
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    logged.add(row[0])
    return logged


def log_attendance(log_path, name, already_logged):
    """Append a new attendance record if this person hasn't been logged yet today."""
    if name in already_logged or name == "Unknown":
        return
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    already_logged.add(name)
    print(f"[ATTENDANCE] {name} logged at {datetime.now().strftime('%H:%M:%S')}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    db = load_embeddings()
    if not db:
        print("[WARN] No known faces found. Run enroll_faces.py first.")

    print(f"[INFO] Loading models on device: {DEVICE}")
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=DEVICE)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    log_path = get_today_log_path()
    already_logged = load_already_logged(log_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("[INFO] Starting attendance system. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        boxes, _ = mtcnn.detect(pil_img)

        if boxes is not None:
            faces = mtcnn.extract(pil_img, boxes, save_path=None)

            for box, face_tensor in zip(boxes, faces):
                if face_tensor is None:
                    continue

                with torch.no_grad():
                    embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()

                name, distance = identify_face(db, embedding, threshold=RECOGNITION_THRESHOLD)

                x1, y1, x2, y2 = [int(v) for v in box]
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
                label = name if distance is None else f"{name} ({distance:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

                log_attendance(log_path, name, already_logged)

        cv2.putText(
            frame, f"Logged today: {len(already_logged)}", (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )
        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
