"""
enroll_faces.py
Step 1 of the pipeline: register people into the known-faces database.

Usage:
    python enroll_faces.py --name "John Doe" --samples 15

Captures several frames of the person's face from the webcam, computes a
FaceNet embedding for each, and stores them under that name in embeddings.pkl.
Capturing multiple samples (different angles/expressions) makes recognition
more robust later.
"""

import argparse
import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from utils import load_embeddings, save_embeddings, add_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")


def main():
    parser = argparse.ArgumentParser(description="Enroll a new person's face.")
    parser.add_argument("--name", required=True, help="Full name of the person")
    parser.add_argument("--samples", type=int, default=15, help="Number of face samples to capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    person_dir = os.path.join(KNOWN_FACES_DIR, args.name.replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)

    print(f"[INFO] Loading models on device: {DEVICE}")
    mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=DEVICE)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    db = load_embeddings()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check --camera index.")

    print(f"[INFO] Enrolling '{args.name}'. Look at the camera.")
    print("[INFO] Press 'c' to capture a sample, 'q' to quit early.")

    captured = 0
    while captured < args.samples:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"Captured: {captured}/{args.samples}  (press 'c' to capture, 'q' to quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            face_tensor = mtcnn(pil_img)
            if face_tensor is None:
                print("[WARN] No face detected, try again.")
                continue

            with torch.no_grad():
                embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()

            db = add_embedding(db, args.name, embedding)

            # Save the captured face crop as a reference image
            save_path = os.path.join(person_dir, f"sample_{captured}.jpg")
            cv2.imwrite(save_path, frame)

            captured += 1
            print(f"[INFO] Sample {captured}/{args.samples} captured.")

    cap.release()
    cv2.destroyAllWindows()

    save_embeddings(db)
    print(f"[DONE] Saved {captured} samples for '{args.name}' to embeddings.pkl")


if __name__ == "__main__":
    main()
