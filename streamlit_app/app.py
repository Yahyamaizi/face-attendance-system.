"""
app.py
Streamlit web app for the Face Attendance System.

Run locally with:
    streamlit run app.py

Three tabs:
1. Take Attendance - snapshot from webcam, detect + recognize faces, log to CSV
2. Enroll a Person  - capture snapshots to register a new face
3. Attendance Log   - view and download today's (or any day's) CSV log

Note: on free hosting (e.g. Streamlit Community Cloud), the filesystem is
NOT permanently persistent across redeploys/restarts. embeddings.pkl and the
CSV logs will survive while the app stays "awake" but may reset after the
app sleeps and restarts. For a school demo this is fine; for production use
you'd want a real database (see README).
"""

import os
import csv
from datetime import datetime

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1

from utils import load_embeddings, save_embeddings, add_embedding, identify_face

# ---------- Config ----------
APP_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(APP_DIR, "attendance_logs")
RECOGNITION_THRESHOLD = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(LOG_DIR, exist_ok=True)

st.set_page_config(page_title="Face Attendance System", page_icon="✅", layout="centered")


# ---------- Cached model loading (only runs once per server) ----------
@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=DEVICE)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return mtcnn, resnet


mtcnn, resnet = load_models()


# ---------- Attendance log helpers ----------
def get_log_path(date_str=None):
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"attendance_{date_str}.csv")


def load_already_logged(log_path):
    logged = set()
    if os.path.exists(log_path):
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    logged.add(row[0])
    return logged


def log_attendance(name):
    log_path = get_log_path()
    already_logged = load_already_logged(log_path)
    if name in already_logged or name == "Unknown":
        return False
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    return True


# ---------- UI ----------
st.title("✅ Face Attendance System")
st.caption("MTCNN + FaceNet (transfer learning, VGGFace2) — snapshot-based recognition")

tab_attendance, tab_enroll, tab_log = st.tabs(
    ["📸 Take Attendance", "➕ Enroll a Person", "📋 Attendance Log"]
)

# ===================== TAB 1: Take Attendance =====================
with tab_attendance:
    st.subheader("Take a photo to check in")
    photo = st.camera_input("Webcam", key="attendance_cam")

    if photo is not None:
        img = Image.open(photo).convert("RGB")
        db = load_embeddings()

        boxes, _ = mtcnn.detect(img)

        if boxes is None:
            st.warning("No face detected. Try again with better lighting/positioning.")
        else:
            faces = mtcnn.extract(img, boxes, save_path=None)
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)

            results = []
            for box, face_tensor in zip(boxes, faces):
                if face_tensor is None:
                    continue
                with torch.no_grad():
                    embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()

                name, distance = identify_face(db, embedding, threshold=RECOGNITION_THRESHOLD)
                x1, y1, x2, y2 = box
                color = "lime" if name != "Unknown" else "red"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label = name if distance is None else f"{name} ({distance:.2f})"
                draw.text((x1, max(y1 - 18, 0)), label, fill=color)

                logged_now = log_attendance(name)
                results.append((name, distance, logged_now))

            st.image(draw_img, caption="Detection result", use_container_width=True)

            for name, distance, logged_now in results:
                if name == "Unknown":
                    st.error("Unrecognized face — not logged.")
                elif logged_now:
                    st.success(f"✅ {name} logged just now (distance: {distance:.2f})")
                else:
                    st.info(f"{name} already logged today (distance: {distance:.2f})")

# ===================== TAB 2: Enroll a Person =====================
with tab_enroll:
    st.subheader("Register a new person")
    st.write("Take several photos (different angles/expressions) for a more robust match.")

    name_input = st.text_input("Person's full name")
    enroll_photo = st.camera_input("Webcam", key="enroll_cam")

    if "enroll_count" not in st.session_state:
        st.session_state.enroll_count = 0

    col1, col2 = st.columns(2)
    with col1:
        add_clicked = st.button("➕ Add this photo as a sample", disabled=not name_input)
    with col2:
        st.metric("Samples added this session", st.session_state.enroll_count)

    if add_clicked:
        if enroll_photo is None:
            st.warning("Take a photo first.")
        else:
            img = Image.open(enroll_photo).convert("RGB")
            face_tensor = mtcnn(img)
            if face_tensor is None:
                st.error("No face detected in this photo — try again.")
            else:
                with torch.no_grad():
                    embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()
                db = load_embeddings()
                db = add_embedding(db, name_input.strip(), embedding)
                save_embeddings(db)
                st.session_state.enroll_count += 1
                st.success(f"Sample {st.session_state.enroll_count} saved for '{name_input}'.")

    st.divider()
    db = load_embeddings()
    if db:
        st.write("**Currently enrolled people:**")
        for person, embs in db.items():
            st.write(f"- {person} ({len(embs)} sample(s))")
    else:
        st.write("No one enrolled yet.")

# ===================== TAB 3: Attendance Log =====================
with tab_log:
    st.subheader("Attendance log")

    available_logs = sorted(
        [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")], reverse=True
    )

    if not available_logs:
        st.write("No attendance logs yet.")
    else:
        selected_file = st.selectbox("Select a date", available_logs)
        log_path = os.path.join(LOG_DIR, selected_file)

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        rows = [r.split(",") for r in content.strip().split("\n")]
        st.table(rows)

        st.download_button(
            "⬇️ Download this log as CSV",
            data=content,
            file_name=selected_file,
            mime="text/csv",
        )
