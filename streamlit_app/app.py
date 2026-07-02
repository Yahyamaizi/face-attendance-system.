"""
app.py
Streamlit web app for the Face Attendance System (Pro version).

Run locally with:
    streamlit run app.py

Tabs:
1. Take Attendance - snapshot from webcam, detects ALL faces in frame,
   recognizes each one, and marks everyone present in a single photo
   (great for a classroom / group check-in).
2. Enroll a Person  - capture snapshots to register a new face
3. Dashboard        - attendance stats + log viewer + CSV download
"""

import os
import csv
from datetime import datetime

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
@st.cache_resource(show_spinner="Loading AI models (first run only)...")
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
    """Log a person as present (once per day). Returns True if newly logged."""
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


def detect_and_recognize(img):
    """Run detection + recognition on a PIL image. Returns (annotated_img, results)."""
    db = load_embeddings()
    boxes, _ = mtcnn.detect(img)

    draw_img = img.copy()
    results = []

    if boxes is None:
        return draw_img, results

    faces = mtcnn.extract(img, boxes, save_path=None)
    draw = ImageDraw.Draw(draw_img)

    for box, face_tensor in zip(boxes, faces):
        if face_tensor is None:
            continue
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()

        name, distance = identify_face(db, embedding, threshold=RECOGNITION_THRESHOLD)
        x1, y1, x2, y2 = box
        color = "lime" if name != "Unknown" else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = name if distance is None else f"{name} ({distance:.2f})"
        draw.text((x1, max(y1 - 20, 0)), label, fill=color)

        results.append({"name": name, "distance": distance, "box": box})

    return draw_img, results


# ---------- UI ----------
st.title("✅ Face Attendance System")
st.caption("MTCNN + FaceNet (transfer learning, VGGFace2) — recognizes everyone in one photo")

tab_attendance, tab_enroll, tab_dashboard = st.tabs(
    ["📸 Take Attendance", "➕ Enroll a Person", "📊 Dashboard"]
)

# ===================== TAB 1: Take Attendance =====================
with tab_attendance:
    st.subheader("Mark attendance")
    st.write(
        "Take a photo — works for one person **or a whole group at once**. "
        "Everyone recognized gets marked present automatically."
    )

    photo = st.camera_input("Webcam", key="attendance_cam")

    if photo is not None:
        img = Image.open(photo).convert("RGB")
        db = load_embeddings()

        if not db:
            st.warning("No one is enrolled yet. Go to the 'Enroll a Person' tab first.")
        else:
            with st.spinner("Detecting and recognizing faces..."):
                draw_img, results = detect_and_recognize(img)

            if not results:
                st.warning("No face detected. Try again with better lighting/positioning.")
            else:
                st.image(draw_img, caption=f"{len(results)} face(s) detected", use_container_width=True)

                newly_logged, already_present, unknown_count = [], [], 0
                for r in results:
                    if r["name"] == "Unknown":
                        unknown_count += 1
                        continue
                    was_new = log_attendance(r["name"])
                    (newly_logged if was_new else already_present).append(r["name"])

                if newly_logged:
                    st.success(f"✅ Marked present: {', '.join(newly_logged)}")
                if already_present:
                    st.info(f"ℹ️ Already marked earlier today: {', '.join(already_present)}")
                if unknown_count:
                    st.error(f"❌ {unknown_count} unrecognized face(s) — not logged.")

# ===================== TAB 2: Enroll a Person =====================
with tab_enroll:
    st.subheader("Register a new person")
    st.write("Take several photos (different angles/expressions) for a more robust match — aim for 8-15 samples.")

    name_input = st.text_input("Person's full name")
    enroll_photo = st.camera_input("Webcam", key="enroll_cam")

    if "enroll_count" not in st.session_state:
        st.session_state.enroll_count = 0

    col1, col2 = st.columns(2)
    with col1:
        add_clicked = st.button(
            "➕ Add this photo as a sample", disabled=not name_input, use_container_width=True
        )
    with col2:
        st.metric("Samples added this session", st.session_state.enroll_count)

    st.write(f"🔧 DEBUG — name: '{name_input}' | photo taken: {enroll_photo is not None} | button clicked: {add_clicked}")

    if add_clicked:
        if enroll_photo is None:
            st.warning("Take a photo first.")
        else:
            img = Image.open(enroll_photo).convert("RGB")
            face_tensor = mtcnn(img)
            if face_tensor is None or len(face_tensor) == 0:
                st.error("No face detected in this photo — try again.")
            else:
                # mtcnn is configured with keep_all=True, so it already returns
                # a batched tensor of shape (num_faces, 3, 160, 160).
                # Take the first detected face for enrollment.
                first_face = face_tensor[0]
                with torch.no_grad():
                    embedding = resnet(first_face.unsqueeze(0).to(DEVICE)).cpu().numpy()
                db = load_embeddings()
                db = add_embedding(db, name_input.strip(), embedding)
                save_embeddings(db)
                st.session_state.enroll_count += 1
                st.success(f"Sample {st.session_state.enroll_count} saved for '{name_input}'.")

    st.divider()
    db = load_embeddings()
    if db:
        st.write("**Currently enrolled people:**")
        for person, embs in sorted(db.items()):
            st.write(f"- {person} ({len(embs)} sample(s))")
    else:
        st.write("No one enrolled yet.")

# ===================== TAB 3: Dashboard =====================
with tab_dashboard:
    st.subheader("Attendance dashboard")

    available_logs = sorted(
        [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")], reverse=True
    )

    if not available_logs:
        st.write("No attendance logs yet — take attendance first.")
    else:
        selected_file = st.selectbox("Select a date", available_logs)
        log_path = os.path.join(LOG_DIR, selected_file)

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        rows = [r.split(",") for r in content.strip().split("\n")]
        data_rows = rows[1:]

        db = load_embeddings()
        total_enrolled = len(db)
        total_present = len(data_rows)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total enrolled", total_enrolled)
        col2.metric("Present that day", total_present)
        col3.metric(
            "Attendance rate",
            f"{(total_present / total_enrolled * 100):.0f}%" if total_enrolled else "—",
        )

        st.divider()
        st.write("**Attendance record:**")
        st.table(rows)

        st.download_button(
            "⬇️ Download this log as CSV",
            data=content,
            file_name=selected_file,
            mime="text/csv",
            use_container_width=True,
        )

        if total_enrolled and data_rows:
            present_names = {r[0] for r in data_rows}
            absent_names = set(db.keys()) - present_names
            if absent_names:
                st.divider()
                st.write("**Not yet marked present:**")
                st.write(", ".join(sorted(absent_names)))
