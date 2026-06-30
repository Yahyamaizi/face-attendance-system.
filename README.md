# Face Attendance System — Presence Detection with Deep Learning

A real-time facial recognition attendance system built with **MTCNN** (face
detection) and **FaceNet / InceptionResnetV1** (face recognition via transfer
learning, pretrained on VGGFace2). Built for use in VSCode with Python.

## How it works

1. **Detection** — MTCNN locates faces in each webcam frame.
2. **Recognition** — Each detected face is converted into a 512-dimension
   embedding vector by FaceNet (a model pretrained on millions of faces —
   this is the "transfer learning" part, you're reusing a model trained on
   VGGFace2 rather than training from scratch).
3. **Matching** — The embedding is compared (cosine distance) against the
   embeddings of known people stored in `embeddings.pkl`. The closest match
   under a distance threshold is the identified person.
4. **Logging** — The first time a known person is seen each day, their name,
   date, and time are appended to a CSV file in `attendance_logs/`.

## Project structure

```
face_attendance/
├── enroll_faces.py        # Step 1: register people (capture face samples)
├── attendance_system.py   # Step 2: live detection + recognition + logging
├── utils.py                # Embedding storage & matching helpers
├── requirements.txt
├── known_faces/             # Saved reference photos per enrolled person
├── attendance_logs/         # Daily CSV attendance logs (auto-created)
└── embeddings.pkl           # Face embedding database (auto-created)
```

## Setup (VSCode)

1. Open this folder in VSCode (`File > Open Folder...`).
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. In VSCode, select this venv as your Python interpreter
   (`Ctrl+Shift+P` → "Python: Select Interpreter").
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > First run will also download the pretrained FaceNet weights
   > automatically (~100MB, needs internet once).

## Usage

### 1. Enroll people
Run once per person you want the system to recognize:
```bash
python enroll_faces.py --name "Yahya Alaoui" --samples 15
```
Look at the camera and press `c` to capture each sample (try a few angles
and expressions), `q` to stop early.

### 2. Run the attendance system
```bash
python attendance_system.py
```
A webcam window opens showing live bounding boxes and names. Green box =
recognized, red box = unknown. Press `q` to quit. Each person is logged
once per day to `attendance_logs/attendance_YYYY-MM-DD.csv`.

## Tuning

- `RECOGNITION_THRESHOLD` in `attendance_system.py` (default `0.4`) controls
  strictness: lower = fewer false positives but more missed matches, higher
  = more lenient. Adjust based on your testing.
- Enroll more samples per person (15–25) in varied lighting for better
  robustness.

## Ideas for extending it (good for the "gestion de projet" report)

- Add a Tkinter/PyQt dashboard to view attendance stats.
- Export weekly/monthly attendance summaries (pandas + matplotlib).
- Add liveness detection (blink detection) to prevent photo spoofing.
- Store embeddings in a real database (SQLite) instead of a pickle file.
- Add email/Slack notification on late arrival.

## Suggested project management framing

Since this is for a gestion de projet module, you could structure your
report around:
- **Scope**: presence detection via facial recognition (problem statement)
- **Architecture**: MTCNN + FaceNet pipeline diagram
- **Planning**: enrollment phase → detection phase → logging/reporting phase
- **Risks**: lighting conditions, spoofing, privacy/GDPR considerations
- **Deliverables & milestones**: enrollment script → recognition engine →
  logging → (optional) dashboard
