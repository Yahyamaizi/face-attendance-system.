"""
fix_enroll.py
One-time helper script: automatically patches streamlit_app/app.py to fix
the enrollment tensor-shape bug (mtcnn with keep_all=True already returns
a batched tensor, so we must not add an extra unsqueeze on top of it).

Usage (run from the face_attendance folder):
    python fix_enroll.py
"""

import os

APP_PATH = os.path.join("streamlit_app", "app.py")

OLD_BLOCK = '''            img = Image.open(enroll_photo).convert("RGB")
            face_tensor = mtcnn(img)
            if face_tensor is None:
                st.error("No face detected in this photo — try again.")
            else:
                with torch.no_grad():
                    embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()'''

NEW_BLOCK = '''            img = Image.open(enroll_photo).convert("RGB")
            face_tensor = mtcnn(img)
            if face_tensor is None or len(face_tensor) == 0:
                st.error("No face detected in this photo — try again.")
            else:
                first_face = face_tensor[0]
                with torch.no_grad():
                    embedding = resnet(first_face.unsqueeze(0).to(DEVICE)).cpu().numpy()'''


def main():
    if not os.path.exists(APP_PATH):
        print(f"[ERROR] Could not find {APP_PATH}. Run this script from the face_attendance folder.")
        return

    with open(APP_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if NEW_BLOCK in content:
        print("[OK] The fix is already applied. Nothing to do.")
        return

    if OLD_BLOCK not in content:
        print("[WARN] Could not find the exact old code block to replace.")
        print("The file may have been edited manually already. Please check app.py by hand.")
        return

    content = content.replace(OLD_BLOCK, NEW_BLOCK)

    with open(APP_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("[SUCCESS] app.py has been patched with the enrollment fix!")


if __name__ == "__main__":
    main()
