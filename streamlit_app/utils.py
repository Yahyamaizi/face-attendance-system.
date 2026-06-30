"""
utils.py
Helper functions for loading/saving face embeddings and matching faces.
(Same logic as the desktop version, kept self-contained for deployment.)
"""

import os
import pickle
import numpy as np

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.pkl")


def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        return {}
    with open(EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)


def save_embeddings(db):
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(db, f)


def add_embedding(db, name, embedding):
    embedding = np.asarray(embedding).flatten()
    if name not in db:
        db[name] = []
    db[name].append(embedding)
    return db


def cosine_distance(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return 1 - cos_sim


def identify_face(db, embedding, threshold=0.4):
    if not db:
        return "Unknown", None

    best_name = "Unknown"
    best_distance = float("inf")

    for name, embeddings_list in db.items():
        for known_emb in embeddings_list:
            dist = cosine_distance(embedding, known_emb)
            if dist < best_distance:
                best_distance = dist
                best_name = name

    if best_distance > threshold:
        return "Unknown", best_distance

    return best_name, best_distance
