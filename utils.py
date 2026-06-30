"""
utils.py
Helper functions for loading/saving face embeddings and matching faces.
"""

import os
import pickle
import numpy as np

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.pkl")


def load_embeddings():
    """Load the known-faces embeddings database from disk.

    Returns a dict: { "person_name": [embedding1, embedding2, ...] }
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        return {}
    with open(EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)


def save_embeddings(db):
    """Save the known-faces embeddings database to disk."""
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(db, f)


def add_embedding(db, name, embedding):
    """Add a new embedding vector for `name` to the database (in-memory)."""
    embedding = np.asarray(embedding).flatten()
    if name not in db:
        db[name] = []
    db[name].append(embedding)
    return db


def cosine_distance(a, b):
    """Cosine distance between two vectors (0 = identical, 2 = opposite)."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return 1 - cos_sim


def identify_face(db, embedding, threshold=0.4):
    """Compare an embedding against all known embeddings in the database.

    For each known person, we use their *closest* stored embedding (a person
    can have several embeddings captured during enrollment for robustness).

    Returns (best_name, best_distance) if a match is found under `threshold`,
    otherwise ("Unknown", best_distance).
    """
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
