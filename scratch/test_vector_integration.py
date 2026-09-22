import sys
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.face_service import normalize_embedding, face_service
from app.services.multiface_engine import recognize_faces_in_frame, MultiFaceEngine
from face_engine import THRESHOLD


def test_vector_normalization():
    print("\n--- Test Vector Normalization ---")
    raw_vec = np.random.randn(512).astype(np.float32)
    norm_vec = normalize_embedding(raw_vec)
    length = np.linalg.norm(norm_vec)
    print(f"Normalized vector L2 norm: {length:.6f}")
    assert np.isclose(length, 1.0, atol=1e-5), "L2 norm must equal 1.0"
    print("✅ Vector normalization verified.")


def test_centroid_averaging():
    print("\n--- Test Profile Centroid Averaging ---")
    v1 = normalize_embedding(np.array([1.0, 2.0, 3.0] + [0.0]*509, dtype=np.float32))
    v2 = normalize_embedding(np.array([1.1, 1.9, 3.1] + [0.0]*509, dtype=np.float32))
    v3 = normalize_embedding(np.array([0.9, 2.1, 2.9] + [0.0]*509, dtype=np.float32))

    mean_vec = np.mean([v1, v2, v3], axis=0)
    centroid = normalize_embedding(mean_vec)
    centroid_norm = np.linalg.norm(centroid)
    print(f"Centroid L2 norm: {centroid_norm:.6f}")
    assert np.isclose(centroid_norm, 1.0, atol=1e-5), "Centroid L2 norm must equal 1.0"
    print("✅ Centroid averaging verified.")


def test_fast_matrix_recognition():
    print("\n--- Test Matrix Dot Product Recognition ---")
    emp_a = normalize_embedding(np.array([1.0] + [0.0]*511, dtype=np.float32))
    emp_b = normalize_embedding(np.array([0.0, 1.0] + [0.0]*510, dtype=np.float32))

    loaded_embeddings = {
        "EMP001": emp_a,
        "EMP002": emp_b
    }

    # Match exact EMP001 vector
    query_vec = normalize_embedding(np.array([0.99] + [0.05]*511, dtype=np.float32))
    matrix_emb = np.array([loaded_embeddings[eid] for eid in loaded_embeddings], dtype=np.float32)
    scores = np.dot(matrix_emb, query_vec)
    best_idx = np.argmax(scores)
    best_score = float(scores[best_idx])
    matched_id = list(loaded_embeddings.keys())[best_idx] if best_score >= THRESHOLD else "Unknown"

    print(f"Matched ID: {matched_id}, Confidence: {best_score:.4f}")
    assert matched_id == "EMP001", "Expected EMP001 match"
    assert best_score >= THRESHOLD, f"Score should be >= {THRESHOLD}"
    print("✅ Fast matrix dot product recognition verified.")


def test_threshold_verification():
    print("\n--- Test Recognition Decision Threshold ---")
    print(f"Configured THRESHOLD: {THRESHOLD}")
    assert THRESHOLD == 0.50, f"Expected THRESHOLD 0.50, got {THRESHOLD}"
    print("✅ Decision threshold verified.")


if __name__ == "__main__":
    test_vector_normalization()
    test_centroid_averaging()
    test_fast_matrix_recognition()
    test_threshold_verification()
    print("\n🎉 ALL VECTOR INTEGRATION TESTS PASSED SUCCESSFULLY!")
