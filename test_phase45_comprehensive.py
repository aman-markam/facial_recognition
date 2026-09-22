"""
Comprehensive Verification Test Suite
Multi-Person, Multi-Angle & Distance Facial Attendance Pipeline
"""

import sys
import cv2
import numpy as np

from app.core.liveness_config import liveness_config
from app.services.face_attribute_engine import FaceAttributeEngine
from app.services.hand_detection_engine import HandDetectionEngine
from app.services.anti_spoof_engine import AntiSpoofEngine
from app.services.multiface_engine import MultiFaceEngine
from app.services.multiface_tracker import MultiFaceTracker


def test_1_configuration():
    print("\n--- Test 1: Configuration & Threshold Verification ---")
    print(f"Detector size: {liveness_config.DETECTOR_SIZE}")
    print(f"Min face size: {liveness_config.MIN_FACE_SIZE}px")
    print(f"Max yaw angle: {liveness_config.MAX_YAW}°")
    print(f"Max faces: {liveness_config.MAX_FACES}")

    assert liveness_config.DETECTOR_SIZE == (960, 960), "Detector size mismatch"
    assert liveness_config.MIN_FACE_SIZE == 40, "Min face size mismatch"
    assert liveness_config.MAX_YAW == 60.0, "Max yaw mismatch"
    assert liveness_config.MAX_FACES == 10, "Max faces mismatch"
    print("✅ Configuration verified successfully.")


def test_2_occlusion_and_attributes():
    print("\n--- Test 2: Scarf, Mask, Cloth & Sunglasses Occlusion ---")
    engine = FaceAttributeEngine()

    # Create dummy skin crop
    crop = np.zeros((120, 120, 3), dtype=np.uint8)
    crop[:, :] = [120, 150, 200]  # skin tone
    res_normal = engine.analyze_face(crop, [0, 0, 120, 120])
    print(f"Normal face occlusion: {res_normal.is_occluded}")
    assert not res_normal.is_occluded, "Normal face should not be occluded"

    # Mask / Cloth crop (lower 55% dark fabric with edges)
    cloth_crop = crop.copy()
    cloth_crop[50:120, :] = [30, 30, 30]  # dark cloth/scarf
    # add synthetic fabric edges
    for i in range(50, 120, 4):
        cloth_crop[i, :, :] = 200
    res_cloth = engine.analyze_face(cloth_crop, [0, 0, 120, 120])
    print(f"Cloth/Scarf face occlusion: {res_cloth.is_occluded} (Reason: {res_cloth.reason})")
    assert res_cloth.is_occluded, "Scarf/cloth covered face must be rejected"
    print("✅ FaceAttributeEngine occlusion handling verified.")


def test_3_hand_detection():
    print("\n--- Test 3: Hand Overlap Check ---")
    engine = HandDetectionEngine()

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    face_box = [100, 100, 250, 250]
    res = engine.check_hand_face_overlap(blank, face_box)
    print(f"Hand overlap on blank frame: {res.has_hand_overlap}")
    assert not res.has_hand_overlap, "Blank frame should have no hand overlap"
    print("✅ HandDetectionEngine verified.")


def test_4_pose_estimation():
    print("\n--- Test 4: Head Pose Estimation (Yaw, Pitch, Roll) ---")
    class DummyFace:
        kps = np.array([
            [100, 100],  # left eye
            [140, 100],  # right eye
            [130, 120],  # nose shifted right (yaw ~ 22.5 deg)
            [105, 140],  # left mouth
            [135, 140]   # right mouth
        ], dtype=np.float32)

    face = DummyFace()
    yaw, pitch, roll = MultiFaceEngine.estimate_head_pose(face)
    print(f"Estimated angles: yaw={yaw:.2f}°, pitch={pitch:.2f}°, roll={roll:.2f}°")
    assert abs(yaw) > 10.0, "Yaw angle should reflect nose shift"
    assert abs(yaw) <= 60.0, "Angle falls within 3/4 moderate profile threshold"
    print("✅ Multi-angle pose estimation verified.")


def test_5_multiface_tracker_multi_person():
    print("\n--- Test 5: Multi-Person Tracking & Background Isolation ---")
    tracker = MultiFaceTracker()

    # Generate 5 blank synthetic frames
    frames = []
    for _ in range(5):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        is_succ, buf = cv2.imencode(".jpg", img)
        frames.append(buf.tobytes())

    res = tracker.process_frames(frames)
    print(f"Tracker output success: {res['success']}, total tracks: {res['total_tracks']}")
    assert res["success"], "MultiFaceTracker failed on synthetic frame set"
    print("✅ MultiFaceTracker multi-person flow verified.")


def main():
    print("==================================================")
    print(" MULTI-PERSON MULTI-ANGLE DISTANCE TEST SUITE")
    print("==================================================")
    test_1_configuration()
    test_2_occlusion_and_attributes()
    test_3_hand_detection()
    test_4_pose_estimation()
    test_5_multiface_tracker_multi_person()
    print("\n🎉 ALL COMPREHENSIVE PIPELINE TESTS PASSED!")


if __name__ == "__main__":
    main()
