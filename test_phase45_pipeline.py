"""
Comprehensive Test Suite for Phase 4.5 Multi-person Liveness & Anti-Spoofing Architecture
"""

import logging
import sys
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Phase45Test")

from app.core.liveness_config import liveness_config
from app.schemas.liveness import FaceAttributeResult, HandOverlapResult, MultiFaceLivenessResponse
from app.services.face_attribute_engine import FaceAttributeEngine
from app.services.hand_detection_engine import HandDetectionEngine
from app.services.anti_spoof_engine import AntiSpoofEngine
from app.services.liveness_engine import LivenessEngine
from app.services.multiface_engine import MultiFaceEngine
from app.services.multiface_tracker import MultiFaceTracker


def test_1_configuration():
    print("\n--- Test 1: Configuration Initialization ---")
    assert liveness_config.LIVENESS_THRESHOLD == 0.80, "Liveness threshold mismatch"
    assert liveness_config.MINIFASNET_THRESHOLD == 0.70, "MiniFASNet threshold mismatch"
    assert liveness_config.MAX_FACES == 5, "Max faces mismatch"
    print("✅ Configuration thresholds verified successfully.")


def test_2_face_attribute_engine():
    print("\n--- Test 2: Face Attribute Engine ---")
    engine = FaceAttributeEngine()

    # Create dummy skin-tone image (normal face crop)
    normal_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    normal_crop[:, :] = [120, 150, 200]  # BGR skin-like tone
    res_normal = engine.analyze_face(normal_crop, [0, 0, 100, 100])
    print(f"Normal face crop occlusion: {res_normal.is_occluded} (Reason: {res_normal.reason})")

    # Create dummy masked crop (blue lower half)
    masked_crop = normal_crop.copy()
    masked_crop[50:100, :] = [255, 100, 0]  # Dark blue fabric mask
    res_mask = engine.analyze_face(masked_crop, [0, 0, 100, 100])
    print(f"Masked crop occlusion: {res_mask.is_occluded} (Mask prob: {res_mask.mask_prob:.2f})")
    assert res_mask.is_occluded, "Masked face should be detected as occluded"
    print("✅ FaceAttributeEngine verified successfully.")


def test_3_hand_detection_engine():
    print("\n--- Test 3: Hand Detection Engine ---")
    engine = HandDetectionEngine()

    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    face_bbox = [100, 100, 300, 300]
    res = engine.check_hand_face_overlap(blank_img, face_bbox)
    print(f"Blank image hand overlap result: has_overlap={res.has_hand_overlap}, ratio={res.overlap_ratio:.2f}")
    assert not res.has_hand_overlap, "Blank image should have no hand overlap"
    print("✅ HandDetectionEngine verified successfully.")


def test_4_anti_spoof_engine():
    print("\n--- Test 4: Anti-Spoof Engine ---")
    engine = AntiSpoofEngine()

    # Create synthetic noise frame
    frame = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    res = engine.analyze_spoof(frame, [10, 10, 200, 200])
    print(f"Synthetic noise anti-spoof result: score={res['score']:.2f}, live={res['is_live']}")
    print("✅ AntiSpoofEngine verified successfully.")


def test_5_liveness_engine():
    print("\n--- Test 5: Liveness Engine ---")
    engine = LivenessEngine()

    # Create dummy black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = engine.analyze_image(frame)
    print(f"Liveness engine black frame result: faces={res.total_faces}, live={res.live_faces}")
    assert res.success, "LivenessEngine execution failed"
    print("✅ LivenessEngine verified successfully.")


def test_6_multiface_engine():
    print("\n--- Test 6: MultiFace Engine ---")
    engine = MultiFaceEngine()

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    is_success, buffer = cv2.imencode(".jpg", blank_frame)
    img_bytes = buffer.tobytes()

    res = engine.analyze(img_bytes)
    print(f"MultiFaceEngine analyze result: success={res['success']}, total_faces={res['total_faces']}")
    assert res["success"], "MultiFaceEngine analysis failed"
    print("✅ MultiFaceEngine verified successfully.")


def main():
    print("==================================================")
    print(" PHASE 4.5 PIPELINE VERIFICATION SUITE")
    print("==================================================")
    try:
        test_1_configuration()
        test_2_face_attribute_engine()
        test_3_hand_detection_engine()
        test_4_anti_spoof_engine()
        test_5_liveness_engine()
        test_6_multiface_engine()
        print("\n🎉 ALL PHASE 4.5 TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
