"""
Unit tests for YoloTillDetector service
"""

import numpy as np
from app.services.yolo_detector import YoloTillDetector, yolo_till_detector


def test_yolo_till_detector_initialization():
    detector = YoloTillDetector()
    assert detector is not None


def test_yolo_till_detector_empty_frame():
    result = yolo_till_detector.analyze_till_frame(None)
    assert isinstance(result, dict)
    assert "till_person_count" in result
    assert "spoof_risk_level" in result


def test_yolo_till_detector_synthetic_frame():
    # Synthetic BGR image 480x640x3
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add dummy white rectangle
    frame[100:300, 200:400] = 255

    fake_face_bbox = [[210, 110, 350, 280]]

    result = yolo_till_detector.analyze_till_frame(frame, fake_face_bbox)
    assert isinstance(result, dict)
    assert "till_person_count" in result
    assert "detected_devices" in result
    assert "device_near_face" in result
    assert "spoof_risk_level" in result


if __name__ == "__main__":
    print("Testing YoloTillDetector...")
    test_yolo_till_detector_initialization()
    test_yolo_till_detector_empty_frame()
    test_yolo_till_detector_synthetic_frame()
    print("YoloTillDetector tests passed successfully!")
