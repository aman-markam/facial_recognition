"""
YOLO Till & Anti-Spoofing Detector
Detects persons at till counters, electronic devices (cell phones, tablets, screens),
and potential presentation attack items near face bounding boxes.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# COCO Class ID mappings relevant for Till & Anti-Spoofing
COCO_PERSON_CLASS_ID = 0
SPOOF_DEVICE_CLASS_IDS = {
    67: "cell phone",
    63: "laptop",
    62: "tv/screen/tablet",
    73: "book/paper"
}


class YoloTillDetector:
    """
    YOLO-based Object and Anti-Spoofing Detector for Till environments.
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.40):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self._is_initialized = False

        self._initialize_model()

    def _initialize_model(self):
        """Attempts lazy initialization of Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
            logger.info("Initializing YOLO Till Detector with model '%s'...", self.model_name)
            self.model = YOLO(self.model_name)
            self._is_initialized = True
            logger.info("YOLO Till Detector successfully loaded.")
        except ImportError:
            logger.warning(
                "Ultralytics package not installed. YOLO Till Detector will operate in fallback mode."
            )
            self._is_initialized = False
        except Exception as e:
            logger.error("Failed to initialize YOLO model '%s': %s", self.model_name, e)
            self._is_initialized = False

    def analyze_till_frame(
        self,
        image: np.ndarray,
        face_bboxes: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a till camera frame for persons, suspicious electronic devices, and spoof risks.

        Args:
            image: BGR numpy image frame.
            face_bboxes: Optional list of detected face bounding boxes [[x1, y1, x2, y2], ...]

        Returns:
            Dict containing till person metrics, device detections, and spoof risk alerts.
        """
        if not self._is_initialized or self.model is None:
            return {
                "yolo_active": False,
                "till_person_count": 0,
                "persons": [],
                "detected_devices": [],
                "device_near_face": False,
                "spoof_risk_level": "LOW",
                "warning_message": "YOLO detector not active"
            }

        if image is None or len(image.shape) != 3:
            return {
                "yolo_active": True,
                "till_person_count": 0,
                "persons": [],
                "detected_devices": [],
                "device_near_face": False,
                "spoof_risk_level": "LOW",
                "warning_message": "Invalid frame received"
            }

        try:
            # Perform inference
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            persons = []
            detected_devices = []
            device_near_face = False
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [float(val) for val in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]

                    if cls_id == COCO_PERSON_CLASS_ID:
                        persons.append({
                            "bbox": xyxy,
                            "confidence": round(conf, 3)
                        })

                    elif cls_id in SPOOF_DEVICE_CLASS_IDS:
                        device_name = SPOOF_DEVICE_CLASS_IDS[cls_id]
                        device_info = {
                            "class_id": cls_id,
                            "label": device_name,
                            "confidence": round(conf, 3),
                            "bbox": xyxy
                        }
                        detected_devices.append(device_info)

                        # Check if device overlaps or is near any detected face
                        if face_bboxes:
                            for face_box in face_bboxes:
                                if self._check_bbox_proximity(xyxy, face_box):
                                    device_near_face = True
                                    device_info["near_face"] = True

            # Determine spoof risk level
            spoof_risk_level = "LOW"
            warning_message = None

            if device_near_face:
                spoof_risk_level = "HIGH"
                warning_message = "SUSPICIOUS DEVICE NEAR FACE: Possible phone/screen spoofing attempt detected at till."
            elif len(detected_devices) > 0:
                spoof_risk_level = "MEDIUM"
                warning_message = f"Electronic device ({detected_devices[0]['label']}) detected in till frame."
            elif len(persons) == 0:
                warning_message = "No person detected at till."

            return {
                "yolo_active": True,
                "till_person_count": len(persons),
                "persons": persons,
                "detected_devices": detected_devices,
                "device_near_face": device_near_face,
                "spoof_risk_level": spoof_risk_level,
                "warning_message": warning_message
            }

        except Exception as e:
            logger.error("Error running YOLO till frame analysis: %s", e)
            return {
                "yolo_active": True,
                "till_person_count": 0,
                "persons": [],
                "detected_devices": [],
                "device_near_face": False,
                "spoof_risk_level": "UNKNOWN",
                "warning_message": f"Analysis error: {str(e)}"
            }

    @staticmethod
    def _check_bbox_proximity(
        device_box: List[float],
        face_box: List[float],
        margin: float = 50.0
    ) -> bool:
        """
        Determines if an electronic device bounding box overlaps or is within
        a margin of pixels from the face bounding box.
        """
        dx1, dy1, dx2, dy2 = device_box[:4]
        fx1, fy1, fx2, fy2 = face_box[:4]

        # Expand face box by margin
        efx1 = fx1 - margin
        efy1 = fy1 - margin
        efx2 = fx2 + margin
        efy2 = fy2 + margin

        # Intersection check
        inter_x1 = max(dx1, efx1)
        inter_y1 = max(dy1, efy1)
        inter_x2 = min(dx2, efx2)
        inter_y2 = min(dy2, efy2)

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            return True
        return False


# Global singleton instance for easy import across app services
yolo_till_detector = YoloTillDetector()
