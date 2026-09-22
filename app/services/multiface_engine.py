import json
import logging
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.core.liveness_config import liveness_config
from app.services.face_attribute_engine import FaceAttributeEngine
from app.services.hand_detection_engine import HandDetectionEngine
from app.services.anti_spoof_engine import AntiSpoofEngine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = ROOT / "face_data" / "embeddings.json"


class MultiFaceEngine:

    def __init__(self):
        det_size = liveness_config.DETECTOR_SIZE
        logger.info("Loading MultiFace InsightFace engine with det_size=%s...", det_size)

        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(ctx_id=-1, det_size=det_size)
        except Exception as e:
            logger.warning("Could not initialize with allowed_modules: %s. Falling back to default loader.", e)
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=-1, det_size=det_size)

        # Auxiliary phase 4.5 engines
        self.attribute_engine = FaceAttributeEngine()
        self.hand_engine = HandDetectionEngine()
        self.anti_spoof_engine = AntiSpoofEngine()

        self.embeddings = self._load_embeddings()

        logger.info(
            "MultiFace engine loaded. Employees: %d",
            len(self.embeddings)
        )

    # =========================================================
    # LOAD EMBEDDINGS
    # =========================================================

    def _load_embeddings(self):
        if not EMBEDDINGS_PATH.exists():
            logger.warning("Embeddings file not found: %s", EMBEDDINGS_PATH)
            return {}

        try:
            with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)

            embeddings = {
                employee_id: np.array(embedding, dtype=np.float32)
                for employee_id, embedding in data.items()
            }
            logger.info("Loaded %d employee embeddings", len(embeddings))
            return embeddings
        except json.JSONDecodeError:
            logger.exception("Invalid embeddings JSON")
            return {}
        except OSError:
            logger.exception("Unable to read embeddings file")
            return {}
        except Exception:
            logger.exception("Unexpected error loading embeddings")
            return {}

    def reload_embeddings(self):
        self.embeddings = self._load_embeddings()
        logger.info("Embeddings reloaded: %d employees", len(self.embeddings))

    @staticmethod
    def cosine_similarity(a, b):
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a, b))

    def find_best_match(self, embedding, threshold: float = 0.50):
        if not self.embeddings or embedding is None:
            return None, 0.0

        emp_ids = list(self.embeddings.keys())
        matrix_embeddings = np.array([self.embeddings[emp_id] for emp_id in emp_ids], dtype=np.float32)

        norm = np.linalg.norm(embedding)
        norm_emb = (embedding / norm) if norm > 0 else embedding

        scores = np.dot(matrix_embeddings, norm_emb)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return emp_ids[best_idx], best_score
        return None, best_score


def recognize_faces_in_frame(frame, loaded_embeddings: dict, threshold: float = 0.50, app=None):
    """
    Detect and recognize all faces in a frame at once using fast vectorized dot product matching.
    """
    if app is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(960, 960))

    faces = app.get(frame)
    results = []

    emp_ids = list(loaded_embeddings.keys())
    if not emp_ids:
        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "employee_id": "Unknown",
                "confidence": 0.0
            })
        return results

    matrix_embeddings = np.array([loaded_embeddings[emp_id] for emp_id in emp_ids], dtype=np.float32)

    for face in faces:
        emb = getattr(face, "embedding", None)
        if emb is None:
            continue

        norm = np.linalg.norm(emb)
        norm_emb = emb / norm if norm > 0 else emb

        scores = np.dot(matrix_embeddings, norm_emb)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            matched_emp = emp_ids[best_idx]
        else:
            matched_emp = "Unknown"

        results.append({
            "bbox": face.bbox.astype(int).tolist(),
            "employee_id": matched_emp,
            "confidence": round(best_score, 4)
        })

    return results

    @staticmethod
    def estimate_head_pose(face) -> tuple[float, float, float]:
        """
        Estimate (yaw, pitch, roll) angles in degrees from facial landmarks (kps).
        kps: 0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth.
        """
        kps = getattr(face, "kps", None)
        pose = getattr(face, "pose", None)

        if pose is not None and len(pose) >= 3:
            return float(pose[0]), float(pose[1]), float(pose[2])

        if kps is None or len(kps) < 5:
            return 0.0, 0.0, 0.0

        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        left_mouth, right_mouth = kps[3], kps[4]

        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        inter_ocular = max(1.0, float(np.linalg.norm([eye_dx, eye_dy])))

        # Roll: eye tilt
        roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

        # Yaw: nose horizontal displacement from eye midpoint
        eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
        yaw = float(((nose[0] - eye_mid_x) / inter_ocular) * 90.0)

        # Pitch: nose vertical displacement from eye-mouth midpoint
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
        face_height = max(1.0, mouth_mid_y - eye_mid_y)
        expected_nose_y = eye_mid_y + (face_height * 0.45)
        pitch = float(((nose[1] - expected_nose_y) / face_height) * 90.0)

        return yaw, pitch, roll

    def analyze(self, image_bytes: bytes):
        self.reload_embeddings()

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            logger.warning("Invalid image supplied")
            return {
                "success": False,
                "message": "Invalid image",
                "faces": []
            }

        faces = self.app.get(image)
        if not faces:
            logger.info("No faces detected")
            return {
                "success": True,
                "message": "No faces detected",
                "faces": []
            }

        if len(faces) > liveness_config.MAX_FACES:
            logger.warning("Truncating faces to MAX_FACES=%d", liveness_config.MAX_FACES)
            faces = faces[:liveness_config.MAX_FACES]

        logger.info("Detected %d face(s)", len(faces))
        results = []

        h, w, _ = image.shape

        for index, face in enumerate(faces):
            face_result = {
                "face_index": index,
                "is_live": False,
                "live_score": 0.0,
                "anti_spoof_score": 0.0,
                "is_occluded": False,
                "occlusion_reason": None,
                "recognized": False,
                "employee_id": None,
                "confidence": 0.0,
                "bbox": None,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "quality_score": 0.0,
                "_embedding": None,
            }

            # Bounding box
            try:
                bbox = face.bbox
                face_result["bbox"] = [
                    round(float(bbox[0]), 2),
                    round(float(bbox[1]), 2),
                    round(float(bbox[2]), 2),
                    round(float(bbox[3]), 2),
                ]
            except Exception:
                logger.exception("Unable to read face bounding box")

            bbox_list = face_result["bbox"] or [0, 0, 0, 0]
            box_w = max(1.0, bbox_list[2] - bbox_list[0])
            box_h = max(1.0, bbox_list[3] - bbox_list[1])

            # Embedding (extracted for recognition & tracking)
            try:
                embedding = getattr(face, "embedding", None)
                if embedding is None and hasattr(self.app, "models") and "recognition" in self.app.models:
                    try:
                        self.app.models["recognition"].get(image, face)
                        embedding = getattr(face, "embedding", None)
                    except Exception:
                        pass

                if embedding is not None:
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                    face_result["_embedding"] = embedding
            except Exception:
                logger.exception("Embedding extraction failed for face %d", index)

            # Distance check: handle small faces
            if box_w < liveness_config.MIN_FACE_SIZE or box_h < liveness_config.MIN_FACE_SIZE:
                face_result["is_occluded"] = True
                face_result["occlusion_reason"] = f"Face too small ({int(box_w)}x{int(box_h)}px)"
                results.append(face_result)
                continue

            # Pose angle estimation
            yaw, pitch, roll = self.estimate_head_pose(face)
            face_result["yaw"] = round(yaw, 2)
            face_result["pitch"] = round(pitch, 2)
            face_result["roll"] = round(roll, 2)

            # Angle gate check: allow 3/4 & moderate profile up to MAX_YAW (60 degrees)
            if (abs(yaw) > liveness_config.MAX_YAW or
                abs(pitch) > liveness_config.MAX_PITCH or
                abs(roll) > liveness_config.MAX_ROLL):
                face_result["is_occluded"] = True
                face_result["occlusion_reason"] = (
                    f"Head angle too extreme (yaw={yaw:.1f}°, pitch={pitch:.1f}°). "
                    "Please turn slightly towards camera."
                )
                results.append(face_result)
                continue

            # Quality scoring: size + pose symmetry + sharpness
            size_score = min(1.0, (box_w * box_h) / (120.0 * 120.0))
            pose_score = max(0.0, 1.0 - (abs(yaw) / 90.0))
            face_result["quality_score"] = round(0.5 * size_score + 0.5 * pose_score, 4)

            # Distance crop-and-upscale for feature optimization if face is small (< 80px)
            proc_image = image
            proc_bbox = bbox_list
            if box_w < 80 or box_h < 80:
                pad_x = int(box_w * 0.2)
                pad_y = int(box_h * 0.2)
                cx1, cy1 = max(0, int(bbox_list[0]) - pad_x), max(0, int(bbox_list[1]) - pad_y)
                cx2, cy2 = min(w, int(bbox_list[2]) + pad_x), min(h, int(bbox_list[3]) + pad_y)
                crop = image[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    proc_image = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                    proc_bbox = [0, 0, 160, 160]

            # 1. Attribute Check (Mask/Scarf/Cloth/Sunglasses)
            attr_res = self.attribute_engine.analyze_face(proc_image, proc_bbox)
            if attr_res.is_occluded:
                face_result["is_occluded"] = True
                face_result["occlusion_reason"] = attr_res.reason
                face_result["is_live"] = False
                logger.info("Face %d rejected due to attributes: %s", index, attr_res.reason)
                results.append(face_result)
                continue

            # 2. Hand Check
            hand_res = self.hand_engine.check_hand_face_overlap(proc_image, proc_bbox)
            if hand_res.has_hand_overlap:
                face_result["is_occluded"] = True
                face_result["occlusion_reason"] = f"Hand covering face (ratio {hand_res.overlap_ratio:.2f})"
                face_result["is_live"] = False
                logger.info("Face %d rejected due to hand overlap", index)
                results.append(face_result)
                continue

            # 3. InsightFace Liveness
            insight_score = 0.0
            insight_live = False
            liveness_data = getattr(face, "liveness", None)
            if liveness_data is not None:
                if isinstance(liveness_data, dict):
                    insight_score = float(liveness_data.get("live_score", 0.0))
                    insight_live = bool(liveness_data.get("is_live", False))
                elif hasattr(liveness_data, "live_score"):
                    insight_score = float(getattr(liveness_data, "live_score", 0.0))
                    insight_live = bool(getattr(liveness_data, "is_live", False))

            face_result["live_score"] = round(insight_score, 4)

            # 4. Anti-Spoofing (MiniFASNet)
            pad_res = self.anti_spoof_engine.analyze_spoof(proc_image, proc_bbox)
            anti_spoof_score = pad_res["score"]
            face_result["anti_spoof_score"] = round(anti_spoof_score, 4)

            # Decision
            is_live = (
                (insight_score >= liveness_config.LIVENESS_THRESHOLD or insight_live) and
                (anti_spoof_score >= liveness_config.MINIFASNET_THRESHOLD)
            )
            face_result["is_live"] = is_live

            if not is_live:
                face_result["occlusion_reason"] = pad_res.get("reason", "Liveness / Anti-spoof check failed")
                results.append(face_result)
                continue

            # 5. Recognition
            if face_result["_embedding"] is not None:
                employee_id, score = self.find_best_match(face_result["_embedding"])
                score = float(score)
                face_result["confidence"] = round(score, 4)

                if employee_id is not None and score >= liveness_config.FACE_RECOGNITION_THRESHOLD:
                    face_result["recognized"] = True
                    face_result["employee_id"] = employee_id

            results.append(face_result)

        live_faces = sum(1 for face in results if face["is_live"])
        recognized_faces = sum(1 for face in results if face["recognized"])

        return {
            "success": True,
            "message": f"Detected {len(results)} face(s), {live_faces} live, {recognized_faces} recognized",
            "total_faces": len(results),
            "live_faces": live_faces,
            "recognized_faces": recognized_faces,
            "faces": results,
        }