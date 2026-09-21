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
        logger.info("Loading MultiFace InsightFace engine with addons=['liveness']...")

        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                addons=["liveness"],
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as e:
            logger.warning("Could not initialize with addons=['liveness']: %s. Falling back to base model.", e)
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))

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

    def find_best_match(self, embedding):
        best_employee = None
        best_score = -1.0

        for employee_id, stored_embedding in self.embeddings.items():
            score = self.cosine_similarity(embedding, stored_embedding)
            if score > best_score:
                best_score = score
                best_employee = employee_id

        return best_employee, best_score

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

            # Embedding (extracted before liveness for tracking)
            try:
                embedding = face.embedding
                if embedding is not None:
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                    face_result["_embedding"] = embedding
            except Exception:
                logger.exception("Embedding extraction failed for face %d", index)

            bbox_list = face_result["bbox"] or [0, 0, 0, 0]

            # 1. Attribute Check (Mask/Sunglasses)
            attr_res = self.attribute_engine.analyze_face(image, bbox_list)
            if attr_res.is_occluded:
                face_result["is_occluded"] = True
                face_result["occlusion_reason"] = attr_res.reason
                face_result["is_live"] = False
                logger.info("Face %d rejected due to attributes: %s", index, attr_res.reason)
                results.append(face_result)
                continue

            # 2. Hand Check
            hand_res = self.hand_engine.check_hand_face_overlap(image, bbox_list)
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
            pad_res = self.anti_spoof_engine.analyze_spoof(image, bbox_list)
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