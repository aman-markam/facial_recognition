# import logging
# from typing import Any

# import numpy as np

# from app.services.multiface_engine import MultiFaceEngine


# logger = logging.getLogger(__name__)


# REQUIRED_FRAMES = 5

# # A person must be live in at least 4 of 5 frames.
# MIN_LIVE_FRAMES = 4

# # This is ONLY for tracking the same person between frames.
# #
# # It is NOT the face-recognition threshold.
# # Face recognition remains 0.55.
# TRACKING_THRESHOLD = 0.70


# class MultiFaceTracker:

#     def __init__(self):

#         self.engine = MultiFaceEngine()

#     # =========================================================
#     # COSINE SIMILARITY
#     # =========================================================

#     @staticmethod
#     def cosine_similarity(a, b):

#         a = a / (
#             np.linalg.norm(a) + 1e-10
#         )

#         b = b / (
#             np.linalg.norm(b) + 1e-10
#         )

#         return float(
#             np.dot(a, b)
#         )

#     # =========================================================
#     # FIND MATCHING TRACK
#     # =========================================================

#     def _find_matching_track(
#         self,
#         embedding,
#         tracks,
#         frame_index,
#         face_index
#     ):

#         best_track = None
#         best_similarity = -1.0

#         for track in tracks:

#             for observation in track["observations"]:

#                 previous_embedding = observation.get(
#                     "_embedding"
#                 )

#                 if previous_embedding is None:
#                     continue

#                 similarity = self.cosine_similarity(
#                     embedding,
#                     previous_embedding
#                 )

#                 logger.info(
#                     "Tracking comparison: "
#                     "frame=%d face=%s "
#                     "track=%d similarity=%.4f",
#                     frame_index,
#                     face_index,
#                     track["track_id"],
#                     similarity
#                 )

#                 if similarity > best_similarity:

#                     best_similarity = similarity
#                     best_track = track

#         if (
#             best_track is not None
#             and best_similarity >= TRACKING_THRESHOLD
#         ):

#             logger.info(
#                 "MATCHED: frame=%d face=%s "
#                 "-> track=%d similarity=%.4f",
#                 frame_index,
#                 face_index,
#                 best_track["track_id"],
#                 best_similarity
#             )

#             return (
#                 best_track,
#                 best_similarity
#             )

#         logger.info(
#             "NEW TRACK: frame=%d face=%s "
#             "best_similarity=%.4f",
#             frame_index,
#             face_index,
#             best_similarity
#         )

#         return (
#             None,
#             best_similarity
#         )

#     # =========================================================
#     # PROCESS FIVE FRAMES
#     # =========================================================

#     def process_frames(
#         self,
#         image_frames: list[bytes]
#     ):

#         # =====================================================
#         # VALIDATE FRAME COUNT
#         # =====================================================

#         if len(image_frames) != REQUIRED_FRAMES:

#             return {
#                 "success": False,
#                 "message": (
#                     f"Exactly {REQUIRED_FRAMES} "
#                     "frames are required"
#                 ),
#                 "total_tracks": 0,
#                 "eligible_employees": 0,
#                 "tracks": [],
#                 "employees": [],
#             }

#         tracks: list[
#             dict[str, Any]
#         ] = []

#         # =====================================================
#         # PROCESS EACH FRAME
#         # =====================================================

#         for frame_index, image_bytes in enumerate(
#             image_frames
#         ):

#             logger.info(
#                 "===================================="
#             )

#             logger.info(
#                 "Processing frame %d/%d",
#                 frame_index + 1,
#                 REQUIRED_FRAMES
#             )

#             result = self.engine.analyze(
#                 image_bytes
#             )

#             if not result.get(
#                 "success",
#                 False
#             ):

#                 logger.warning(
#                     "Frame %d failed: %s",
#                     frame_index + 1,
#                     result.get(
#                         "message"
#                     )
#                 )

#                 continue

#             faces = result.get(
#                 "faces",
#                 []
#             )

#             logger.info(
#                 "Frame %d detected %d face(s)",
#                 frame_index + 1,
#                 len(faces)
#             )

#             # =================================================
#             # PROCESS EVERY FACE
#             # =================================================

#             for face in faces:

#                 embedding = face.get(
#                     "_embedding"
#                 )

#                 if embedding is None:

#                     logger.warning(
#                         "Face %s has no embedding",
#                         face.get(
#                             "face_index"
#                         )
#                     )

#                     continue

#                 # -------------------------------------------------
#                 # Find an existing track
#                 # -------------------------------------------------

#                 matched_track, similarity = (
#                     self._find_matching_track(
#                         embedding,
#                         tracks,
#                         frame_index,
#                         face.get("face_index")
#                     )
#                 )

#                 # -------------------------------------------------
#                 # Create new track if necessary
#                 # -------------------------------------------------

#                 if matched_track is None:

#                     matched_track = {
#                         "track_id": len(
#                             tracks
#                         ),
#                         "observations": [],
#                     }

#                     tracks.append(
#                         matched_track
#                     )

#                     logger.info(
#                         "Created new track %d",
#                         matched_track[
#                             "track_id"
#                         ]
#                     )

#                 # -------------------------------------------------
#                 # Store observation
#                 # -------------------------------------------------

#                 observation = {
#                     "frame_index": frame_index,
#                     "face_index": face.get(
#                         "face_index"
#                     ),

#                     "is_live": bool(
#                         face.get(
#                             "is_live",
#                             False
#                         )
#                     ),

#                     "live_score": float(
#                         face.get(
#                             "live_score",
#                             0.0
#                         )
#                     ),

#                     "recognized": bool(
#                         face.get(
#                             "recognized",
#                             False
#                         )
#                     ),

#                     "employee_id": face.get(
#                         "employee_id"
#                     ),

#                     "confidence": float(
#                         face.get(
#                             "confidence",
#                             0.0
#                         )
#                     ),

#                     "bbox": face.get(
#                         "bbox"
#                     ),

#                     # Internal value used by tracking.
#                     "_embedding": embedding,
#                 }

#                 matched_track[
#                     "observations"
#                 ].append(
#                     observation
#                 )

#         # =====================================================
#         # AGGREGATE TRACKS
#         # =====================================================

#         final_tracks = []

#         for track in tracks:

#             observations = track[
#                 "observations"
#             ]

#             if not observations:
#                 continue

#             # =================================================
#             # LIVENESS
#             # =================================================

#             live_observations = [
#                 observation
#                 for observation in observations
#                 if observation[
#                     "is_live"
#                 ]
#             ]

#             live_frames = len(
#                 live_observations
#             )

#             passed_liveness = (
#                 live_frames
#                 >= MIN_LIVE_FRAMES
#             )

#             # =================================================
#             # RECOGNITION VOTING
#             # =================================================

#             recognized_observations = [
#                 observation
#                 for observation in observations
#                 if (
#                     observation[
#                         "recognized"
#                     ]
#                     and observation[
#                         "employee_id"
#                     ] is not None
#                 )
#             ]

#             # -------------------------------------------------
#             # Count employee votes
#             # -------------------------------------------------

#             employee_votes = {}

#             for observation in (
#                 recognized_observations
#             ):

#                 employee_id = observation[
#                     "employee_id"
#                 ]

#                 employee_votes[
#                     employee_id
#                 ] = employee_votes.get(
#                     employee_id,
#                     0
#                 ) + 1

#             employee_id = None

#             if employee_votes:

#                 employee_id = max(
#                     employee_votes,
#                     key=employee_votes.get
#                 )

#             # =================================================
#             # RECOGNITION CONFIDENCE
#             # =================================================

#             confidence_values = [
#                 observation[
#                     "confidence"
#                 ]
#                 for observation in (
#                     recognized_observations
#                 )
#                 if observation[
#                     "confidence"
#                 ] > 0
#             ]

#             if confidence_values:

#                 recognition_confidence = (
#                     float(
#                         np.mean(
#                             confidence_values
#                         )
#                     )
#                 )

#             else:

#                 recognition_confidence = 0.0

#             # =================================================
#             # LIVENESS AVERAGE
#             # =================================================

#             liveness_values = [
#                 observation[
#                     "live_score"
#                 ]
#                 for observation in observations
#                 if observation[
#                     "live_score"
#                 ] > 0
#             ]

#             if liveness_values:

#                 liveness_average = (
#                     float(
#                         np.mean(
#                             liveness_values
#                         )
#                     )
#                 )

#             else:

#                 liveness_average = 0.0

#             # =================================================
#             # FINAL RECOGNITION STATE
#             # =================================================

#             recognized = (
#                 employee_id is not None
#             )

#             # =================================================
#             # ATTENDANCE ELIGIBILITY
#             # =================================================

#             eligible_for_attendance = (
#                 passed_liveness
#                 and recognized
#             )

#             final_tracks.append(
#                 {
#                     "track_id": track[
#                         "track_id"
#                     ],

#                     "employee_id": employee_id,

#                     "total_observations": len(
#                         observations
#                     ),

#                     "live_frames": live_frames,

#                     "total_frames": (
#                         REQUIRED_FRAMES
#                     ),

#                     "passed_liveness": (
#                         passed_liveness
#                     ),

#                     "recognized": recognized,

#                     "recognition_confidence": (
#                         round(
#                             recognition_confidence,
#                             4
#                         )
#                     ),

#                     "liveness_average": (
#                         round(
#                             liveness_average,
#                             4
#                         )
#                     ),

#                     "eligible_for_attendance": (
#                         eligible_for_attendance
#                     ),

#                     "employee_votes": (
#                         employee_votes
#                     ),

#                     "observations": observations,
#                 }
#             )

#         # =====================================================
#         # DEDUPLICATE EMPLOYEES
#         # =====================================================

#         unique_employees = {}

#         for track in final_tracks:

#             employee_id = track.get(
#                 "employee_id"
#             )

#             if employee_id is None:
#                 continue

#             if not track[
#                 "eligible_for_attendance"
#             ]:
#                 continue

#             existing = unique_employees.get(
#                 employee_id
#             )

#             # First occurrence.
#             if existing is None:

#                 unique_employees[
#                     employee_id
#                 ] = track

#                 continue

#             # If duplicate employee appears in
#             # multiple tracks, keep the track
#             # with stronger recognition.
#             if (
#                 track[
#                     "recognition_confidence"
#                 ]
#                 >
#                 existing[
#                     "recognition_confidence"
#                 ]
#             ):

#                 unique_employees[
#                     employee_id
#                 ] = track

#         employees = list(
#             unique_employees.values()
#         )

#         # =====================================================
#         # REMOVE INTERNAL EMBEDDINGS
#         # =====================================================

#         for track in final_tracks:

#             for observation in track[
#                 "observations"
#             ]:

#                 observation.pop(
#                     "_embedding",
#                     None
#                 )

#         # =====================================================
#         # FINAL RESULT
#         # =====================================================

#         logger.info(
#             "Multi-face tracking completed: "
#             "tracks=%d eligible=%d",
#             len(final_tracks),
#             len(employees)
#         )

#         return {
#             "success": True,

#             "message": (
#                 f"Processed {REQUIRED_FRAMES} "
#                 "frames"
#             ),

#             "total_tracks": len(
#                 final_tracks
#             ),

#             "eligible_employees": len(
#                 employees
#             ),

#             "tracks": final_tracks,

#             "employees": employees,
#         }

import logging
from typing import Any

import numpy as np

from app.services.multiface_engine import MultiFaceEngine


logger = logging.getLogger(__name__)


REQUIRED_FRAMES = 5
MIN_LIVE_FRAMES = 4

# Face embedding similarity.
EMBEDDING_TRACK_THRESHOLD = 0.60

# Maximum allowed movement of the face center,
# normalized by the face size.
POSITION_TRACK_THRESHOLD = 1.50


class MultiFaceTracker:

    def __init__(self):
        self.engine = MultiFaceEngine()

    # =========================================================
    # COSINE SIMILARITY
    # =========================================================

    @staticmethod
    def cosine_similarity(a, b):

        a = a / (
            np.linalg.norm(a) + 1e-10
        )

        b = b / (
            np.linalg.norm(b) + 1e-10
        )

        return float(
            np.dot(a, b)
        )

    # =========================================================
    # BBOX CENTER
    # =========================================================

    @staticmethod
    def bbox_center(bbox):

        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox

        center_x = (
            float(x1) + float(x2)
        ) / 2.0

        center_y = (
            float(y1) + float(y2)
        ) / 2.0

        width = max(
            float(x2) - float(x1),
            1.0
        )

        height = max(
            float(y2) - float(y1),
            1.0
        )

        return (
            center_x,
            center_y,
            width,
            height
        )

    # =========================================================
    # POSITION SIMILARITY
    # =========================================================

    @classmethod
    def position_distance(
        cls,
        bbox_a,
        bbox_b
    ):

        a = cls.bbox_center(bbox_a)
        b = cls.bbox_center(bbox_b)

        if a is None or b is None:
            return None

        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        distance = np.sqrt(
            (ax - bx) ** 2
            + (ay - by) ** 2
        )

        face_size = max(
            (aw + ah) / 2.0,
            (bw + bh) / 2.0,
            1.0
        )

        return float(
            distance / face_size
        )

    # =========================================================
    # FIND MATCHING TRACK
    # =========================================================

    def _find_matching_track(
        self,
        face,
        tracks,
        frame_index
    ):

        embedding = face.get(
            "_embedding"
        )

        bbox = face.get(
            "bbox"
        )

        if embedding is None:
            return None, None

        best_track = None
        best_score = -1.0

        for track in tracks:

            observations = track[
                "observations"
            ]

            if not observations:
                continue

            # Compare against the most recent observation.
            previous = observations[-1]

            previous_embedding = previous.get(
                "_embedding"
            )

            previous_bbox = previous.get(
                "bbox"
            )

            if previous_embedding is None:
                continue

            # -------------------------------------------------
            # Embedding similarity
            # -------------------------------------------------

            embedding_similarity = (
                self.cosine_similarity(
                    embedding,
                    previous_embedding
                )
            )

            # -------------------------------------------------
            # Position distance
            # -------------------------------------------------

            position_distance = (
                self.position_distance(
                    bbox,
                    previous_bbox
                )
            )

            # -------------------------------------------------
            # Determine whether this is the same person.
            #
            # We use:
            #
            # 1. Strong embedding match
            # OR
            # 2. Reasonable embedding + nearby position
            # -------------------------------------------------

            embedding_match = (
                embedding_similarity
                >= EMBEDDING_TRACK_THRESHOLD
            )

            position_match = (
                position_distance is not None
                and position_distance
                <= POSITION_TRACK_THRESHOLD
            )

            if not (
                embedding_match
                or (
                    embedding_similarity >= 0.50
                    and position_match
                )
            ):
                logger.debug(
                    "Track %d rejected: "
                    "embedding=%.4f position=%s",
                    track["track_id"],
                    embedding_similarity,
                    position_distance
                )

                continue

            # -------------------------------------------------
            # Combined tracking score
            # -------------------------------------------------

            if position_distance is None:
                position_score = 0.0
            else:
                position_score = max(
                    0.0,
                    1.0
                    - (
                        position_distance
                        / POSITION_TRACK_THRESHOLD
                    )
                )

            combined_score = (
                0.75 * embedding_similarity
                + 0.25 * position_score
            )

            logger.info(
                "Frame %d face %s -> "
                "track %d | "
                "embedding=%.4f | "
                "position=%s | "
                "combined=%.4f",
                frame_index,
                face.get("face_index"),
                track["track_id"],
                embedding_similarity,
                (
                    f"{position_distance:.4f}"
                    if position_distance is not None
                    else "N/A"
                ),
                combined_score
            )

            if combined_score > best_score:

                best_score = combined_score
                best_track = track

        # -----------------------------------------------------
        # Return best match
        # -----------------------------------------------------

        if best_track is not None:

            logger.info(
                "MATCHED frame %d face %s "
                "to track %d "
                "score=%.4f",
                frame_index,
                face.get("face_index"),
                best_track["track_id"],
                best_score
            )

            return (
                best_track,
                best_score
            )

        logger.info(
            "NEW TRACK for frame %d face %s",
            frame_index,
            face.get("face_index")
        )

        return (
            None,
            None
        )

    # =========================================================
    # PROCESS FIVE FRAMES
    # =========================================================

    def process_frames(
        self,
        image_frames: list[bytes]
    ):

        if len(image_frames) != REQUIRED_FRAMES:

            return {
                "success": False,
                "message": (
                    f"Exactly {REQUIRED_FRAMES} "
                    "frames are required"
                ),
                "total_tracks": 0,
                "eligible_employees": 0,
                "tracks": [],
                "employees": [],
            }

        tracks: list[
            dict[str, Any]
        ] = []

        # =====================================================
        # PROCESS EACH FRAME
        # =====================================================

        for frame_index, image_bytes in enumerate(
            image_frames
        ):

            logger.info(
                "========================================"
            )

            logger.info(
                "Processing frame %d/%d",
                frame_index + 1,
                REQUIRED_FRAMES
            )

            result = self.engine.analyze(
                image_bytes
            )

            if not result.get(
                "success",
                False
            ):

                logger.warning(
                    "Frame %d failed: %s",
                    frame_index + 1,
                    result.get(
                        "message"
                    )
                )

                continue

            faces = result.get(
                "faces",
                []
            )

            logger.info(
                "Frame %d detected %d face(s)",
                frame_index + 1,
                len(faces)
            )

            # =================================================
            # PROCESS EVERY FACE
            # =================================================

            for face in faces:

                embedding = face.get(
                    "_embedding"
                )

                if embedding is None:

                    logger.warning(
                        "Face %s has no embedding",
                        face.get(
                            "face_index"
                        )
                    )

                    continue

                # -------------------------------------------------
                # Find existing track
                # -------------------------------------------------

                matched_track, match_score = (
                    self._find_matching_track(
                        face,
                        tracks,
                        frame_index
                    )
                )

                # -------------------------------------------------
                # Create new track
                # -------------------------------------------------

                if matched_track is None:

                    matched_track = {
                        "track_id": len(
                            tracks
                        ),
                        "observations": [],
                    }

                    tracks.append(
                        matched_track
                    )

                    logger.info(
                        "Created track %d "
                        "for frame %d face %s",
                        matched_track["track_id"],
                        frame_index,
                        face.get("face_index")
                    )

                # -------------------------------------------------
                # Store observation
                # -------------------------------------------------

                observation = {
                    "frame_index": frame_index,

                    "face_index": face.get(
                        "face_index"
                    ),

                    "is_live": bool(
                        face.get(
                            "is_live",
                            False
                        )
                    ),

                    "live_score": float(
                        face.get(
                            "live_score",
                            0.0
                        )
                    ),

                    "recognized": bool(
                        face.get(
                            "recognized",
                            False
                        )
                    ),

                    "employee_id": face.get(
                        "employee_id"
                    ),

                    "confidence": float(
                        face.get(
                            "confidence",
                            0.0
                        )
                    ),

                    "bbox": face.get(
                        "bbox"
                    ),

                    "_embedding": embedding,
                }

                matched_track[
                    "observations"
                ].append(
                    observation
                )

        # =====================================================
        # AGGREGATE TRACKS
        # =====================================================

        final_tracks = []

        for track in tracks:

            observations = track[
                "observations"
            ]

            if not observations:
                continue

            # -------------------------------------------------
            # LIVENESS
            # -------------------------------------------------

            live_observations = [
                obs
                for obs in observations
                if obs["is_live"]
            ]

            live_frames = len(
                live_observations
            )

            passed_liveness = (
                live_frames
                >= MIN_LIVE_FRAMES
            )

            # -------------------------------------------------
            # RECOGNITION VOTING
            # -------------------------------------------------

            recognized_observations = [
                obs
                for obs in observations
                if (
                    obs["recognized"]
                    and obs["employee_id"]
                    is not None
                )
            ]

            employee_votes = {}

            for obs in recognized_observations:

                employee_id = obs[
                    "employee_id"
                ]

                employee_votes[
                    employee_id
                ] = employee_votes.get(
                    employee_id,
                    0
                ) + 1

            employee_id = None

            if employee_votes:

                employee_id = max(
                    employee_votes,
                    key=employee_votes.get
                )

            # -------------------------------------------------
            # Recognition confidence
            # -------------------------------------------------

            confidence_values = [
                obs["confidence"]
                for obs in recognized_observations
                if obs["confidence"] > 0
            ]

            recognition_confidence = (
                float(
                    np.mean(
                        confidence_values
                    )
                )
                if confidence_values
                else 0.0
            )

            # -------------------------------------------------
            # Liveness average
            # -------------------------------------------------

            liveness_values = [
                obs["live_score"]
                for obs in observations
                if obs["live_score"] > 0
            ]

            liveness_average = (
                float(
                    np.mean(
                        liveness_values
                    )
                )
                if liveness_values
                else 0.0
            )

            # -------------------------------------------------
            # Final state
            # -------------------------------------------------

            recognized = (
                employee_id is not None
            )

            eligible_for_attendance = (
                passed_liveness
                and recognized
            )

            final_tracks.append(
                {
                    "track_id": track[
                        "track_id"
                    ],

                    "employee_id": employee_id,

                    "total_observations": len(
                        observations
                    ),

                    "live_frames": live_frames,

                    "total_frames": (
                        REQUIRED_FRAMES
                    ),

                    "passed_liveness": (
                        passed_liveness
                    ),

                    "recognized": recognized,

                    "recognition_confidence": round(
                        recognition_confidence,
                        4
                    ),

                    "liveness_average": round(
                        liveness_average,
                        4
                    ),

                    "eligible_for_attendance": (
                        eligible_for_attendance
                    ),

                    "employee_votes": (
                        employee_votes
                    ),

                    "observations": observations,
                }
            )

        # =====================================================
        # DEDUPLICATE EMPLOYEES
        # =====================================================

        unique_employees = {}

        for track in final_tracks:

            employee_id = track.get(
                "employee_id"
            )

            if employee_id is None:
                continue

            if not track[
                "eligible_for_attendance"
            ]:
                continue

            existing = unique_employees.get(
                employee_id
            )

            if existing is None:

                unique_employees[
                    employee_id
                ] = track

                continue

            if (
                track[
                    "recognition_confidence"
                ]
                >
                existing[
                    "recognition_confidence"
                ]
            ):

                unique_employees[
                    employee_id
                ] = track

        employees = list(
            unique_employees.values()
        )

        # =====================================================
        # REMOVE INTERNAL EMBEDDINGS
        # =====================================================

        for track in final_tracks:

            for observation in track[
                "observations"
            ]:

                observation.pop(
                    "_embedding",
                    None
                )

        # =====================================================
        # RESULT
        # =====================================================

        logger.info(
            "========================================"
        )

        logger.info(
            "Multi-face tracking completed: "
            "tracks=%d eligible=%d",
            len(final_tracks),
            len(employees)
        )

        return {
            "success": True,

            "message": (
                f"Processed {REQUIRED_FRAMES} frames"
            ),

            "total_tracks": len(
                final_tracks
            ),

            "eligible_employees": len(
                employees
            ),

            "tracks": final_tracks,

            "employees": employees,
        }