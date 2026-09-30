import json
import logging
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.database.database import SessionLocal
from app.models.employee import Employee
from app.models.face_embedding import FaceEmbedding


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_JSON_PATH = ROOT / "face_data" / "embeddings.json"

DUPLICATE_FACE_THRESHOLD = 0.60


def _session(db: Session | None) -> tuple[Session, bool]:
    if db is not None:
        return db, False
    return SessionLocal(), True


def _to_float_list(embedding) -> list[float]:
    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32).tolist()
    return [float(value) for value in embedding]


def _import_json_if_needed(session: Session) -> int:
    if not EMBEDDINGS_JSON_PATH.exists():
        return 0

    with open(EMBEDDINGS_JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        logger.warning("embeddings.json is not a JSON object; skipping import")
        return 0

    existing_codes = {
        row.employee_code
        for row in session.query(FaceEmbedding.employee_code).all()
    }

    imported = 0
    for employee_code, embedding in data.items():
        code = str(employee_code)
        if code in existing_codes:
            continue

        employee = (
            session.query(Employee)
            .filter(Employee.employee_code == code)
            .first()
        )
        if employee is None:
            logger.warning(
                "Skipping JSON embedding for unknown employee_code=%s",
                code,
            )
            continue

        try:
            vector = _to_float_list(embedding)
        except (TypeError, ValueError):
            logger.warning(
                "Skipping invalid embedding for employee_code=%s",
                code,
            )
            continue

        session.add(
            FaceEmbedding(
                employee_id=employee.id,
                employee_code=employee.employee_code,
                embedding=vector,
            )
        )
        existing_codes.add(employee.employee_code)
        imported += 1

    if imported:
        session.commit()
        logger.info(
            "Imported %d face embeddings from %s into PostgreSQL",
            imported,
            EMBEDDINGS_JSON_PATH,
        )
    return imported


def load_all_embeddings(db: Session | None = None) -> dict[str, np.ndarray]:
    session, owns_session = _session(db)
    try:
        try:
            _import_json_if_needed(session)
        except Exception:
            logger.exception("Unable to import embeddings.json into PostgreSQL")
            session.rollback()

        rows = session.query(FaceEmbedding).all()
        embeddings = {}
        for row in rows:
            if not row.embedding:
                continue
            embeddings[row.employee_code] = np.array(
                row.embedding,
                dtype=np.float32,
            )
        logger.info("Loaded %d employee embeddings from PostgreSQL", len(embeddings))
        return embeddings
    except Exception:
        logger.exception("Unable to load face embeddings from PostgreSQL")
        return {}
    finally:
        if owns_session:
            session.close()


def is_enrolled(employee_code: str, db: Session | None = None) -> bool:
    session, owns_session = _session(db)
    try:
        row = (
            session.query(FaceEmbedding.id)
            .filter(FaceEmbedding.employee_code == str(employee_code))
            .first()
        )
        return row is not None
    finally:
        if owns_session:
            session.close()


def save_embedding(
    employee_code: str,
    embedding: list[float],
    db: Session | None = None,
) -> bool:
    employee_code = str(employee_code)
    vector = _to_float_list(embedding)
    candidate = np.array(vector, dtype=np.float32)
    candidate = candidate / (np.linalg.norm(candidate) + 1e-10)

    session, owns_session = _session(db)
    try:
        employee = (
            session.query(Employee)
            .filter(Employee.employee_code == employee_code)
            .first()
        )
        if employee is None:
            raise ValueError(
                f"Employee '{employee_code}' was not found. "
                "Enroll a registered employee before saving a face embedding."
            )

        existing_rows = session.query(FaceEmbedding).all()
        for row in existing_rows:
            if row.employee_code == employee_code:
                continue
            existing = np.array(row.embedding, dtype=np.float32)
            existing = existing / (np.linalg.norm(existing) + 1e-10)
            similarity = float(np.dot(candidate, existing))
            if similarity >= DUPLICATE_FACE_THRESHOLD:
                logger.warning(
                    "Duplicate face enrollment blocked: employee=%s matches existing employee=%s (similarity=%.4f)",
                    employee_code,
                    row.employee_code,
                    similarity,
                )
                raise ValueError(
                    f"This face is already enrolled in the system for employee "
                    f"'{row.employee_code}' (similarity: {similarity:.2f}). "
                    "Re-enrolling an existing face for a new employee is not allowed."
                )

        record = (
            session.query(FaceEmbedding)
            .filter(FaceEmbedding.employee_id == employee.id)
            .first()
        )
        if record is None:
            record = FaceEmbedding(
                employee_id=employee.id,
                employee_code=employee.employee_code,
                embedding=vector,
            )
            session.add(record)
        else:
            record.employee_code = employee.employee_code
            record.embedding = vector

        session.commit()
        logger.info("Face embedding saved in PostgreSQL: employee=%s", employee_code)
        return True
    except ValueError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        logger.exception("Failed to save face embedding for employee=%s", employee_code)
        raise RuntimeError("Failed to save face embedding.")
    finally:
        if owns_session:
            session.close()


def delete_embedding(employee_code: str, db: Session | None = None) -> bool:
    employee_code = str(employee_code)
    session, owns_session = _session(db)
    try:
        deleted = (
            session.query(FaceEmbedding)
            .filter(FaceEmbedding.employee_code == employee_code)
            .delete(synchronize_session=False)
        )
        session.commit()
        if deleted:
            logger.info(
                "Face embedding deleted from PostgreSQL: employee=%s",
                employee_code,
            )
            return True
        return False
    except Exception:
        session.rollback()
        logger.exception(
            "Failed to delete face embedding for employee=%s",
            employee_code,
        )
        raise RuntimeError("Failed to delete face embedding.")
    finally:
        if owns_session:
            session.close()


def import_json_embeddings_if_needed(db: Session | None = None) -> int:
    """
    One-time import of face_data/embeddings.json into PostgreSQL
    when the face_embeddings table is empty.
    """
    session, owns_session = _session(db)
    try:
        return _import_json_if_needed(session)
    except Exception:
        session.rollback()
        logger.exception("Failed to import embeddings.json into PostgreSQL")
        return 0
    finally:
        if owns_session:
            session.close()
