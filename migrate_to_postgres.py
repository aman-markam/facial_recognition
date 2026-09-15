import os
import sqlite3
from pathlib import Path

import psycopg


SQLITE_PATH = Path(__file__).resolve().parent / "data" / "attendance.db"


def postgres_connection() -> psycopg.Connection:
	return psycopg.connect(
		host=os.getenv("FACE_DB_HOST", "localhost"),
		port=os.getenv("FACE_DB_PORT", "5432"),
		dbname=os.getenv("FACE_DB_NAME", "face_recognition"),
		user=os.getenv("FACE_DB_USER", "postgres"),
		password=os.environ["FACE_DB_PASSWORD"],
	)


def migrate() -> None:
	if not SQLITE_PATH.exists():
		raise FileNotFoundError(f"SQLite database not found: {SQLITE_PATH}")

	with sqlite3.connect(SQLITE_PATH) as source, postgres_connection() as target:
		employees = source.execute(
			"SELECT employee_id, name, created_at FROM employees"
		).fetchall()
		attendance = source.execute(
			"SELECT employee_id, name, attended_at, attendance_date FROM attendance"
		).fetchall()
		target.execute(
			"""
			CREATE TABLE IF NOT EXISTS employees (
				employee_id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				created_at TIMESTAMP NOT NULL
			)
			"""
		)
		target.execute(
			"""
			CREATE TABLE IF NOT EXISTS attendance (
				id BIGSERIAL PRIMARY KEY,
				employee_id TEXT NOT NULL,
				name TEXT NOT NULL,
				attended_at TIMESTAMP NOT NULL,
				attendance_date DATE NOT NULL,
				UNIQUE(employee_id, attendance_date)
			)
			"""
		)
		target.executemany(
			"""
			INSERT INTO employees(employee_id, name, created_at)
			VALUES (%s, %s, %s)
			ON CONFLICT(employee_id) DO UPDATE SET name = EXCLUDED.name
			""",
			employees,
		)
		target.executemany(
			"""
			INSERT INTO attendance(employee_id, name, attended_at, attendance_date)
			VALUES (%s, %s, %s, %s)
			ON CONFLICT(employee_id, attendance_date) DO NOTHING
			""",
			attendance,
		)
	print(f"Migrated {len(employees)} employees and {len(attendance)} attendance records.")


if __name__ == "__main__":
	migrate()