import os
from datetime import datetime

import psycopg
from psycopg.rows import dict_row


def connect() -> psycopg.Connection:
	password = os.getenv("FACE_DB_PASSWORD")
	if not password:
		raise RuntimeError(
			"FACE_DB_PASSWORD is not set. Set the PostgreSQL password in PowerShell "
			"before starting the application."
		)
	return psycopg.connect(
		host=os.getenv("FACE_DB_HOST", "localhost"),
		port=os.getenv("FACE_DB_PORT", "5432"),
		dbname=os.getenv("FACE_DB_NAME", "face_recognition"),
		user=os.getenv("FACE_DB_USER", "postgres"),
		password=password,
		row_factory=dict_row,
	)


def initialize() -> None:
	with connect() as connection:
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS employees (
				employee_id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				created_at TIMESTAMP NOT NULL
			)
			"""
		)
		connection.execute(
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


def save_employee(employee_id: str, name: str) -> None:
	with connect() as connection:
		updated = connection.execute(
			"""
			UPDATE employees
			SET name = %s, created_at = %s
			WHERE employee_id = %s
			""",
			(name, datetime.now(), employee_id),
		)
		if updated.rowcount == 0:
			connection.execute(
				"INSERT INTO employees(employee_id, name, created_at) VALUES (%s, %s, %s)",
				(employee_id, name, datetime.now()),
			)


def get_employees() -> dict[str, str]:
	with connect() as connection:
		rows = connection.execute("SELECT employee_id, name FROM employees").fetchall()
	return {str(row["employee_id"]): row["name"] for row in rows}


def record_attendance(employee_id: str, name: str, action: str = "check_in") -> tuple[bool, str]:
	now = datetime.now()
	act = (action or "check_in").lower().strip()
	action_label = "Check-out" if act in ["check_out", "checkout", "out"] else "Check-in"
	with connect() as connection:
		existing = connection.execute(
			"SELECT 1 FROM attendance WHERE employee_id = %s AND attendance_date = %s LIMIT 1",
			(employee_id, now.date()),
		).fetchone()
		if existing and act not in ["check_out", "checkout", "out"]:
			return False, f"Already recorded today: {employee_id} - {name}"
		connection.execute(
			"""
			INSERT INTO attendance(employee_id, name, attended_at, attendance_date)
			VALUES (%s, %s, %s, %s)
			""",
			(employee_id, name, now, now.date()),
		)
	return True, f"{action_label} successful: {employee_id} - {name}"


def attendance_rows() -> list[dict[str, str]]:
	with connect() as connection:
		rows = connection.execute(
			"""
			SELECT employee_id, name, attendance_date::text AS date,
			       TO_CHAR(attended_at, 'HH24:MI:SS') AS time
			FROM attendance
			ORDER BY attended_at DESC
			"""
		).fetchall()
	return rows
