import sys
from datetime import date, time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models.attendance import Attendance
from app.schemas.attendance import AttendanceResponse, AttendanceEmployeeResponse


def test_attendance_model_geolocation():
    print("\n--- Test Attendance Model Geolocation Fields ---")
    att = Attendance(
        employee_id=1,
        attendance_date=date.today(),
        check_in=time(9, 15),
        status="PRESENT",
        confidence=0.95,
        latitude=12.9716,
        longitude=77.5946
    )
    print(f"Latitude: {att.latitude}, Longitude: {att.longitude}")
    assert att.latitude == 12.9716, "Latitude mismatch"
    assert att.longitude == 77.5946, "Longitude mismatch"
    print("✅ Attendance model geolocation fields verified.")


def test_attendance_schemas_geolocation():
    print("\n--- Test Attendance Schemas Geolocation Fields ---")
    resp_data = {
        "id": 10,
        "employee_id": 1,
        "employee_code": "EMP001",
        "employee_name": "Aman",
        "attendance_date": date.today(),
        "check_in": time(9, 15),
        "check_out": None,
        "working_minutes": 0,
        "status": "PRESENT",
        "confidence": Decimal("0.9500"),
        "latitude": 12.9716,
        "longitude": 77.5946
    }

    schema_resp = AttendanceEmployeeResponse(**resp_data)
    print(f"Schema output latitude: {schema_resp.latitude}, longitude: {schema_resp.longitude}")
    assert schema_resp.latitude == 12.9716, "Schema latitude mismatch"
    assert schema_resp.longitude == 77.5946, "Schema longitude mismatch"
    print("✅ Attendance Pydantic response schemas verified.")


if __name__ == "__main__":
    test_attendance_model_geolocation()
    test_attendance_schemas_geolocation()
    print("\n🎉 ALL ATTENDANCE GEOLOCATION TESTS PASSED SUCCESSFULLY!")
