from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.dependencies import get_current_admin

from app.database.database import get_db
from app.models.attendance import Attendance
from app.models.employee import Employee
from app.schemas.report import (
    DailyReportItem,
    EmployeeReportResponse,
    OverviewResponse,
)


router = APIRouter(
    prefix="/reports",
    tags=["Dashboard - Reports"],
    dependencies=[Depends(get_current_admin)]
)


# ==================================================
# OVERVIEW
# ==================================================

@router.get(
    "/overview",
    response_model=OverviewResponse
)
def get_overview(
    db: Session = Depends(get_db)
):

    today = date.today()

    total_employees = (
        db.query(Employee)
        .filter(
            Employee.is_active == True
        )
        .count()
    )

    present = (
        db.query(Attendance)
        .filter(
            Attendance.attendance_date == today,
            Attendance.status == "PRESENT"
        )
        .count()
    )

    absent = max(
        total_employees - present,
        0
    )

    percentage = (
        (present / total_employees) * 100
        if total_employees > 0
        else 0
    )

    return {
        "date": today,
        "total_employees": total_employees,
        "present": present,
        "absent": absent,
        "attendance_percentage": round(
            percentage,
            2
        )
    }


# ==================================================
# DAILY REPORT
# ==================================================

@router.get(
    "/daily",
    response_model=list[DailyReportItem]
)
def get_daily_report(
    start_date: date = Query(...),
    end_date: date = Query(...),
    db: Session = Depends(get_db)
):

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date cannot be after end_date"
        )

    total_employees = (
        db.query(Employee)
        .filter(
            Employee.is_active == True
        )
        .count()
    )

    results = []

    current_date = start_date

    while current_date <= end_date:

        present = (
            db.query(Attendance)
            .filter(
                Attendance.attendance_date
                == current_date,
                Attendance.status
                == "PRESENT"
            )
            .count()
        )

        absent = max(
            total_employees - present,
            0
        )

        percentage = (
            (present / total_employees) * 100
            if total_employees > 0
            else 0
        )

        results.append({
            "date": current_date,
            "total_employees": total_employees,
            "present": present,
            "absent": absent,
            "attendance_percentage": round(
                percentage,
                2
            )
        })

        current_date += timedelta(days=1)

    return results


# ==================================================
# EMPLOYEE REPORT
# ==================================================

@router.get(
    "/employee/{employee_id}",
    response_model=EmployeeReportResponse
)
def get_employee_report(
    employee_id: int,
    start_date: date = Query(...),
    end_date: date = Query(...),
    db: Session = Depends(get_db)
):

    employee = (
        db.query(Employee)
        .filter(
            Employee.id == employee_id
        )
        .first()
    )

    if not employee:
        raise HTTPException(
            status_code=404,
            detail="Employee not found"
        )

    total_days = (
        end_date - start_date
    ).days + 1

    present_days = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id == employee_id,
            Attendance.attendance_date >= start_date,
            Attendance.attendance_date <= end_date,
            Attendance.status == "PRESENT"
        )
        .count()
    )

    absent_days = max(
        total_days - present_days,
        0
    )

    percentage = (
        (present_days / total_days) * 100
        if total_days > 0
        else 0
    )

    return {
        "employee_id": employee.id,
        "employee_code": employee.employee_code,
        "employee_name": employee.name,
        "total_days": total_days,
        "present_days": present_days,
        "absent_days": absent_days,
        "attendance_percentage": round(
            percentage,
            2
        )
    }