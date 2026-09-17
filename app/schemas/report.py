from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class OverviewResponse(BaseModel):
    date: date
    total_employees: int
    present: int
    absent: int
    attendance_percentage: float


class DailyReportItem(BaseModel):
    date: date
    total_employees: int
    present: int
    absent: int
    attendance_percentage: float


class EmployeeReportResponse(BaseModel):
    employee_id: int
    employee_code: str
    employee_name: str
    total_days: int
    present_days: int
    absent_days: int
    attendance_percentage: float