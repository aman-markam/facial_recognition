from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EmployeeCreate(BaseModel):
    employee_code: str
    name: str
    email: str | None = None
    department: str | None = None


class EmployeeUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
    department: str | None = None
    is_active: bool | None = None


class EmployeeResponse(BaseModel):
    id: int
    employee_code: str
    name: str
    email: str | None
    department: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )