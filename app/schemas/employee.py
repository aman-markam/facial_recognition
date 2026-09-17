from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class EmployeeCreate(BaseModel):
    employee_code: str
    name: str
    email: EmailStr | None = None
    department: str | None = None


class EmployeeUpdate(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
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