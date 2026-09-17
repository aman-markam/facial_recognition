from pydantic import BaseModel


class FaceEnrollmentResponse(BaseModel):
    success: bool
    employee_id: int
    employee_code: str
    employee_name: str
    message: str