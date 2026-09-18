from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.employee import Employee
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeResponse,
    EmployeeUpdate,
)

from app.dependencies.auth import get_current_admin
from app.models.admin import Admin
# from app.core.dependencies import get_current_admin
# from app.models.admin import Admin

router = APIRouter(
    prefix="/employees",
    tags=["Dashboard - Employees"],
    dependencies=[Depends(get_current_admin)]
)


@router.post(
    "",
    response_model=EmployeeResponse,
    status_code=status.HTTP_201_CREATED
)
def create_employee(
    employee_data: EmployeeCreate,
    db: Session = Depends(get_db)
):

    existing_employee = (
        db.query(Employee)
        .filter(
            Employee.employee_code
            == employee_data.employee_code
        )
        .first()
    )

    if existing_employee:
        raise HTTPException(
            status_code=409,
            detail="Employee code already exists"
        )

    if employee_data.email:

        existing_email = (
            db.query(Employee)
            .filter(
                Employee.email
                == employee_data.email
            )
            .first()
        )

        if existing_email:
            raise HTTPException(
                status_code=409,
                detail="Email already exists"
            )

    employee = Employee(
        employee_code=employee_data.employee_code,
        name=employee_data.name,
        email=employee_data.email,
        department=employee_data.department,
    )

    db.add(employee)
    db.commit()
    db.refresh(employee)

    return employee


@router.get(
    "",
    response_model=list[EmployeeResponse]
)
def get_employees(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(
        get_current_admin
    )
):

    employees = (
        db.query(Employee)
        .order_by(Employee.id)
        .all()
    )

    return employees


@router.get(
    "/{employee_id}",
    response_model=EmployeeResponse
)
def get_employee(
    employee_id: int,
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

    return employee


@router.put(
    "/{employee_id}",
    response_model=EmployeeResponse
)
def update_employee(
    employee_id: int,
    employee_data: EmployeeUpdate,
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

    if employee_data.name is not None:
        employee.name = employee_data.name

    if employee_data.email is not None:
        employee.email = employee_data.email

    if employee_data.department is not None:
        employee.department = employee_data.department

    if employee_data.is_active is not None:
        employee.is_active = employee_data.is_active

    db.commit()
    db.refresh(employee)

    return employee


@router.delete(
    "/{employee_id}"
)
def delete_employee(
    employee_id: int,
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

    db.delete(employee)
    db.commit()

    return {
        "success": True,
        "message": "Employee deleted successfully"
    }