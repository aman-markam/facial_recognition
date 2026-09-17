from app.core.security import hash_password
from app.database.database import SessionLocal
from app.models.admin import Admin


db = SessionLocal()

try:

    admin = Admin(
        username="admin",
        email="admin@example.com",
        password_hash=hash_password("admin123"),
        full_name="System Administrator"
    )

    db.add(admin)
    db.commit()

    print("Admin created successfully")

except Exception as e:

    db.rollback()
    print("Error:", e)

finally:

    db.close()