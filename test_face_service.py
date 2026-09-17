from pathlib import Path

from app.services.face_service import face_service


image_path = Path(
    r"D:\project - face_recognition\face_recognition2.0\data\faces\001\001.jpg"
)


with open(image_path, "rb") as file:
    image_bytes = file.read()


try:

    embedding = face_service.get_face_embedding(
        image_bytes
    )

    print("SUCCESS")
    print("Embedding length:", len(embedding))

except Exception as e:

    print("FAILED")
    print("Error:", e)