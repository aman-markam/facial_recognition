from pathlib import Path

from face_engine import FaceEngine


engine = FaceEngine()

image_path = Path(
    r"D:\project - face_recognition\face_recognition2.0\data\faces\001\001.jpg"
)

with open(image_path, "rb") as file:
    image_bytes = file.read()

result = engine.recognize(image_bytes)

print(result)