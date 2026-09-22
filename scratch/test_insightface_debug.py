import sys
import numpy as np
import cv2
from insightface.app import FaceAnalysis

print("Testing InsightFace loading...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(960, 960))

print("Models loaded in app.models:")
for k, v in app.models.items():
    print(f"  - {k}: {v}")

# Create dummy image with a face-like shape
img = np.zeros((480, 640, 3), dtype=np.uint8)
faces = app.get(img)
print(f"Detected faces in blank image: {len(faces)}")
