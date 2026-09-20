from pathlib import Path

from app.services.multiframe_liveness_service import (
    multiframe_liveness_service,
)


# ==========================================
# Test images
# ==========================================

IMAGE_DIR = Path("test_images")

IMAGE_PATHS = [
    IMAGE_DIR / "live6.jpg",
    IMAGE_DIR / "live7.jpg",
    IMAGE_DIR / "live8.jpg",
    IMAGE_DIR / "live9.jpg",
    IMAGE_DIR / "live0.jpg",
]


# ==========================================
# Load images
# ==========================================

image_frames = []

for image_path in IMAGE_PATHS:

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        raise SystemExit(1)

    with open(image_path, "rb") as file:
        image_frames.append(file.read())


# ==========================================
# Run multi-frame liveness
# ==========================================

result = multiframe_liveness_service.check_frames(
    image_frames
)


# ==========================================
# Display result
# ==========================================

print()
print("==============================")
print("MULTI-FRAME LIVENESS TEST")
print("==============================")

print(f"Success       : {result['success']}")
print(f"Is Live       : {result['is_live']}")
print(f"Total Frames  : {result['total_frames']}")
print(f"Live Frames   : {result['live_frames']}")
print(f"Spoof Frames  : {result['spoof_frames']}")
print(f"Scores        : {result['scores']}")
print(f"Average Score : {result['average_score']}")
print(f"Message       : {result['message']}")

print("==============================")