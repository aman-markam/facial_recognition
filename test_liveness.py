from pathlib import Path

from app.services.liveness_service import (
    liveness_service,
)


IMAGE_PATH = Path("test_images/spoof_test.jpg")

if not IMAGE_PATH.exists():
    fallback_jpg = Path("test_images/test_face.jpg")
    if fallback_jpg.exists():
        IMAGE_PATH = fallback_jpg
    else:
        print(
            "Test image not found (looked for test_images/test_face.png and test_images/test_face.jpg)"
        )
        raise SystemExit(1)


with open(
    IMAGE_PATH,
    "rb"
) as file:

    image_bytes = file.read()


result = (
    liveness_service.check_liveness(
        image_bytes
    )
)


print()
print(
    "=============================="
)
print(
    "LIVENESS TEST RESULT"
)
print(
    "=============================="
)
print(
    f"Success    : {result['success']}"
)
print(
    f"Is Live    : {result['is_live']}"
)
print(
    f"Live Score : {result['live_score']}"
)
print(
    f"Message    : {result['message']}"
)
print(
    "=============================="
)
