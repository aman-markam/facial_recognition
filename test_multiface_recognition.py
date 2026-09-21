from pathlib import Path

from app.services.multiface_engine import MultiFaceEngine


IMAGE_PATH = Path("test_images/1655538997918.jpg")


def main():

    print()
    print("======================================")
    print(" MULTI-FACE RECOGNITION TEST")
    print("======================================")

    if not IMAGE_PATH.exists():

        print(
            f"Image not found: {IMAGE_PATH}"
        )

        print()
        print(
            "Put a test image here:"
        )

        print(
            "test_images/multiface.jpg"
        )

        return

    engine = MultiFaceEngine()

    image_bytes = IMAGE_PATH.read_bytes()

    result = engine.analyze(
        image_bytes
    )

    print()
    print("========== RESULT ==========")

    print(
        f"Success          : "
        f"{result['success']}"
    )

    print(
        f"Message          : "
        f"{result['message']}"
    )

    print(
        f"Total Faces      : "
        f"{result.get('total_faces', 0)}"
    )

    print(
        f"Live Faces       : "
        f"{result.get('live_faces', 0)}"
    )

    print(
        f"Recognized Faces  : "
        f"{result.get('recognized_faces', 0)}"
    )

    print()

    print("Face Details:")

    for face in result.get(
        "faces",
        []
    ):

        print(
            f"  Face {face['face_index'] + 1}"
            f" | Live={face['is_live']}"
            f" | Live Score={face['live_score']}"
            f" | Recognized={face['recognized']}"
            f" | Employee={face['employee_id']}"
            f" | Confidence={face['confidence']}"
        )

    print(
        "============================"
    )


if __name__ == "__main__":
    main()