from pathlib import Path

from app.services.liveness_service import (
    liveness_service
)


IMAGE_PATH = Path(
    "test_images/IMG_20240713_103509.jpg"
)


def main():

    if not IMAGE_PATH.exists():

        print(
            f"Image not found: {IMAGE_PATH}"
        )

        print(
            "\nCreate this test image first:"
        )

        print(
            "test_images/multiple_faces.jpg"
        )

        return

    with open(
        IMAGE_PATH,
        "rb"
    ) as file:

        image_bytes = file.read()

    print(
        "\nRunning multi-face liveness..."
    )

    result = (
        liveness_service
        .check_multiple_faces(
            image_bytes
        )
    )

    print(
        "\n========== RESULT =========="
    )

    print(
        f"Success      : "
        f"{result.get('success')}"
    )

    print(
        f"Total Faces  : "
        f"{result.get('total_faces')}"
    )

    print(
        f"Live Faces   : "
        f"{result.get('live_faces')}"
    )

    print(
        f"Spoof Faces  : "
        f"{result.get('spoof_faces')}"
    )

    print(
        f"Message      : "
        f"{result.get('message')}"
    )

    print(
        "\nFace Details:"
    )

    for face in result.get(
        "faces",
        []
    ):

        print(
            f"  Face "
            f"{face['face_index']} | "
            f"Live={face['is_live']} | "
            f"Score={face['live_score']}"
        )

    print(
        "============================\n"
    )


if __name__ == "__main__":
    main()