import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

from pathlib import Path

from app.services.multiface_tracker import (
    MultiFaceTracker
)


TEST_DIR = Path("test_images")

FRAME_FILES = [
    TEST_DIR / "frame1.jpg",
    TEST_DIR / "frame2.jpg",
    TEST_DIR / "frame3.jpg",
    TEST_DIR / "frame4.jpg",
    TEST_DIR / "frame5.jpg",
]


def main():

    print()
    print("==========================================")
    print(" MULTI-FACE 5-FRAME TRACKING TEST")
    print("==========================================")

    missing = [
        path
        for path in FRAME_FILES
        if not path.exists()
    ]

    if missing:

        print()
        print("Missing frames:")

        for path in missing:
            print(f"  {path}")

        print()
        print(
            "Put exactly 5 frames inside:"
        )

        print(
            "test_images/"
        )

        return

    image_frames = [
        path.read_bytes()
        for path in FRAME_FILES
    ]

    tracker = MultiFaceTracker()

    result = tracker.process_frames(
        image_frames
    )

    print()
    print("============== RESULT ==============")

    print(
        f"Success             : "
        f"{result['success']}"
    )

    print(
        f"Message             : "
        f"{result['message']}"
    )

    print(
        f"Total Tracks        : "
        f"{result.get('total_tracks', 0)}"
    )

    print(
        f"Eligible Employees  : "
        f"{result.get('eligible_employees', 0)}"
    )

    print()

    print("TRACK DETAILS")
    print("--------------------------------------")

    for track in result.get(
        "tracks",
        []
    ):

        print(
            f"Track {track['track_id']}"
            f" | Employee={track['employee_id']}"
            f" | Observations={track['total_observations']}"
            f" | Live={track['live_frames']}/5"
            f" | Liveness={track['liveness_average']}"
            f" | Recognition={track['recognition_confidence']}"
            f" | Eligible={track['eligible_for_attendance']}"
        )

    print()
    print("EMPLOYEES TO ATTENDANCE")

    for employee in result.get(
        "employees",
        []
    ):

        print(
            f"  Employee {employee['employee_id']}"
            f" | Confidence={employee['recognition_confidence']}"
            f" | Live={employee['live_frames']}/5"
        )

    print()
    print("====================================")


if __name__ == "__main__":
    main()