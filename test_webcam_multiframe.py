import time
from pathlib import Path

import cv2

from app.services.multiframe_liveness_service import (
    multiframe_liveness_service,
)


CAMERA_INDEX = 0
NUMBER_OF_FRAMES = 5

OUTPUT_DIR = Path("test_images/webcam_frames")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)


# Try to use a reasonable webcam resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print()
print("================================")
print("WEBCAM MULTI-FRAME LIVENESS TEST")
print("================================")
print()
print("Instructions:")
print("1. Look directly at the camera.")
print("2. Make sure your face is clearly visible.")
print("3. Use good lighting.")
print("4. Keep only one person in the frame.")
print("5. Press SPACE when ready.")
print("6. ESC exits.")
print()


captured_frames = []


try:

    # ==========================================
    # Live preview
    # ==========================================

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam.")
            break

        cv2.putText(
            frame,
            "Look at camera - Press SPACE to capture",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "Webcam Liveness Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            break

        if key == 27:  # ESC
            print("Test cancelled.")
            raise SystemExit(0)


    # ==========================================
    # Capture 5 frames
    # ==========================================

    print()
    print("Capturing frames...")
    print()

    for frame_number in range(
        1,
        NUMBER_OF_FRAMES + 1,
    ):

        success, frame = camera.read()

        if not success:

            print(
                f"ERROR: Could not capture "
                f"frame {frame_number}"
            )

            continue


        # Save frame
        frame_path = (
            OUTPUT_DIR
            / f"frame_{frame_number}.jpg"
        )

        cv2.imwrite(
            str(frame_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )


        # Convert to JPEG bytes
        success_encode, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success_encode:

            print(
                f"ERROR: Could not encode "
                f"frame {frame_number}"
            )

            continue


        image_bytes = buffer.tobytes()

        captured_frames.append(
            image_bytes
        )

        print(
            f"Captured frame "
            f"{frame_number}/{NUMBER_OF_FRAMES}"
        )

        # Small delay between frames
        time.sleep(0.3)


finally:

    camera.release()
    cv2.destroyAllWindows()


# ==========================================
# Validate frames
# ==========================================

print()

print(
    f"Captured {len(captured_frames)} "
    f"out of {NUMBER_OF_FRAMES} frames."
)


if len(captured_frames) != NUMBER_OF_FRAMES:

    print(
        "ERROR: Failed to capture all frames."
    )

    raise SystemExit(1)


# ==========================================
# Run liveness
# ==========================================

print()
print("Running multi-frame liveness...")
print()


result = (
    multiframe_liveness_service.check_frames(
        captured_frames
    )
)


# ==========================================
# Result
# ==========================================

print()
print("==============================")
print("WEBCAM LIVENESS RESULT")
print("==============================")

print(
    f"Success       : {result['success']}"
)

print(
    f"Is Live       : {result['is_live']}"
)

print(
    f"Total Frames  : {result['total_frames']}"
)

print(
    f"Live Frames   : {result['live_frames']}"
)

print(
    f"Spoof Frames  : {result['spoof_frames']}"
)

print(
    f"Scores        : {result['scores']}"
)

print(
    f"Average Score : {result['average_score']}"
)

print(
    f"Message       : {result['message']}"
)

print("==============================")