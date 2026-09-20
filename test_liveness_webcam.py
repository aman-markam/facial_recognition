import cv2

OUTPUT_PATH = "test_images/spoof_test.jpg"

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit(1)

print("================================")
print("LIVENESS SPOOF TEST")
print("================================")
print("1. Display test_face.jpg on your phone.")
print("2. Hold the phone in front of the camera.")
print("3. Make sure the phone screen is visible.")
print("4. Press SPACE to capture.")
print("5. Press ESC to cancel.")
print("================================")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    cv2.imshow("Spoof Test - Phone Photo", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE
        cv2.imwrite(OUTPUT_PATH, frame)
        print(f"\nCaptured: {OUTPUT_PATH}")
        break

    if key == 27:  # ESC
        print("\nTest cancelled.")
        break

camera.release()
cv2.destroyAllWindows()