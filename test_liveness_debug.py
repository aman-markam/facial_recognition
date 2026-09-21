import cv2
from insightface.app import FaceAnalysis


IMAGE_PATH = "test_images/IMG20210109210741.jpg"


def main():

    print("Loading InsightFace...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        addons=["liveness"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("ERROR: Could not load image:")
        print(IMAGE_PATH)
        return

    faces = app.get(image)

    print()
    print("=" * 60)
    print("TOTAL FACES:", len(faces))
    print("=" * 60)

    for index, face in enumerate(faces):

        print()
        print(f"FACE {index + 1}")
        print("-" * 60)

        print("Object type:")
        print(type(face))

        print()
        print("Face object:")
        print(face)

        print()
        print("Face __dict__:")

        try:
            print(face.__dict__)
        except Exception as e:
            print("ERROR:", e)

        print()
        print("face.liveness:")

        try:
            value = face.liveness
            print("VALUE:", value)
            print("TYPE :", type(value))
        except Exception as e:
            print("ERROR:", e)

        print()
        print("Available attributes:")

        try:
            print(dir(face))
        except Exception as e:
            print("ERROR:", e)

        print()
        print("=" * 60)

    print()
    print("DEBUG COMPLETE")


if __name__ == "__main__":
    main()