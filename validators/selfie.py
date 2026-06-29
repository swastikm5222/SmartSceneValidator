import cv2

from validators.image_quality import (
    validate_image_quality
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def validate_selfie(image):

    # ----------------------------
    # BLUR CHECK
    # ----------------------------

    quality_result = validate_image_quality(image)

    if not quality_result.is_valid:
        return quality_result.to_dict()

    # ----------------------------
    # FACE DETECTION
    # ----------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(80, 80)
    )

    image_height, image_width = gray.shape

    image_area = (
        image_height *
        image_width
    )

    # ----------------------------
    # LARGE FACE FILTER
    # ----------------------------

    valid_faces = []

    for (x, y, w, h) in faces:

        face_area = w * h

        face_ratio = (
            face_area /
            image_area
        )

        # Face must occupy
        # at least 3% of image

        if face_ratio >= 0.03:

            valid_faces.append(
                (x, y, w, h)
            )

    face_count = len(
        valid_faces
    )

    # ----------------------------
    # FACE COUNT CHECK
    # ----------------------------

    if face_count < 2:

        if face_count == 1:

            return {
                "status": "INVALID",
                "reason": "Only One Valid Face Detected"
            }

        return {
            "status": "INVALID",
            "reason": "No Valid Faces Detected"
        }

    # ----------------------------
    # FACE DISTANCE CHECK
    # ----------------------------

    centers = []

    for (x, y, w, h) in valid_faces:

        centers.append(
            (
                x + w // 2,
                y + h // 2
            )
        )

    x1, y1 = centers[0]
    x2, y2 = centers[1]

    distance = (
        (
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        ) ** 0.5
    )

    # Faces too far apart
    # likely not a selfie

    if distance > image_width * 0.7:

        return {
            "status": "INVALID",
            "reason": "Faces Too Far Apart"
        }

    # ----------------------------
    # VALID
    # ----------------------------

    return {
        "status": "VALID",
        "reason": "VALID IMAGE"
    }

