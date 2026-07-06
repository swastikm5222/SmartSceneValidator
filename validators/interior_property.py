import torch
import torch.nn.functional as F

from torchvision import transforms
from models.model_manager import MODELS, device

# YOLO is injected at FastAPI startup into module-level `yolo_model`.
# Keeping YOLO import out prevents import-time weight loading.

from validators.image_quality import (
    validate_image_quality
)

# =====================================================
# YOLO MODEL
# =====================================================

# Injected once by FastAPI startup in api.py.
# If startup didn’t run, detection will fail fast.

yolo_model = None


# =====================================================
# INTERIOR OBJECTS
# =====================================================

STRONG_INTERIOR_OBJECTS = {
    "bed",
    "couch",
    "chair",
    "dining table",
    "tv",
    "refrigerator",
    "microwave",
    "oven",
    "sink",
    "toilet",
    "potted plant"
}

WEAK_INTERIOR_OBJECTS = {
    "cup",
    "bottle",
    "book",
    "laptop",
    "keyboard",
    "mouse",
    "remote",
    "clock",
    "cell phone",
    "vase",
    "wine glass",
    "bowl",
    "fork",
    "knife",
    "spoon"
}

INTERIOR_OBJECTS = (
    STRONG_INTERIOR_OBJECTS | WEAK_INTERIOR_OBJECTS
)

# =====================================================
# SWIN V2 TINY MODEL
# =====================================================

model = MODELS["interior_property"]

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.ToPILImage(),
    transforms.Resize(
        (256, 256)
    ),
    transforms.ToTensor()

])

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = [
    "interior_property",
    "not_interior_property"
]

# =====================================================
# THRESHOLDS
# =====================================================

INTERIOR_CONFIDENCE_THRESHOLD = 0.85
NOT_INTERIOR_CONFIDENCE_THRESHOLD = 0.90

YOLO_CONFIDENCE_THRESHOLD = 0.40

MIN_WEAK_OBJECTS_REQUIRED = 2

# =====================================================
# YOLO OBJECT DETECTION
# =====================================================

def detect_interior_objects(image):

    global yolo_model
    if yolo_model is None:
        raise RuntimeError(
            "YOLO model is not loaded. FastAPI startup did not run or YOLO failed to load."
        )

    results = yolo_model(
        image,
        verbose=False
    )


    detected_objects = []

    for result in results:

        names = result.names

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            object_name = names[
                cls_id
            ]

            if confidence < YOLO_CONFIDENCE_THRESHOLD:
                continue

            if object_name in INTERIOR_OBJECTS:

                detected_objects.append(
                    object_name
                )

    return list(
        set(detected_objects)
    )


# =====================================================
# RESPONSE HELPER
# =====================================================

def build_response(
    status,
    reason,
    decision_basis,
    confidence,
    predicted_class,
    detected_objects,
    strong_objects,
    weak_objects
):

    return {

        "status": status,
        "reason": reason,
        "decision_basis": decision_basis,
        "confidence": round(
            confidence,
            4
        ),

       "classifier_prediction": predicted_class,
        "detected_objects": detected_objects,
        "strong_objects_found": strong_objects,
        "weak_objects_found": weak_objects

    }


# =====================================================
# DEBUG LOGGER
# =====================================================

def print_debug_info(
    predicted_class,
    confidence,
    detected_objects,
    strong_objects,
    weak_objects
):

    print("=" * 60)

    print("Interior Property Validator")
    print(
        "Prediction :",
        predicted_class
    )
    print(
        "Confidence :",
        round(confidence, 4)
    )
    print(
        "Detected Objects :",
        detected_objects
    )
    print(
        "Strong Objects :",
        strong_objects
    )
    print(
        "Weak Objects :",
        weak_objects
    )
    print("=" * 60)


# =====================================================
# MAIN VALIDATOR
# =====================================================

def validate_interior_property(image):

    # -------------------------------------------------
    # IMAGE QUALITY CHECK
    # -------------------------------------------------

    quality_result = validate_image_quality(image)

    if not quality_result.is_valid:
        return quality_result.to_dict()

    # -------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    # -------------------------------------------------
    # MODEL INFERENCE
    # -------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    confidence = float(
        confidence.item()
    )

    predicted_class = CLASS_NAMES[
        prediction.item()
    ]

    detected_objects = []
    strong_objects_found = []
    weak_objects_found = []

    if confidence >= NOT_INTERIOR_CONFIDENCE_THRESHOLD:

        print_debug_info(

            predicted_class,

            confidence,

            detected_objects,

            strong_objects_found,

            weak_objects_found

        )

        if predicted_class == "interior_property":

            return build_response(
                "VALID",
                "VALID IMAGE",
                "classifier",
                confidence,
                predicted_class,
                detected_objects,
                strong_objects_found,
                weak_objects_found

            )

        return build_response(

            "INVALID",
            "INVALID IMAGE",
            "classifier",
            confidence,
            predicted_class,
            detected_objects,
            strong_objects_found,
            weak_objects_found
        )

    # -------------------------------------------------
    # YOLO DETECTION
    # -------------------------------------------------

    detected_objects = detect_interior_objects(
        image
    )

    strong_objects_found = [

        obj

        for obj in detected_objects

        if obj in STRONG_INTERIOR_OBJECTS

    ]

    weak_objects_found = [

        obj

        for obj in detected_objects

        if obj in WEAK_INTERIOR_OBJECTS

    ]

    # -------------------------------------------------
    # DEBUG
    # -------------------------------------------------

    print_debug_info(

        predicted_class,

        confidence,

        detected_objects,

        strong_objects_found,

        weak_objects_found

    )

    if detected_objects:

        return build_response(

            "VALID",
            "VALID IMAGE",
            "yolo",
            confidence,
            predicted_class,
            detected_objects,
            strong_objects_found,
            weak_objects_found

        )

    if predicted_class == "interior_property":

        return build_response(
            "VALID",
            "VALID IMAGE",
            "classifier",
            confidence,
            predicted_class,
            detected_objects,
            strong_objects_found,
            weak_objects_found

        )

    return build_response(

        "INVALID",
        "INVALID IMAGE",
        "classifier",
        confidence,
        predicted_class,
        detected_objects,
        strong_objects_found,
        weak_objects_found
    )
