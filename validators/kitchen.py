import torch
import torch.nn.functional as F
import timm

from torchvision import transforms
from ultralytics import YOLO

from validators.image_quality import (
    validate_image_quality
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Kitchen Validator Device : {device}"
)

# =====================================================
# YOLO MODEL
# =====================================================

yolo_model = YOLO(
    "yolov8n.pt"
)

# =====================================================
# KITCHEN OBJECTS
# =====================================================


STRONG_KITCHEN_OBJECTS = {
    "oven","microwave","refrigerator","toaster"
}

WEAK_KITCHEN_OBJECTS = {
    "bottle","wine glass","cup","bowl","spoon","knife","fork"
}

KITCHEN_OBJECTS = (
    STRONG_KITCHEN_OBJECTS | WEAK_KITCHEN_OBJECTS
)

# =====================================================
# SWIN V2 TINY MODEL
# =====================================================

model = timm.create_model(

    "swinv2_tiny_window8_256",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "models/swinv2_tiny.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print(
    "Kitchen SwinV2 Tiny Loaded Successfully"
)

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
    "kitchen",
    "not_kitchen"
]

# =====================================================
# THRESHOLDS
# =====================================================

# Classifier thresholds

KITCHEN_CONFIDENCE_THRESHOLD = 0.75
NOT_KITCHEN_CONFIDENCE_THRESHOLD = 0.90

# YOLO thresholds

YOLO_CONFIDENCE_THRESHOLD = 0.40

# Weak object recovery

MIN_WEAK_OBJECTS_REQUIRED = 1

# =====================================================
# YOLO OBJECT DETECTION
# =====================================================

def detect_kitchen_objects(image):

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

            # Ignore weak detections

            if confidence < YOLO_CONFIDENCE_THRESHOLD:
                continue

            if object_name in KITCHEN_OBJECTS:

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

    print("Kitchen Validator")
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

def validate_kitchen(image):

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

    # -------------------------------------------------
    # YOLO DETECTION
    # -------------------------------------------------

    detected_objects = detect_kitchen_objects(
        image
    )

    strong_objects_found = [

        obj

        for obj in detected_objects

        if obj in STRONG_KITCHEN_OBJECTS

    ]

    weak_objects_found = [

        obj

        for obj in detected_objects

        if obj in WEAK_KITCHEN_OBJECTS

    ]

    has_strong_objects = (
        len(strong_objects_found) > 0
    )

    has_enough_weak_objects = (
        len(weak_objects_found)
        >=
        MIN_WEAK_OBJECTS_REQUIRED
    )

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

    # =================================================
    # CASE 1
    # CLASSIFIER SAYS KITCHEN
    # =================================================

    if predicted_class == "kitchen":

        # High confidence
        if confidence >= KITCHEN_CONFIDENCE_THRESHOLD:

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

        # Recovery using strong kitchen object
        if has_strong_objects:

            return build_response(

                "VALID",
                "VALID IMAGE",
                "yolo_strong",
                confidence,
                predicted_class,
                detected_objects,
                strong_objects_found,
                weak_objects_found

            )

        # Recovery using weak objects
        if has_enough_weak_objects:

            return build_response(
                "VALID",
                "VALID IMAGE",
                "yolo_weak",
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

    # =================================================
    # CASE 2
    # CLASSIFIER SAYS NOT KITCHEN
    # =================================================

    # Strong NOT-KITCHEN prediction

    if confidence >= NOT_KITCHEN_CONFIDENCE_THRESHOLD:

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
    # Low-confidence NOT-KITCHEN can be rescued
    # by YOLO evidence.

    if (
        confidence < 0.70
        and

     (
        has_strong_objects
         or
            has_enough_weak_objects
     )

    ):

        return build_response(
            "VALID",
            "VALID IMAGE",
            "yolo_override",
            confidence,
            predicted_class,
            detected_objects,
            strong_objects_found,
            weak_objects_found

        )

    # =================================================
    # FINAL INVALID
    # =================================================

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