import torch
import torch.nn.functional as F

from torchvision import transforms

from models.model_manager import MODELS, device
from validators.image_quality import (
    validate_image_quality
)

# ---------------------------------------------------
# SWIN V2 TINY MODEL
# ---------------------------------------------------

model = MODELS["front_property"]

# ---------------------------------------------------
# IMAGE TRANSFORM
# ---------------------------------------------------

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((256, 256)),

    transforms.ToTensor()

])

# ---------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------

CLASS_NAMES = [

    "front_property",

    "not_front_property"

]

# ---------------------------------------------------
# CONFIDENCE THRESHOLD
# ---------------------------------------------------

FRONT_PROPERTY_CONFIDENCE_THRESHOLD = 0.90

# ---------------------------------------------------
# MAIN VALIDATOR
# ---------------------------------------------------

def validate_property_front(image):

    # ------------------------------------------
    # IMAGE QUALITY CHECK
    # ------------------------------------------

    quality_result = validate_image_quality(
        image
    )

    if not quality_result.is_valid:

        return quality_result.to_dict()

    # ------------------------------------------
    # IMAGE PREPROCESSING
    # ------------------------------------------

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    # ------------------------------------------
    # MODEL INFERENCE
    # ------------------------------------------

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

    predicted_class = CLASS_NAMES[
        prediction.item()
    ]

    confidence = float(
        confidence.item()
    )

    # ------------------------------------------
    # DEBUG LOGS
    # ------------------------------------------

    print("=" * 60)
    print("Front Property Validator")
    print("Prediction :", predicted_class)
    print("Confidence :", round(confidence, 4))
    print("=" * 60)

    # ------------------------------------------
    # RESULT HELPER
    # ------------------------------------------

    def _result(status):

        return {

            "status": status,

            "reason": (
                "VALID IMAGE"
                if status == "VALID"
                else "INVALID IMAGE"
            ),

            "decision_basis": "classifier",

            "confidence": round(
                confidence,
                4
            ),

            "classifier_prediction": predicted_class

        }

    # ------------------------------------------
    # DECISION LOGIC
    # ------------------------------------------

    if (

        predicted_class == "front_property"

        and

        confidence >= FRONT_PROPERTY_CONFIDENCE_THRESHOLD

    ):

        return _result(
            "VALID"
        )

    return _result(
        "INVALID"
    )
