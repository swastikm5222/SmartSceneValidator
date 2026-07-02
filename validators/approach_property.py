import torch
import torch.nn.functional as F
from torchvision import transforms

from models.model_manager import MODELS, device


# ----------------------------
# MODEL
# ----------------------------

model = MODELS["approach_property"]


# ----------------------------
# TRANSFORM
# ----------------------------

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ----------------------------
# CLASS NAMES
# ----------------------------

CLASS_NAMES = [
    "approach_property",
    "not_approach_property"
]


# ----------------------------
# VALIDATOR
# ----------------------------

def validate_approach_property(image):

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = CLASS_NAMES[
        predicted.item()
    ]

    confidence = confidence.item()

    if (
        predicted_class == "approach_property"
        and confidence >= 0.90
    ):

        return {
            "status": "VALID",
            "reason": "VALID IMAGE",
            "confidence": round(confidence, 4)
        }

    return {
        "status": "INVALID",
        "reason": "INVALID IMAGE",
        "confidence": round(confidence, 4)
    }
