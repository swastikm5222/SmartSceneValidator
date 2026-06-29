import torch
import torch.nn.functional as F
import timm
from torchvision import transforms


# ----------------------------
# DEVICE
# ----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------
# MODEL
# ----------------------------

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "models/property_front_classifier.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


# ----------------------------
# TRANSFORM
# ----------------------------

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ----------------------------
# CLASS NAMES
# ----------------------------

CLASS_NAMES = [
    "front_property",
    "not_front_property"
]


# ----------------------------
# VALIDATOR
# ----------------------------

def validate_property_front(image):

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

    # Strict threshold
    if (
        predicted_class == "front_property"
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