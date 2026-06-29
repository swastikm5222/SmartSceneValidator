import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import timm
import easyocr
from PIL import Image
from torchvision import transforms

from validators.image_quality import validate_image_quality


# --------------------------------
# PATHS / CONFIG
# --------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "name_board_swin_tiny.pth"
)

# Swin-Tiny model was trained with different mean/std.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FAST_PATH_CONFIDENCE_THRESHOLD = 0.98
MIN_OCR_TEXT_LENGTH = 3


MAX_TEXT_REGIONS_FOR_FAST_PATH = 6
MIN_AVG_OCR_CONFIDENCE_FOR_FAST_PATH = 0.5
MIN_ALPHA_RATIO_FOR_FAST_PATH = 0.4


ADDRESS_DIGIT_RUN_PATTERN = re.compile(r"\d{3,}")


# --------------------------------
# DEVICE
# --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------
# SWIN TINY MODEL
# --------------------------------

def _load_model():
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=2,
    )
    

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


try:
    model = _load_model()
    _model_load_error = None
except Exception as exc:  # noqa: BLE001 - we want to surface this at call time
    model = None
    _model_load_error = str(exc)


# --------------------------------
# OCR
# --------------------------------

# English + Hindi. Override via env var if other scripts are needed, e.g.

OCR_LANGUAGES = os.environ.get("NAME_BOARD_OCR_LANGUAGES", "en,hi").split(",")

reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)


def _run_ocr(rgb_image):
    """Run OCR and return a list of (text, confidence) tuples so callers
    can assess how trustworthy the OCR output is, not just what it says.
    """
    detected = []
    seen = set()
    for _bbox, text, confidence in reader.readtext(rgb_image):
        text = text.strip()
        if len(text) >= MIN_OCR_TEXT_LENGTH and text not in seen:
            seen.add(text)
            detected.append((text, confidence))
    return detected


def _assess_ocr_quality(detected_pairs, full_text):
    """Return quality stats used to decide whether OCR output is clean
    enough to trust the model's fast path, or whether it looks like
    cluttered/garbled text from a busy poster rather than a name board.
    """
    num_regions = len(detected_pairs)
    confidences = [c for _, c in detected_pairs]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    stripped = full_text.replace(" ", "")
    alpha_count = sum(1 for ch in stripped if ch.isalpha())
    alpha_ratio = alpha_count / len(stripped) if stripped else 0.0

    is_noisy = bool(
    num_regions > MAX_TEXT_REGIONS_FOR_FAST_PATH
    or avg_confidence < MIN_AVG_OCR_CONFIDENCE_FOR_FAST_PATH
    or alpha_ratio < MIN_ALPHA_RATIO_FOR_FAST_PATH
)

    return {
        "num_text_regions": num_regions,
        "avg_ocr_confidence": round(avg_confidence, 4),
        "alpha_ratio": round(alpha_ratio, 4),
        "is_noisy": is_noisy,
    }


# --------------------------------
# IMAGE TRANSFORM
# --------------------------------

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# --------------------------------
# CLASS NAMES
# --------------------------------

CLASS_NAMES = ["name_board", "not_name_board"]


# --------------------------------
# NEGATIVE KEYWORDS
# --------------------------------

NEGATIVE_KEYWORDS = [
    # Sale / Promotion
    "sale", "discount", "offer", "offers", "mega sale", "festival offer",
    "limited offer", "clearance sale", "buy now", "special offer",

    # Political
    "vote", "election", "candidate", "party", "campaign",

    # Greeting
    "welcome", "congratulations", "happy birthday", "birthday",
    "anniversary", "best wishes",

    # Coaching / Ads
    "admission", "coaching", "training", "seminar", "workshop",
    "batch starting",

    # Road Signs / Posters
    "speed limit", "married", "divorce", "warning", "notice", "traffic",
    "danger", "caution", "school zone",

    # Hindi
    "छूट", "ऑफर", "सेल", "बिक्री", "मेगा सेल", "विशेष ऑफर", "मतदान",
    "चुनाव", "उम्मीदवार", "स्वागत", "जन्मदिन", "शुभकामनाएं", "प्रवेश",
    "कोचिंग", "प्रशिक्षण", "कार्यशाला",
]
NEGATIVE_KEYWORDS.extend([
    "bjp",
    "congress",
    "aap",
    "rally",
    "meeting",
    "conference",
    "convention",
    "adhiveshan",

    "भाजपा",
    "कांग्रेस",
    "अधिवेशन",
    "सभा",
    "रैली",
    "राजनीतिक",
    "पार्टी",
    "कार्यकर्ता"
])




# --------------------------------
# ADDRESS KEYWORDS
# --------------------------------

ADDRESS_KEYWORDS = [
    # English
    "house", "house no", "h.no", "h no", "ward", "block", "sector",
    "plot", "flat", "road", "street", "lane", "colony", "district",
    "village", "town", "city", "near", "post", "pin", "nagar",
    "residency", "residence", "villa", "niwas", "nivas", "bhawan",
    "apartment", "society",

    # Hindi
    "ग्राम", "वार्ड", "जिला", "नगर", "मोहल्ला", "गली", "मार्ग", "रोड",
    "पोस्ट", "निकट", "निवास", "प्लॉट", "ब्लॉक", "सेक्टर", "भवन",
    "अपार्टमेंट", "सोसायटी", "परिषद्",
]


def _build_keyword_pattern(keywords):

    escaped = [re.escape(kw.lower()) for kw in keywords]
    pattern = r"(?:\b|(?<=\s)|^)(?:%s)(?:\b|(?=\s)|$)" % "|".join(escaped)
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)


_NEGATIVE_PATTERN = _build_keyword_pattern(NEGATIVE_KEYWORDS)
_ADDRESS_PATTERN = _build_keyword_pattern(ADDRESS_KEYWORDS)


def _matched_keywords(pattern, keywords, text):
    """Return the list of keywords (from `keywords`) that match `text`
    via individual word-boundary checks. Used to report which specific
    keywords triggered a decision.
    """
    matches = []
    for kw in keywords:
        kw_pattern = r"\b%s\b" % re.escape(kw.lower())
        if re.search(kw_pattern, text, flags=re.IGNORECASE | re.UNICODE):
            matches.append(kw)
    return matches


# --------------------------------
# IMAGE NORMALIZATION HELPER
# --------------------------------

def _to_rgb_array(image):

    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path does not exist: {image}")
        pil_image = Image.open(image).convert("RGB")
        return np.array(pil_image)

    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        if image.ndim == 2:  # grayscale
            return np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:  # RGBA -> RGB
            return image[..., :3]
        return image

    raise TypeError(
        f"Unsupported image type: {type(image)!r}. "
        "Expected a file path, PIL.Image, or numpy.ndarray."
    )


# --------------------------------
# VALIDATOR
# --------------------------------

def validate_name_board(image):

    # --------------------------------
    # MODEL AVAILABILITY CHECK
    # --------------------------------

    if model is None:
        return {
            "status": "ERROR",
            "reason": f"Model failed to load: {_model_load_error}",
        }

    # --------------------------------
    # NORMALIZE INPUT IMAGE
    # --------------------------------

    try:
        rgb_image = _to_rgb_array(image)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": f"Could not read input image: {exc}",
        }

    # --------------------------------
    # QUALITY CHECK
    # --------------------------------

    try:
        quality_result = validate_image_quality(rgb_image)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": f"Image quality check failed: {exc}",
        }

    if quality_result.get("status") == "INVALID":
        return quality_result

    # --------------------------------
    # SWIN TINY INFERENCE
    # --------------------------------

    try:
        image_tensor = transform(rgb_image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence_tensor, predicted = torch.max(probabilities, 1)

        predicted_class = CLASS_NAMES[predicted.item()]
        model_confidence = round(confidence_tensor.item(), 4)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": f"Model inference failed: {exc}",
        }

 

    try:
        ocr_pairs = _run_ocr(rgb_image)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": f"OCR failed: {exc}",
            "model_confidence": model_confidence,
        }

    detected_text = [text for text, _confidence in ocr_pairs]
    full_text = " ".join(detected_text).lower()
    ocr_quality = _assess_ocr_quality(ocr_pairs, full_text)

    # --------------------------------
    # NEGATIVE KEYWORD CHECK (overrides the model, always)
    # --------------------------------

    if full_text and _NEGATIVE_PATTERN.search(full_text):
        matched = _matched_keywords(_NEGATIVE_PATTERN, NEGATIVE_KEYWORDS, full_text)
        return {
            "status": "INVALID",
            "reason": "INVALID IMAGE",
            "decision_basis": "negative_keyword",
            "matched_negative_keywords": matched,
            "model_confidence": model_confidence,
            "detected_text": detected_text,
            "ocr_quality": ocr_quality,
        }

 
    if (
        predicted_class == "name_board"
        and model_confidence >= FAST_PATH_CONFIDENCE_THRESHOLD
        and not ocr_quality["is_noisy"]
    ):
        return {
            "status": "VALID",
            "reason": "VALID IMAGE",
            "decision_basis": "model_confirmed_by_ocr",
            "model_confidence": model_confidence,
            "detected_text": detected_text,
            "ocr_quality": ocr_quality,
        }

    if not detected_text:
        return {
            "status": "INVALID",
            "reason": "INVALID IMAGE",
            "decision_basis": "ocr_no_text",
            "model_confidence": model_confidence,
        }

    # --------------------------------
    # ADDRESS SCORE
    # --------------------------------

    matched_address_keywords = _matched_keywords(
        _ADDRESS_PATTERN, ADDRESS_KEYWORDS, full_text
    )
    address_score = len(matched_address_keywords)

    has_address_like_number = bool(ADDRESS_DIGIT_RUN_PATTERN.search(full_text))
    if (has_address_like_number
    and len(matched_address_keywords) >= 1):
        address_score += 1

    # --------------------------------
    # FINAL DECISION
    # --------------------------------

    is_valid = address_score >= 1

    return {
        "status": "VALID" if is_valid else "INVALID",
        "reason": "VALID IMAGE" if is_valid else "INVALID IMAGE",
        "decision_basis": "address_score",
        "model_confidence": model_confidence,
        "detected_text": detected_text,
        "address_score": address_score,
        "matched_address_keywords": matched_address_keywords,
        "has_address_like_number": has_address_like_number,
        "ocr_quality": ocr_quality,
    }