from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Thresholds:
    """Central place for all quality thresholds — change here, affects all checks."""

    BLUR: float = 10.0

    BLACK_BRIGHTNESS: float = 8.0

    WHITE_BRIGHTNESS: float = 247.0

    # Minimum pixel dimensions accepted (width, height).
    MIN_WIDTH: int = 100
    MIN_HEIGHT: int = 100

    # Maximum file size in bytes to guard against memory-exhaustion attacks.
    MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB

class ValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass
class ValidationResult:
    """Structured, type-safe result returned by validate_image_quality()."""

    status: ValidationStatus
    reason: Optional[str] = None          # Present only when INVALID
    blur_score: Optional[float] = None    # Laplacian variance (higher = sharper)
    brightness: Optional[float] = None   # Mean grayscale brightness (0-255)
    warnings: list[str] = field(default_factory=list)  # Non-fatal advisory notes

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-friendly, drops None values)."""
        out: dict = {"status": self.status.value}
        if self.reason is not None:
            out["reason"] = self.reason
        if self.blur_score is not None:
            out["blur_score"] = self.blur_score
        if self.brightness is not None:
            out["brightness"] = self.brightness
        if self.warnings:
            out["warnings"] = self.warnings
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ValidationResult {self.status.value}" + (
            f" | {self.reason}>" if self.reason else ">"
        )
# ---------------------------------------------------------------------------
# Individual check helpers  (private — underscore-prefixed)
# ---------------------------------------------------------------------------

def _check_array_integrity(image: np.ndarray) -> Optional[str]:
    """Return a failure reason string if the raw ndarray is unusable, else None."""
    if image is None:
        return "Image is None"
    if not isinstance(image, np.ndarray):
        return f"Expected np.ndarray, got {type(image).__name__}"
    if image.size == 0:
        return "Image array is empty (size == 0)"
    if image.ndim not in (2, 3):
        return f"Unexpected array dimensions: {image.ndim} (expected 2 or 3)"
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return f"Unexpected channel count: {image.shape[2]}"
    return None


def _check_dimensions(image: np.ndarray) -> Optional[str]:
    """Return a failure reason if the image is too small to be useful."""
    h, w = image.shape[:2]
    if w < Thresholds.MIN_WIDTH or h < Thresholds.MIN_HEIGHT:
        return (
            f"Image too small ({w}×{h} px); "
            f"minimum is {Thresholds.MIN_WIDTH}×{Thresholds.MIN_HEIGHT} px"
        )
    return None


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Safely convert any supported image format to 8-bit grayscale."""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.shape[2] == 4:
        # BGRA → BGR → gray  (drop alpha first)
        gray = cv2.cvtColor(
            cv2.cvtColor(image, cv2.COLOR_BGRA2BGR),
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure uint8 for consistent metric computation
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _compute_blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher value = sharper image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity over the whole grayscale image (0–255)."""
    return float(np.mean(gray))


def _check_uniform_color(gray: np.ndarray) -> Optional[str]:
  
    std = float(np.std(gray))
    if std < 1.0:
        mean = float(np.mean(gray))
        return f"Uniform solid-color image (mean brightness {mean:.1f}, std {std:.2f})"
    return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_image_quality(
    image: np.ndarray,
    *,
    thresholds: type[Thresholds] = Thresholds,
) -> ValidationResult:
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Array integrity
    # ------------------------------------------------------------------
    integrity_error = _check_array_integrity(image)
    if integrity_error:
        logger.warning("Image integrity check failed: %s", integrity_error)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=integrity_error,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 2. Minimum dimensions
    # ------------------------------------------------------------------
    dim_error = _check_dimensions(image)
    if dim_error:
        logger.warning("Image dimension check failed: %s", dim_error)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=dim_error,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 3. Convert to grayscale (all metric checks work on gray)
    # ------------------------------------------------------------------
    try:
        gray = _to_gray(image)
    except cv2.error as exc:
        logger.error("cv2 error during grayscale conversion: %s", exc)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=f"Color conversion failed: {exc}",
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 4. Blur check
    # ------------------------------------------------------------------
    blur_score = round(_compute_blur_score(gray), 2)
    if blur_score < thresholds.BLUR:
        logger.info("Image rejected — blurry (score=%.2f, threshold=%.2f)", blur_score, thresholds.BLUR)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason="Image is too blurry",
            blur_score=blur_score,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 5. Brightness checks (black / white)
    # ------------------------------------------------------------------
    brightness = round(_compute_brightness(gray), 2)

    if brightness < thresholds.BLACK_BRIGHTNESS:
        logger.info("Image rejected — too dark (brightness=%.2f)", brightness)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason="Image is too dark (near black)",
            brightness=brightness,
            warnings=warnings,
        )

    if brightness > thresholds.WHITE_BRIGHTNESS:
        logger.info("Image rejected — overexposed (brightness=%.2f)", brightness)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason="Image is overexposed (near white)",
            brightness=brightness,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 6. Uniform-color check (catches solid grey etc. missed above)
    # ------------------------------------------------------------------
    uniform_error = _check_uniform_color(gray)
    if uniform_error:
        logger.info("Image rejected — uniform color: %s", uniform_error)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=uniform_error,
            brightness=brightness,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 7. Advisory warnings (non-fatal — image is still accepted)
    # ------------------------------------------------------------------
    if blur_score < thresholds.BLUR * 3:
        warnings.append(
            f"Image sharpness is low (blur_score={blur_score}); "
            "classifier accuracy may be reduced."
        )
    if brightness < 20:
        warnings.append(
            f"Image is very dark (brightness={brightness}); "
            "consider requesting better lighting."
        )
    if brightness > 230:
        warnings.append(
            f"Image is very bright (brightness={brightness}); "
            "slight overexposure detected."
        )

    # ------------------------------------------------------------------
    # 8. All checks passed
    # ------------------------------------------------------------------
    logger.debug("Image accepted (blur=%.2f, brightness=%.2f)", blur_score, brightness)
    return ValidationResult(
        status=ValidationStatus.VALID,
        blur_score=blur_score,
        brightness=brightness,
        warnings=warnings,
    )

def validate_image_file(
    file_path: str,
    *,
    max_bytes: int = Thresholds.MAX_BYTES,
    thresholds: type[Thresholds] = Thresholds,
) -> ValidationResult:

    import os

    if not os.path.isfile(file_path):
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=f"File not found: {file_path}",
        )

    file_size = os.path.getsize(file_path)
    if file_size > max_bytes:
        mb = file_size / (1024 * 1024)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=f"File too large ({mb:.1f} MB); maximum is {max_bytes // (1024 * 1024)} MB",
        )

    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=f"cv2.imread could not decode the file: {file_path}",
        )

    return validate_image_quality(image, thresholds=thresholds)


# ---------------------------------------------------------------------------
# Convenience wrapper: validate from raw bytes (e.g. HTTP upload body)
# ---------------------------------------------------------------------------

def validate_image_bytes(
    data: bytes,
    *,
    max_bytes: int = Thresholds.MAX_BYTES,
    thresholds: type[Thresholds] = Thresholds,
) -> ValidationResult:
   
    if not isinstance(data, (bytes, bytearray)):
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=f"Expected bytes, got {type(data).__name__}",
        )

    if len(data) == 0:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason="Empty byte payload",
        )

    if len(data) > max_bytes:
        mb = len(data) / (1024 * 1024)
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason=f"Payload too large ({mb:.1f} MB); maximum is {max_bytes // (1024 * 1024)} MB",
        )

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        return ValidationResult(
            status=ValidationStatus.INVALID,
            reason="Could not decode image bytes — unsupported or corrupt format",
        )

    return validate_image_quality(image, thresholds=thresholds)