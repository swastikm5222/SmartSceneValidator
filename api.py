from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np

import os
from ultralytics import YOLO

from validators.selfie import validate_selfie
from validators.name_board import validate_name_board
from validators.kitchen import validate_kitchen
from validators.property_front import validate_property_front
from validators.approach_property import validate_approach_property
from validators.interior_property import validate_interior_property as validate_interior_property_validator

app = FastAPI(
    title="SMC Image Validation API",
    description="API for validating uploaded images",
    version="1.0.0"
)

# Load YOLO once during app startup to avoid import-time weight loading.
@app.on_event("startup")
def _startup_load_yolo():
    yoloweights = os.environ.get("YOLO_WEIGHTS_PATH", "yolov8n.pt")
    app.state.yolo_model = YOLO(yoloweights)

    # Inject into validators modules that expect module-level `yolo_model`.
    try:
        import validators.kitchen as kitchen_validator
        kitchen_validator.yolo_model = app.state.yolo_model
    except Exception:
        pass

    try:
        import validators.interior_property as interior_validator
        interior_validator.yolo_model = app.state.yolo_model
    except Exception:
        pass



TAG_TO_VALIDATOR = {
    "selfie_with_person_met": validate_selfie,
    "approach_to_property": validate_approach_property,
    "front_image_with_name_board": validate_name_board,
    "interior_rooms_photograph": validate_interior_property_validator,
    "kitchen_photograph": validate_kitchen,
    "front_view_of_property": validate_property_front
}


# --------------------------------------------------
# GLOBAL EXCEPTION HANDLER
# --------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "ERROR",
            "reason": f"Internal server error: {exc}",
        },
    )


@app.get("/")
def home():
    return {
        "message": "SMC Image Validation API is running"
    }


# --------------------------------------------------
# SHARED IMAGE DECODING HELPER
# --------------------------------------------------

def _decode_uploaded_image(contents: bytes):

    np_arr = np.frombuffer(contents, np.uint8)
    bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if bgr_image is None:
        return None

    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def _invalid_image_response():
    return {
        "status": "INVALID",
        "reason": "Unable to read image",
    }


# --------------------------------------------------
# TAG-BASED VALIDATOR DISPATCH
# --------------------------------------------------

@app.post("/validate")
async def validate_image_by_tag(
    tag: str = Form(...),
    file: UploadFile = File(...)
):
    validator = TAG_TO_VALIDATOR.get(tag)

    if validator is None:
        raise HTTPException(
            status_code=400,
            detail="Unknown tag"
        )

    contents = await file.read()
    image = _decode_uploaded_image(contents)

    if image is None:
        return _invalid_image_response()

    result = validator(image)

    return result


# --------------------------------------------------
# SELFIE VALIDATOR
# --------------------------------------------------

@app.post("/validate/selfie")
async def validate_selfie_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = _decode_uploaded_image(contents)

        if image is None:
            return _invalid_image_response()

        return validate_selfie(image)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "reason": f"Selfie validation failed: {exc}"},
        )


# --------------------------------------------------
# NAME BOARD VALIDATOR
# --------------------------------------------------

@app.post("/validate/name-board")
async def validate_name_board_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = _decode_uploaded_image(contents)

        if image is None:
            return _invalid_image_response()

        return validate_name_board(image)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "reason": f"Name board validation failed: {exc}"},
        )


# --------------------------------------------------
# KITCHEN VALIDATOR
# --------------------------------------------------

@app.post("/validate/kitchen")
async def validate_kitchen_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = _decode_uploaded_image(contents)

        if image is None:
            return _invalid_image_response()

        return validate_kitchen(image)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "reason": f"Kitchen validation failed: {exc}"},
        )


# --------------------------------------------------
# PROPERTY FRONT VALIDATOR
# --------------------------------------------------

@app.post("/validate/property-front")
async def validate_property_front_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = _decode_uploaded_image(contents)

        if image is None:
            return _invalid_image_response()

        return validate_property_front(image)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "reason": f"Property front validation failed: {exc}"},
        )


# --------------------------------------------------
# APPROACH PROPERTY VALIDATOR
# --------------------------------------------------

@app.post("/validate/approach-property")
async def validate_approach_property_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = _decode_uploaded_image(contents)

        if image is None:
            return _invalid_image_response()

        return validate_approach_property(image)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "reason": f"Approach property validation failed: {exc}"},
        )


# --------------------------------------------------
# INTERIOR PROPERTY VALIDATOR
# --------------------------------------------------

@app.post("/validate/interior-property")
async def validate_interior_property(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = _decode_uploaded_image(contents)

        if image is None:
            return _invalid_image_response()

        return validate_interior_property_validator(image)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "reason": f"Interior property validation failed: {exc}"},
        )

