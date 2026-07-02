from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np

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
