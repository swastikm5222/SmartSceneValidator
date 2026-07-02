import os

import timm
import torch


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def _model_path(filename):
    return os.path.join(
        PROJECT_ROOT,
        "models",
        filename
    )


def load_model(model_name, model_path, num_classes=2, *, weights_only=None):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    torch_load_kwargs = {
        "map_location": device
    }

    if weights_only is not None:
        torch_load_kwargs["weights_only"] = weights_only

    model.load_state_dict(
        torch.load(
            model_path,
            **torch_load_kwargs
        )
    )

    model.to(device)
    model.eval()

    return model


def _load_logged_model(
    model_name,
    model_path,
    device_message=None,
    success_message=None
):
    if device_message:
        print(
            f"{device_message} : {device}"
        )

    loaded_model = load_model(
        model_name,
        model_path
    )

    if success_message:
        print(
            success_message
        )

    return loaded_model


MODEL_LOAD_ERRORS = {}


def _load_name_board_model():
    try:
        MODEL_LOAD_ERRORS["name_board"] = None
        return load_model(
            "swin_tiny_patch4_window7_224",
            _model_path("name_board_swin_tiny.pth"),
            weights_only=True
        )
    except Exception as exc:  # noqa: BLE001 - preserve validator error response
        MODEL_LOAD_ERRORS["name_board"] = str(exc)
        return None


MODELS = {
    "front_property": _load_logged_model(
        "swinv2_tiny_window8_256",
        _model_path("front_property_swinv2_tiny_best.pth"),
        "Front Property Validator Device",
        "Front Property SwinV2 Tiny Loaded Successfully"
    ),
    "approach_property": load_model(
        "swinv2_tiny_window8_256",
        _model_path("approach_property_swinv2_tiny.pth")
    ),
    "interior_property": _load_logged_model(
        "swinv2_tiny_window8_256",
        _model_path("interior_property_swinv2_tiny_best.pth"),
        "Interior Property Validator Device",
        "Interior Property SwinV2 Tiny Loaded Successfully"
    ),
    "kitchen": _load_logged_model(
        "swinv2_tiny_window8_256",
        _model_path("swinv2_tiny.pth"),
        "Kitchen Validator Device",
        "Kitchen SwinV2 Tiny Loaded Successfully"
    ),
    "name_board": _load_name_board_model()
}
