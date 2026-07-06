from validators.approach_property import validate_approach_property
from validators.interior_property import validate_interior_property
from validators.kitchen import validate_kitchen
from validators.name_board import validate_name_board
from validators.selfie import validate_selfie
from validators.property_front import validate_property_front



TAG_TO_VALIDATOR = {
    "selfie_with_person_met": validate_selfie,
    "approach_to_property": validate_approach_property,
    "front_image_with_name_board": validate_name_board,
    "interior_rooms_photograph": validate_interior_property,
    "kitchen_photograph": validate_kitchen,
    "front_view_of_property": validate_property_front
}


def get_validator_func_for_tag(tag: str):
    """
    Returns the appropriate validator function for a given tag.

    Args:
        tag (str): The tag for which to retrieve the validator. 
    """
    validator = TAG_TO_VALIDATOR.get(tag)
    if not validator:
        raise ValueError(f"No validator found for tag: {tag}")
    return validator
