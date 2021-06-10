"""File for json utils."""
import json
from typing import Dict, NoReturn


def load_json(path: str) -> Dict:
    """Load a json and handle IO stream opening and closing.

    Args:
        path (str): The path of the json to load

    Returns:
        Dict: The dict associated with the json
    """
    with open(path, "r") as file:
        return json.load(file)


def write_json(path: str, data: Dict) -> NoReturn:
    """Write a dict to a json object.

    Args:
        path (str): The path to write the json to.
        data (Dict): The dict data to write

    Returns:
        NoReturn: There is no return for this function.
    """
    with open(path, "w") as file:
        return json.dump(data, file)
