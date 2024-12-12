import json
from typing import Any, Dict

def json_parse_model_output(output: str) -> Dict[str, Any]:
    """
    This function parses the output of a model and returns a dictionary.
    it works by finding the first opening bracket and removing everything before it.
    Then it finds the last closing bracket and removes everything after it.
    Finally, it returns the JSON object.
    """
    
    # Find the first opening bracket and remove everything before it
    start = output.find("[")
    output = output[start:]

    # Find the last closing bracket and remove everything after it
    end = output.rfind("]")
    output = output[:end+1]

    return json.loads(output)