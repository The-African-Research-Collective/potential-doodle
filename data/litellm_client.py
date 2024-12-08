import json

from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from litellm import acompletion, batch_completion

class Generation_Models(Enum):
    GPT4O = "gpt-4o-2024-08-06"
    GPT4O_MINI = "gpt-4o-mini-2024-07-18"
    CLAUDE_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_HAIKU = "claude-3-haiku-20240307"

@dataclass
class ModelCompletion:
    generation: Dict[str, Any]
    model: str

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def get_completion(prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
            model: Generation_Models,
            structured_object: Any,
            max_tokens: int = 512,
            temperature: float = 1.0,
            ) -> List[ModelCompletion]:
    """
    """
    completions = []

    if isinstance(prompt[0], list):
        response = batch_completion(
            model=model.value,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=structured_object
        )

        for model_resp in response:
            try:
                completions.append(ModelCompletion(generation=json.loads(model_resp.choices[0].message.content), model=model.value))
            except json.JSONDecodeError:
                completions.append(ModelCompletion(generation=model_resp.choices[0].message.content, model=model.value))

    elif isinstance(prompt[0], dict):
        response = await acompletion(
            model=model.value,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=structured_object
        )

        try:
            completions.append(ModelCompletion(generation=json.loads(response.choices[0].message.content), model=model.value))
        except json.JSONDecodeError:
            completions.append(ModelCompletion(generation=response.choices[0].message.content, model=model.value))

    else:
        raise ValueError("Prompt should be a list of dictionaries or a list of lists of dictionaries")

    return completions

