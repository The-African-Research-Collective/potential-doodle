import os
import json
import openai
import requests
from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
from litellm import acompletion, batch_completion
from openai import AzureOpenAI
from huggingface_hub import InferenceClient

class Generation_Models(Enum):
    GPT4O = "gpt-4o-2024-08-06"
    GPT4O_MINI = "gpt-4o-mini-2024-07-18"
    CLAUDE_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_HAIKU = "claude-3-haiku-20240307"
    AZURE_GPT4O = "azure_ai/newgpt4o"

    def __str__(self):
        return self.value

@dataclass
class ModelCompletion:
    generation: Dict[str, Any]
    model: str


def json_parse_model_output(output: str) -> Dict[str, Any]:
    
    # Find the first opening bracket and remove everything before it
    start = output.find("[")
    output = output[start:]

    # Find the last closing bracket and remove everything after it
    end = output.rfind("]")
    output = output[:end+1]

    return json.loads(output)

@retry(wait=wait_fixed(120), stop=stop_after_attempt(3))
async def get_completion(prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
            model: Generation_Models,
            structured_object: Any,
            max_tokens: int = 512,
            temperature: float = 1.0,
            ) -> List[ModelCompletion]:
    """
    """
    completions = []

    try:
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
                    raise ValueError(f"Error decoding response: {model_resp.choices[0].message.content}")

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
                raise ValueError(f"Error decoding response: {response.choices[0].message.content}")

        else:
            raise ValueError("Prompt should be a list of dictionaries or a list of lists of dictionaries")
    
    except openai.APITimeoutError as e:
        print(f"API Timeout Error: {e}")
        raise e
    except openai.APIConnectionError as e:
        print(f"API Connection Error: {e}")
        raise e
    except openai.RateLimitError as e:
        print(f"Rate Limit Error: {e}")
        raise e
    except Exception as e:
        print(f"Error: {e}")
        raise e

    return completions
       

@retry(wait=wait_fixed(20), stop=stop_after_attempt(3))
async def get_completion_azure_openai(prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
            model: Generation_Models,
            structured_object: Any,
            max_tokens: int = 512,
            temperature: float = 1.0,
            ) -> List[ModelCompletion]:
    
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_ENDPOINT"), 
        api_key=os.getenv("AZURE_API_KEY"),  
        api_version=os.getenv("AZURE_API_VERSION")
        )
    tools = [openai.pydantic_function_tool(structured_object)]

    def llm_inference(message):
        response = client.chat.completions.create(
            model="MODEL_DEPLOYMENT_NAME",
            messages=message,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        except TypeError as e:
            return {}

    completions = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(llm_inference, message) for message in prompt]
        for future in futures:
            response = future.result()
            try:
                completions.append(ModelCompletion(generation=response, model=model.value))
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding response")
    
    return completions

@retry(wait=wait_fixed(20), stop=stop_after_attempt(3))    
async def get_completion_tgi(prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
                        model: str,
                        max_tokens: int = 512,
                        temperature: float = 1.0,
                        ) -> List[ModelCompletion]:
    
    client = InferenceClient("http://localhost:8080")

    def llm_inference(message):
        response = client.chat_completion(
            messages=message,
            seed=42,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return json_parse_model_output(response.choices[0].message.content)
        except json.JSONDecodeError:
            return [{}]

    completions = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(llm_inference, message) for message in prompt]
        for future in futures:
            response = future.result()
            try:
                completions.append(ModelCompletion(generation=response, model=model))
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding response")
    
    return completions


