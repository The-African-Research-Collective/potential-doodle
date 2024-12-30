import os
import json
import openai
import asyncio

from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
from openai import AzureOpenAI
from src.llms.base import BaseLLM, ModelCompletion, Generation_Models, ModelProvider

load_dotenv()


class AzureOPENAILLM(BaseLLM):
    def __init__(self,
            model_name: Generation_Models = Generation_Models.AZURE_GPT4O,
            model_provider:ModelProvider = ModelProvider.AZURE):
        super().__init__(model_name)
        self.model_provider = model_provider
        self._check_environment_variables()

        self.client  = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_API_BASE"), 
            api_key=os.getenv("AZURE_API_KEY"),  
            api_version=os.getenv("AZURE_API_VERSION")
            )

    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(3))
    async def completion(
            self,
            prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
            structured_object: Optional[Any] = None,
            **generation_kwargs: Any,
            ) -> List[ModelCompletion]:
        
        """
        Generate completions for the given prompt using the model.
        """
        temperature = generation_kwargs.get("temperature", 1.0)
        max_tokens = generation_kwargs.get("max_tokens", 512)
        
        if structured_object:
            tools = [openai.pydantic_function_tool(structured_object)]
        else:
            tools = None
        
        def llm_inference(message: List[Dict[str, str]]):
            try:
                response = self.client.chat.completions.create(
                    model="newgpt4o",
                    messages=message,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            except TypeError as e:
                return {}
            except openai.BadRequestError as e:
                return {}
        
        completions = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(llm_inference, message) for message in prompt]
            for future in futures:
                response = future.result()
                completions.append(ModelCompletion(generation=response,
                                            model=self.model_name.value))
        
        return completions

async def _test():
    model = AzureOPENAILLM()
    prompt = [{"message": "Hello, how are you?", "role": "user"}]
    completions = await model.completion(prompt)
    print(completions)

if __name__ == "__main__":
    asyncio.run(_test())