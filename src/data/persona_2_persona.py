"""
"""
import os
import json
import time
import asyncio
import argparse
import jsonlines
from pydantic import BaseModel
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from persona_generation.prompt_template import PERSONA_2_PERSONA_GENERATION

from src.llms.base import Generation_Models, ModelProvider
from src.llms.litellm_client import LiteLLM
from src.llms.azure_client import AzureOPENAILLM
from src.llms.tgi_inference_client import TGI_client

load_dotenv()

DATASET_PATH = "files"

def prompt_processor(row: Dict):
    prompt = [ {"role": "user", "content": PERSONA_2_PERSONA_GENERATION + f"\n\n{json.dumps(row['persona'])}"}]
    
    row['templated_prompt'] = prompt
    return row

def batch_dataset_generator(dataset: List[Dict], batch_size):

    # Create prompt
    dataset = list(map(prompt_processor, dataset))

    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

async def main(args):
    
    if args.model == Generation_Models.AZURE_GPT4O:
        llm = AzureOPENAILLM(model_name=args.model)
    elif args.model in [Generation_Models.TGI_GEMINI_9B]:
        llm = TGI_client(model_name=args.model)
    else:
        llm = LiteLLM(model_name=args.model, model_provider=args.model_provider)

    with jsonlines.open(f"{DATASET_PATH}/{args.language}_personas.jsonl") as reader:
        dataset = list(reader)
    
    # create a text file for managing processed pages
    file_path = f"{args.data_directory}/persona_2_persona/{args.language}_processed_urls.txt"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_pages = f.read().splitlines()
    else:
        processed_pages = []
    
    # Filter out processed pages
    dataset = [row for row in dataset if row['id'] not in processed_pages]

    with jsonlines.open(f"{args.data_directory}/persona_2_persona/{args.language}_personas.jsonl", "a") as writer, open(file_path, "a") as f:
        for i, batch in tqdm(enumerate(batch_dataset_generator(dataset, args.batch_size)), total=len(dataset)//args.batch_size):
            batch_persona = [row['templated_prompt'] for row in batch]
            
            completions = await llm.completion(batch_persona, args.model, args.max_tokens, args.temperature)

            for i, persona in enumerate(batch):
                for completion in completions[i].generation:
                    if completion:
                        writer.write({"id": persona["id"], "persona": completion, "model": args.model})

                f.write(f"{persona['id']}\n")
                
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create personas")
    parser.add_argument("--language", type=str, help="Language code")
    parser.add_argument("--model", type=Generation_Models, choices=list(Generation_Models))
    parser.add_argument("--model_provider", type=str, choices=list(ModelProvider), required=False)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--data_directory", type=str, default="files", help="Data directory")
    args = parser.parse_args()

    asyncio.run(main(args))

