"""
"""
import os
import time
import asyncio
import argparse
import jsonlines
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

from persona_generation.prompt_template import PERSONA_GENERATION
from litellm_client import get_completion, Generation_Models, get_completion_azure_openai

load_dotenv()

DATASET_NAME="wikimedia/wikipedia"

class Persona(BaseModel):
  countries: list[str]
  languages: list[str]
  persona: str

class PersonaList(BaseModel):
    persona_list: list[Persona]

def prompt_processor(example):

    # get the first 200 words of the text
    text = example["text"].split()[:200]
    text = " ".join(text)

    prompt = [ {"role": "system", "content": PERSONA_GENERATION, "cache_control": {"type": "ephemeral"}}, 
              {"role": "user", "content": text}]
    
    example['templated_prompt'] = prompt
    return example


def batch_dataset_generator(dataset, batch_size):

    # Create prompt
    dataset = dataset.map(prompt_processor)

    # Shuffle the dataset
    dataset = dataset.shuffle()

    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]


async def main(args):
    dataset = load_dataset(DATASET_NAME, f"20231101.{args.language}")
    model = Generation_Models

    # create a text file for managing processed pages
    file_path = f"{args.data_directory}/{args.language}_processed_urls.txt"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_pages = f.read().splitlines()
    else:
        processed_pages = []
    
    # filter dataset to pages that have more than 1000 words
    dataset = dataset.filter(lambda x: len(x["text"].split()) > 500)

    print(f"Processing {len(dataset['train'])} pages")

    # Remove pages that have already been processed
    dataset = dataset.filter(lambda x: str(x["id"]) not in set(processed_pages))

    print(f"Processing {len(dataset['train'])} pages")
    
    with jsonlines.open(f"{args.data_directory}/{args.language}_personas.jsonl", "a") as writer, open(file_path, "a") as f:
        # iterate over the dataset and use multiprocessing to generate personas

        for i, batch in tqdm(enumerate(batch_dataset_generator(dataset['train'], args.batch_size)), total=len(dataset)//args.batch_size):
            print(f"Processing batch {i}")

            # results = await get_completion(batch['templated_prompt'], args.model, PersonaList)

            results = await get_completion_azure_openai(batch['templated_prompt'], args.model, PersonaList)

            for i, id, url, res in zip(range(len(batch)), batch["id"], batch["url"], results):
                model_used = res.model
                try:

                    for persona in res.generation['persona_list']:
                        writer.write({"id": id, "url": url, "persona": persona, "model": model_used})
                    f.write(f"{id}\n")
                except KeyError:
                    print(f"Error processing {id} with model {model_used}")
            
            time.sleep(10)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, default="files")
    parser.add_argument("--language", type=str)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--model", type=Generation_Models, choices=list(Generation_Models))
    args = parser.parse_args()

    asyncio.run(main(args))
