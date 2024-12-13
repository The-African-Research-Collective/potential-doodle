"""
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import asyncio
import argparse
import jsonlines
import random
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv
from typing import Dict, List
from pydantic import BaseModel

from src.data.persona_generation.prompt_template import PROMPT_GENERATION
from src.llms.base import Generation_Models, ModelProvider
from src.llms.litellm_client import LiteLLM
from src.llms.azure_client import AzureOPENAILLM
from src.llms.tgi_inference_client import TGI_client
from src.constant import TARGET_LANGUAGES, TARGET_DOMAINS, SUBDOMAINS


SEED_PROMPTS_FILE = "src/data/persona_generation/seed_prompts.jsonl"
CONSTRAINTS_FILE = "src/data/persona_generation/ifeval_instructions.csv"
PERSONA_FILE = "files/wikipedia_personas/en_personas.jsonl"

class GeneratedPrompt(BaseModel):
    prompt: str
    language: str

class GeneratedPromptList(BaseModel):
    prompts: List[GeneratedPrompt]

def _load_constraints():
    constraints_df = pd.read_csv(CONSTRAINTS_FILE)
    return constraints_df.to_json(orient='records', lines=True)

def _load_seed_prompts():
    with jsonlines.open(SEED_PROMPTS_FILE) as reader:
        seed_prompts_list = list(reader)
    return seed_prompts_list

def _load_personas():
    with jsonlines.open(PERSONA_FILE) as reader:
        personas_list = list(reader)
    return personas_list

def _build_prompt_message(seed_prompt: str, seed_language: str, persona: str, subdomain: str, target_domain: str, language: str):
    system_prompt = PROMPT_GENERATION.replace("{seed_language}", seed_language).replace("{seed_prompt}", seed_prompt)
    user_prompt = f"""Persona: {persona['persona']['persona']}
Domain: {subdomain}
Style: {target_domain}
Language: {language}"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    

async def main(args):
    constraints_list = _load_constraints()
    seed_prompts_list = _load_seed_prompts()
    persona_list = _load_personas()

    seed_prompts_list = [prompt for prompt in seed_prompts_list if prompt['domain'] == args.domain]

    if args.model == Generation_Models.AZURE_GPT4O:
        llm = AzureOPENAILLM(model_name=args.model)
    elif args.model in [Generation_Models.TGI_GEMINI_9B]:
        llm = TGI_client(model_name=args.model)
    else:
        llm = LiteLLM(model_name=args.model, model_provider=args.model_provider)


    for i in range(100):
        # Randomly sample a subdomain
        subdomain = random.choice(SUBDOMAINS)
        target_domain = random.choice(TARGET_DOMAINS)
        language = random.choice(TARGET_LANGUAGES)
        sampled_seed_prompt = random.choice(seed_prompts_list)
        sample_persona = random.choice(persona_list)

        prompt = _build_prompt_message(sampled_seed_prompt['prompt'], sampled_seed_prompt['language'], sample_persona, "chemistry", target_domain, language)
    
        print(prompt)

        results = await llm.completion([prompt], GeneratedPromptList)

        print(results)


        break


    pass

if  __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Prompts")
    parser.add_argument("--domain", type=str, default="instruction_following", help="Domain of the prompt", choices=["instruction_following", "math"])
    parser.add_argument("--model", type=Generation_Models, choices=list(Generation_Models))
    parser.add_argument("--model_provider", type=ModelProvider, choices=list(ModelProvider), required=False)
    args = parser.parse_args()
    asyncio.run(main(args))
