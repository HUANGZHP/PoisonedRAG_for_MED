# Purpose: Implements src/models/__init__.py in the PoisonedRAG project.

import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'palm2':
        from .PaLM2 import PaLM2

        model = PaLM2(config)
    elif provider == 'vicuna':
        from .Vicuna import Vicuna

        model = Vicuna(config)
    elif provider == 'gpt':
        from .GPT import GPT

        model = GPT(config)
    elif provider in ['llama', 'hf-local', 'qwen', 'deepseek']:
        from .Llama import Llama

        model = Llama(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model