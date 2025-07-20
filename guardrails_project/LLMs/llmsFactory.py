"""sfrutta il factory pattern per creare istanze di modelli LLM."""

from guardrails_project.LLMs.base_llm import BaseLLM
from guardrails_project.LLMs.llama import Llama
from guardrails_project.LLMs.llama_chat import Llama_chat
from guardrails_project.LLMs.mistral import Mistral

class LLMsFactory:

    @staticmethod
    def create_llm(model_name: str):
        """Creates an instance of the specified LLM model."""
        if model_name == "llama":
            print("Creating Llama model instance...")
            llm = Llama()
            print("Llama model instance created.",llm.getModel())
            return llm
        elif model_name == "llama chat":
            return Llama_chat()
        elif model_name == "mistral":
            return Mistral()
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
    @staticmethod
    def create_llm_custom(model_name: str, model=None, tokenizer=None):
        """Creates an instance of the specified LLM model with custom model and tokenizer."""
        print(f"Creating LLM instance for model: {model_name} with custom model and tokenizer.")
        if model_name == "llama":
            return Llama(model=model, tokenizer=tokenizer)
        elif model_name == "llamaChat":
            return Llama_chat(model=model, tokenizer=tokenizer)
        elif model_name == "mistral":
            return Mistral(model=model, tokenizer=tokenizer)
        else:
            raise ValueError(f"Model {model_name} is not supported.")