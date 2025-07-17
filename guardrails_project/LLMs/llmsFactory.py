"""sfrutta il factory pattern per creare istanze di modelli LLM."""


from guardrails_project.LLMs.llama import llama
from guardrails_project.LLMs.llama_chat import llama_chat
from guardrails_project.LLMs.mistral import mistral

class LLMsFactory:

    @staticmethod
    def create_llm(model_name: str):
        """Creates an instance of the specified LLM model."""
        if model_name == "llama":
            return llama()
        elif model_name == "llama chat":
            return llama_chat()
        elif model_name == "mistral":
            return mistral()
        else:
            raise ValueError(f"Model {model_name} is not supported.")