"""sfrutta il factory pattern per creare istanze di modelli LLM."""


from guardrails_project.LLMs.llama import Llama
from guardrails_project.LLMs.llama_chat import Llama_chat
from guardrails_project.LLMs.mistral import Mistral

class LLMsFactory:

    @staticmethod
    def create_llm(model_name: str):
        """Creates an instance of the specified LLM model."""
        if model_name == "llama":
            return Llama()
        elif model_name == "llama chat":
            return Llama_chat()
        elif model_name == "mistral":
            return Mistral()
        else:
            raise ValueError(f"Model {model_name} is not supported.")