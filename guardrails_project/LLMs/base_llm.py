""""
Base class for all LLM models.
This class defines the interface that all LLM models must implement.
It includes methods for generating responses, retrieving model information,
and accessing the tokenizer and model instance.
"""

from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Base class for all LLM models."""

    @abstractmethod
    def get_model_info(self) -> dict:
        """Returns information about the model."""
        pass

    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Generates a response based on a list of messages."""
        pass

    @abstractmethod
    def getTokenizer(self):
        """Returns the tokenizer associated with the model."""
        pass

    @abstractmethod
    def getModel(self):
        """Returns the model instance."""
        pass