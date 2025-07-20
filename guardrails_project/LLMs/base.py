
from abc import ABC, abstractmethod

class Base(ABC):
    """Base class for all LLM models."""

    def __init__(self):
        """Initialize the base LLM model."""
        print("Base model initialized.")
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Returns information about the model."""
        pass

    @abstractmethod
    def getTokenizer(self):
        """Returns the tokenizer associated with the model."""
        pass

    @abstractmethod
    def getModel(self):
        """Returns the model instance."""
        pass