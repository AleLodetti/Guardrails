""""
Base class for all LLM models.
This class defines the interface that all LLM models must implement.
It includes methods for generating responses, retrieving model information,
and accessing the tokenizer and model instance.
"""

from abc import ABC, abstractmethod
from guardrails_project.LLMs.base import Base

class BaseLLM(Base):
    """Base class for all LLM models."""

    def __init__(self):
        """Initialize the base LLM model."""
        super().__init__()
        print("Base LLM model initialized.")
        pass

    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Generates a response based on a list of messages."""
        pass