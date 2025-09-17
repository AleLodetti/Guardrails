from abc import ABC, abstractmethod
from guardrails_project.LLMs.base import Base

class BaseGuard(Base):
    """
    This class is the abstract class for guardrails and defines a new
    method called validate_response().
    """

    def __init__(self):
        """Initialize the base LLM model."""
        super().__init__()
        print("Base Guardrail model initialized.")
        pass

    @abstractmethod
    def validate_response(self, chat: dict) -> dict:
        """
        This method checks if the chat is safe, meaning that there is not attempt of jailbreaking the LLM.

        Args:
            chat (dict): is a dictionary that contains the prompt and the LLM response. Every guardrail has its own format.

        Returns:
            dict: it return a dictionary of a format:
                {
                "status": "unsafe",
                "reason": "S1: Hate Speech",
                "raw_output": "unsafe, S1: Hate Speech",
                "chat": "..."
                }
        """
        pass