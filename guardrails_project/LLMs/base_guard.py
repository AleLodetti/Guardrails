from abc import ABC, abstractmethod
from guardrails_project.LLMs.base import Base

class BaseGuard(Base):
    """
    Questa classe è la classe astratta per i guardrail e definisce un nuovo
    metodo chaiamto validate_response()
    """

    def __init__(self):
        """Initialize the base LLM model."""
        super().__init__()
        print("Base Guardrail model initialized.")
        pass

    @abstractmethod
    def validate_response(self, chat: dict) -> dict:
        """Generates a response based on a conversation."""
        pass