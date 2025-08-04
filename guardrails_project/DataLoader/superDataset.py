from abc import ABC, abstractmethod

class SuperDataset(ABC):
    """this is the base class for all Database. Since they have different formats I decided
    to implement a simple parser in order to retrieve the correct line each time."""

    def __init__(self):
        pass

    def loadData(self):
        pass

    @abstractmethod
    def parseInput(self) -> dict:
        """it returns the current row in a well known format
        Returns:
            dict: {
                prompt: ... ,
                type: ... ,
                something else: ... ,
            }
        """
        pass