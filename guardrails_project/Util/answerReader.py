import json
import os
from typing import Dict, Any

class PromptReader:

    def __init__(self, file_path: str, model_name: str):
        """
        Initializes the PromptReader with the file path and model name.

        Args:
            file_path (str): The path to the JSON file.
            model_name (str): The name of the model to determine the file name.
        """
        self.file_path = os.path.join(file_path, model_name)
        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, 'r', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def getNextDict(self) -> Dict[str, Any]:
        """
        Reads the next dictionary from the JSON file.

        Returns:
            Dict[str, Any]: The next dictionary from the JSON file.
        """
        if self.file is None:
            raise ValueError("File not opened. Use 'with' statement to open the file.")
    
        while True:
            line = self.file.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return None