import json
import os
from typing import Dict, Any

class PromptSaver:
    @staticmethod
    def saveOnJsonFile(file_path: str, data: Dict[str, str], model_name: str) -> None:
        """
        Saves the given data to a JSON file.

        Args:
            file_path (str): The path to the JSON file.
            data (Dict[str, Any]): The data to save.
        """

        if model_name == "llama chat":
            fileName = "responsesLlamaChat.json"
        elif model_name == "llama":
            fileName = "responsesLlama.json"
        elif model_name == "mistral":
            fileName = "responsesMistral.json"
        else:
            raise ValueError("Unsupported model name. Please choose from: Llama Chat, Llama, Mistral.")

        file_path = os.path.join(file_path, fileName)

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = []

        existing_data.append(data)

        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"An error occurred while saving to {file_path}: {e}")
        