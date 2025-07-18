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
            fileName = "responsesLlamaChat.jsonl"
        elif model_name == "llama":
            fileName = "responsesLlama.jsonl"
        elif model_name == "mistral":
            fileName = "responsesMistral.jsonl"
        else:
            raise ValueError("Unsupported model name. Please choose from: Llama Chat, Llama, Mistral.")

        file_path = os.path.join(file_path, fileName)

        try:
            with open(file_path, 'a', encoding='utf-8') as file:
                json_line = json.dumps(data, ensure_ascii=False)
                file.write(json_line + '\n')
        except Exception as e:
            print(f"An error occurred while saving to {file_path}: {e}")