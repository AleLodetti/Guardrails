import json
import os
from typing import Dict, Any

class PromptSaver:

    def __init__(self, file_path: str, model_name: str):
        if model_name == "llamachat":
            fileName = "responsesLlamaChat.jsonl"
        elif model_name == "llama":
            fileName = "responsesLlama.jsonl"
        elif model_name == "mistral":
            fileName = "responsesMistral.jsonl"
        elif model_name == "llamaguard":
            fileName = "resultLlamaguard.jsonl"
        else:
            raise ValueError("Unsupported model name. Please choose from: Llama Chat, Llama, Mistral.")

        self.file_path = os.path.join(file_path, fileName)
        self.file = None
    
    def __enter__(self):
        self.file = open(self.file_path, 'w', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def saveOnJsonFile(self, file_path: str, data: Dict[str, str], model_name: str) -> None:
        """
        Saves the given data on a JSON file.

        Args:
            data (Dict[str, Any]): The data to save.
        """
        try:
            json_line = json.dumps(data, ensure_ascii=False)
            self.file.write(json_line + '\n')
            self.file.flush() 
        except Exception as e:
            print(f"An error occurred while saving to {file_path}: {e}")

    def saveResponseOnJsonl(self, data: Dict[str, str]) -> None:
        """
        This method is used to save the guardrail response  

        Args:
            data (Dict[str, str]): this is what I actually save, it has the following format
                {
                    'originalDetaction': safe or unsafe
                    'guardrailDetection': safe or unsafe
                    'groundTruth': safe or unsafe
                }

        Raises:
            ValueError: in case the name of the guardrail is invalid
        """

        try:
            json_line = json.dumps(data, ensure_ascii=False)
            self.file.write(json_line + '\n')
            self.file.flush() 
        except Exception as e:
            print(f"An error occurred while saving: {e}")
        