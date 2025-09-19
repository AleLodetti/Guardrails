from constants import GUARDRAIL_TOKEN
from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd

class Wildjailbreak(SuperDataset):

    def __init__(self):
        super().__init__()
    
    def loadData(self):
        login(token = GUARDRAIL_TOKEN)
        ds = load_dataset("allenai/wildjailbreak", "eval")["train"]
        return ds

    def parseInput(self, item: dict) -> dict:
        """
        It retrieves the prompt and its type (e.g., 'safe' or 'unsafe') from a dataset item.

        Args:
            item (dict): A dictionary representing a dataset example,
                         e.g., {"Goal": "...", "Type": "safe"}  
        Returns:
            dict: A dictionary with keys 'prompt' and 'type'
                  e.g., {'prompt': 'What is 2+2?', 'type': 'safe'}
        """
        prompt = item.get("adversarial", "").strip()

        if item.get("data_type", "").strip() == "adversarial_harmful":
            typePrompt = "unsafe"
        else:
            typePrompt = "safe"


        return {
            "prompt": prompt,
            "type": typePrompt
        }