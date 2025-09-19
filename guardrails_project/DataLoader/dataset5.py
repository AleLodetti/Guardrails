from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset
import pandas as pd

class Wildguardmix(SuperDataset):

    def __init__(self):
        super().__init__()
    
    def loadData(self):
        selected_dataset = "allenai/wildguardmix"
        split = "wildguardtest"
        dataset = load_dataset(selected_dataset, split)["test"]
        return dataset


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
        prompt = item.get("prompt", "").strip()

        if item.get("prompt_harm_label", "") is not None:
            label = item.get("prompt_harm_label", "").strip()
        else:
            label = "unharmful"

        if label == "harmful" or item.get("adversarial", "") == True:
            typePrompt = "unsafe"
        else:
            typePrompt = "safe"


        return {
            "prompt": prompt,
            "type": typePrompt
        }