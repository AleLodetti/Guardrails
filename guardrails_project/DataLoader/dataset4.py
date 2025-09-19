from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset
import pandas as pd

class ToxicChat(SuperDataset):

    def __init__(self):
        super().__init__()
    
    def loadData(self):
        selected_dataset = "lmsys/toxic-chat"
        split='toxicchat0124'
        dataset = load_dataset(selected_dataset, split)["train"]
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

        prompt = item.get("user_input", "").strip()

        if item.get("jailbreaking", "") == 1:
            typePrompt = "unsafe"
        else:
            typePrompt = "safe"

        return {
            "prompt": prompt,
            "type": typePrompt
        }