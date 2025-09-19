from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset
import pandas as pd

class JailbreakV_28K(SuperDataset):
    def __init__(self):
        super().__init__()
    
    def loadData(self):
        return load_dataset("JailbreakV-28K/JailBreakV-28k", split="JailBreakV_28K")["mini_JailBreakV_28K"]  # Adjust this key for other datasets

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
        prompt = item.get("redteam_query", "").strip()

        return {
            "prompt": prompt,
            "type": "unsafe"
        }