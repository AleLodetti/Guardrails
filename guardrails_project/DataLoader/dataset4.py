from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset
import pandas as pd

class ToxicChat(SuperDataset):
    """this dataset can be filtered?"""

    def __init__(self):
        super.__init__()
    
    def loadData(self):
        selected_dataset = "lmsys/toxic-chat"
        split='toxicchat0124'
        dataset = load_dataset(selected_dataset, split)["train"]
        return dataset


    def parseInput(self, item: dict) -> dict:
        """
        Estrae il prompt e il tipo (es. 'safe' o 'unsafe') da un item del dataset.

        Args:
            item (dict): Un dizionario che rappresenta un esempio del dataset,
                         ad esempio: {"Goal": "...", "Type": "safe"}

        Returns:
            dict: Un dizionario con le chiavi 'prompt' e 'type'
                  es: {'prompt': 'What is 2+2?', 'type': 'safe'}
        """
        typePrompt = item.get("toxicity", "").strip()
        while True:
            if typePrompt == "0":
                prompt = item.get("user_input", "").strip()
                break
            typePrompt = item.get("toxicity", "").strip()

        return {
            "prompt": prompt,
            "type": "unsafe"
        }