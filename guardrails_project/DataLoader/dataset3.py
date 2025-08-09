from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset
import pandas as pd

class Wildjailbreak(SuperDataset):
    """this dataset can be filtered?"""

    def __init__(self):
        super.__init__()
    
    def loadData(self):
        return load_dataset("allenai/wildjailbreak", "eval")["train"]

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
        prompt = item.get("user_input", "").strip()
        #type_of_prompt = item.get("Type", "").strip().lower()  # safe o unsafe

        if item.get("data_type", "").strip() == "adversarial_harmful":
            typePrompt = "unsafe"
        else:
            typePrompt = "safe"


        return {
            "prompt": prompt,
            "type": typePrompt
        }