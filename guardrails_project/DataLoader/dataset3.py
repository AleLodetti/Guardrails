from constants import GUARDRAIL_TOKEN
from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd

class Wildjailbreak(SuperDataset):
    """this dataset can be filtered?"""

    def __init__(self):
        super().__init__()
    
    def loadData(self):
        login(token = GUARDRAIL_TOKEN)
        ds = load_dataset("allenai/wildjailbreak", "eval")["train"]
        return ds

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
        prompt = item.get("adversarial", "").strip()
        #type_of_prompt = item.get("Type", "").strip().lower()  # safe o unsafe

        if item.get("data_type", "").strip() == "adversarial_harmful":
            typePrompt = "unsafe"
        else:
            typePrompt = "safe"


        return {
            "prompt": prompt,
            "type": typePrompt
        }