from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset
import pandas as pd

class JailbreakV_28K(SuperDataset):
    def __init__(self):
        super.__init__()
    
    def loadData(self):
        return load_dataset("JailbreakV-28K/JailBreakV-28k", split="JailBreakV_28K")["mini_JailBreakV_28K"]  # Adjust this key for other datasets

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
        prompt = item.get("redteam_query", "").strip()
        #type_of_prompt = item.get("Type", "").strip().lower()  # safe o unsafe

        return {
            "prompt": prompt,
            "type": "unsafe"
        }