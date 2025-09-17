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
        Estrae il prompt e il tipo (es. 'safe' o 'unsafe') da un item del dataset.

        Args:
            item (dict): Un dizionario che rappresenta un esempio del dataset,
                         ad esempio: {"Goal": "...", "Type": "safe"}

        Returns:
            dict: Un dizionario con le chiavi 'prompt' e 'type'
                  es: {'prompt': 'What is 2+2?', 'type': 'safe'}
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