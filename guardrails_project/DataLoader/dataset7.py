from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset, concatenate_datasets
import pandas as pd

class CSVfile(SuperDataset):
    """this dataset can be filtered?"""

    def __init__(self):
        super().__init__()
    
    def loadData(self):
        #ds1 = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_05_07', split='train')
        #ds2 = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_12_25', split='train')
        ds3 = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_12_25', split='train')

        #combined = concatenate_datasets([ds1, ds2, ds3])

        return ds3

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
        #type_of_prompt = item.get("Type", "").strip().lower()  # safe o unsafe

        if item.get("jailbreak", "") == True:
            typePrompt = "unsafe"
        else:
            typePrompt = "safe"

        return {
            "prompt": prompt,
            "type": typePrompt
        }