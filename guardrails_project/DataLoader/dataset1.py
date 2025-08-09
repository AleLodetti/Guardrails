from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset,concatenate_datasets
import pandas as pd

class JailbreakBench(SuperDataset):
    def __init__(self):
        super.__init__()
    
    def loadData(self):
        set1 = load_dataset("JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful")
        set2 = load_dataset("JailbreakBench/JBB-Behaviors", name="behaviors", split="benign")

        combined = concatenate_datasets([set1, set2])
        return combined

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
        prompt = item.get("Goal", "").strip()
        #type_of_prompt = item.get("Type", "").strip().lower()  # safe o unsafe

        return {
            "prompt": prompt,
            "type": "unsafe"
        }