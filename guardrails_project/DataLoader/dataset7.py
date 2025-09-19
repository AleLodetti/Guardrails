from guardrails_project.DataLoader.superDataset import SuperDataset
from datasets import load_dataset, concatenate_datasets

class CSVfile(SuperDataset):

    def __init__(self):
        super().__init__()
    
    def loadData(self):
        ds = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_12_25', split='train')

        return ds

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

        if item.get("jailbreak", "") == True:
            typePrompt = "unsafe"
        else:
            typePrompt = "safe"

        return {
            "prompt": prompt,
            "type": typePrompt
        }