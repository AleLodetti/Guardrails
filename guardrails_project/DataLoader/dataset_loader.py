#questa classe va rivista per caricare i dataset in maniera intelligente e funzionale
from datasets import load_dataset
import pandas as pd

class DatasetLoader:
    def __init__(self):
        super().__init__()

    def loadDataset(self):
        """
        Load a dataset from HuggingFace based on user choice.
        Returns:
            dataset: The loaded dataset.
        """
        print("Select a dataset to load:")
        print("1 - JailbreakBench Behaviors (HuggingFace)")
        print("2 - Another HuggingFace Dataset")

        choice = input("Enter your choice (1/2): ").strip()

        if choice == "1":
            dataset = load_dataset("JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful")
        elif choice == "2":
            dataset = load_dataset("JailbreakV-28K/JailBreakV-28k", name="mini_JailBreakV_28K", split="train")
        else:
            print("Invalid choice")
            return None

        return dataset