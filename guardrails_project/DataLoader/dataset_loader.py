#questa classe va rivista per caricare i dataset in maniera intelligente e funzionale
from datasets import load_dataset
import pandas as pd

from guardrails_project.DataLoader.dataset1 import JailbreakBenchBadDataset, JailbreakBenchGoodDataset
from guardrails_project.DataLoader.dataset2 import JailbreakV_28K
from guardrails_project.DataLoader.dataset3 import Wildjailbreak
from guardrails_project.DataLoader.dataset4 import ToxicChat
from guardrails_project.DataLoader.dataset5 import Wildguardmix
from guardrails_project.DataLoader.dataset6 import XSTest
from guardrails_project.DataLoader.dataset7 import CSVfile

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
        print("1 - JailbreakBench Harmful Behaviors (HuggingFace)")
        print("2 - JailBreakV_28K (this one only contains adversarial prompts)")
        print("3 - WIldjailbreak")
        print("4 - ToxicChat")
        print("5 - Wildguardmix")
        print("6 - XSTest")
        print("7 - CSV file")
        print("8 - JailbreakBench good Behaviors (HuggingFace)")

        choice = input("Enter your choice (1/2): ").strip()

        #here I instantiate the correct dataset as an object and then I load it by using its methods.
        if choice == "1":
            dataset = JailbreakBenchBadDataset()
            return dataset
        elif choice == "2":
            dataset = JailbreakV_28K() 
            return dataset
        elif choice == "3":
            dataset = Wildjailbreak() 
            return dataset
        elif choice == "4":
            dataset = ToxicChat()
            return dataset
        elif choice == "5":
            dataset = Wildguardmix() 
            return dataset
        elif choice == "6":
            dataset = XSTest()
            return dataset
        elif choice == "7":
            dataset = CSVfile()
            return dataset
        elif choice == "8":
            dataset = JailbreakBenchGoodDataset()
            return dataset
        else:
            print("Invalid choice")
            return None