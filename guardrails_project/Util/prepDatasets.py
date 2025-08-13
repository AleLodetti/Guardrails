import os
from guardrails_project import constants
from guardrails_project.Guardrails.llamaGuard import LlamaGuard
from guardrails_project.Guardrails.perspectiveAPI import PerspectiveAPI
from guardrails_project.Guardrails.detectJailbreak import DetectJailbreak
from guardrails_project.Guardrails.promptShields import PromptShields

from guardrails_project.constants import *


def runGuardrail(choice):
    datasets = [
    "JailbreakV-28K/JailBreakV-28k",
    "allenai/wildjailbreak",
    "lmsys/toxic-chat",
    "JailbreakBench/JBB-Behaviors",
    "allenai/wildguardmix",
    "walledai/XSTest",
    "TrustAIRLab/in-the-wild-jailbreak-prompts"
    ]

    for selected_dataset in datasets:
        print(f"\nProcessing dataset: {selected_dataset}")
        if selected_dataset == "allenai/wildjailbreak":
            split = "eval"
            dataset = load_dataset(selected_dataset, split)["train"]
            dataset = dataset.shuffle(seed=42)
            num_samples = len(dataset)
            text_column = "adversarial"
            label_column = "data_type"

        elif selected_dataset == "lmsys/toxic-chat":
            split = 'toxicchat0124'
            dataset = load_dataset(selected_dataset, split)["train"]
            dataset = dataset.shuffle(seed=42)
            num_samples = len(dataset)
            text_column = "user_input"
            label_column = "jailbreaking"

        elif selected_dataset == "JailbreakBench/JBB-Behaviors":
            split = "behaviors"
            datasetHarmful = load_dataset(selected_dataset, split)["harmful"]
            datasetBenign = load_dataset(selected_dataset, split)["benign"]
            datasetHarmful = datasetHarmful.add_column("label", [1] * len(datasetHarmful))
            datasetBenign = datasetBenign.add_column("label", [0] * len(datasetBenign))
            dataset = concatenate_datasets([datasetHarmful, datasetBenign])
            dataset = dataset.shuffle(seed=42)
            num_samples = len(dataset)
            text_column = "Goal"
            label_column = "label"

        elif selected_dataset == "allenai/wildguardmix":
            split = "wildguardtest"
            dataset = load_dataset(selected_dataset, split)["test"]
            dataset = dataset.shuffle(seed=42)
            num_samples = len(dataset)
            text_column = "prompt"
            label_column = "adversarial"

        elif selected_dataset == "walledai/XSTest":
            split = "test"
            dataset = load_dataset(selected_dataset, split=split)
            dataset = dataset.shuffle(seed=42)
            num_samples = len(dataset)
            text_column = "prompt"
            label_column = "label"

        elif selected_dataset == "TrustAIRLab/in-the-wild-jailbreak-prompts":
            datasetHarmful = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts",
                "jailbreak_2023_05_07"
            )
            datasetBenign = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts",
                "regular_2023_05_07"
            )
            datasetBenign = datasetBenign["train"].select(range(len(datasetHarmful["train"])))
            datasetHarmful = datasetHarmful["train"]
            dataset = concatenate_datasets([datasetHarmful, datasetBenign])
            dataset = dataset.shuffle(seed=42)
            num_samples = len(dataset)
            text_column = "prompt"
            label_column = "jailbreak"
    if choice == 1:
        guardrail = PerspectiveAPI()
        print("Starting the analysis of datasets using PerspectiveAPI. This may take a while...")
        guardrail.runEvaluation(
            dataset=dataset,
            num_samples=num_samples,
            text_column=text_column,
            label_column=label_column
        )
        guardrail.saveResults()
    elif choice == 2:
        guardrail = DetectJailbreak()
        print("Starting the analysis of datasets using Detect Jailbreak. This may take a while...")
        guardrail.runEvaluation(
            dataset=dataset,
            num_samples=num_samples,
            text_column=text_column,
            label_column=label_column
        )
        guardrail.saveResults()
    elif choice == 3:
        guardrail = PromptShields()
        print("Starting the analysis of datasets using Prompt Shields. This may take a while...")
        guardrail.runEvaluation(
            dataset=dataset,
            num_samples=num_samples,
            text_column=text_column,
            label_column=label_column
        )
        guardrail.saveResults()
    return guardrail