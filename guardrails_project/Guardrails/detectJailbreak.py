

import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from guardrails import Guard
guard = Guard().use("hub://guardrails/detect_jailbreak")


def runEvaluation(dataset, num_samples, text_column, label_column):
    """
    Runs the evaluation of the dataset using the Detect Jailbreak guardrail.
    
    Args:
        dataset: The dataset to evaluate.
        num_samples: The number of samples to evaluate.
        text_column: The column containing the text prompts.
        label_column: The column containing the ground truth label.
    """
    #analyse_dataset_guardrailsAI(dataset, num_samples, text_column, label_column)


'''def analyse_dataset_guardrailsAI(dataset, num_samples, text_column, label_column):
  detected_cases = 0
  true_labels = []
  predicted_labels = []

  for i in range(num_samples):
    if isinstance(dataset, pd.DataFrame):
        text = dataset.iloc[i][text_column]
        true_label = dataset.iloc[i][label_column]
    else:
        sample = dataset.select(range(num_samples))[i]
        text = sample[text_column]
        true_label = sample[label_column]

    if isinstance(true_label, bool):
        true_label = int(true_label)
    elif isinstance(true_label, str):
        true_label = 1 if true_label.lower() in ["attack", "unsafe", "adversarial_harmful", "harmful", "malicious", "jailbreak", "true"] else 0
    elif isinstance(true_label, (int, float)):
        true_label = int(true_label)
    else:
        raise ValueError(f"Unsupported label type: {type(true_label)}")

    #print("\nPrompt:", text, "\n")
    try:
        result = guard.validate(text)
        detected = not result["valid"]
    except Exception as e:
        detected = True
       # print("\nException:", e)

    predicted_labels.append(1 if detected else 0)
    true_labels.append(true_label)
    if detected and true_label == 1:
        detected_cases += 1
      #  print("\nJailbreak detected")

    time.sleep(1)

  accuracy = accuracy_score(true_labels, predicted_labels)
  precision = precision_score(true_labels, predicted_labels)
  recall = recall_score(true_labels, predicted_labels)
  f1 = f1_score(true_labels, predicted_labels)
  print(f"\nNumber of jailbreak cases detected: {detected_cases}")
  print(f"Accuracy: {accuracy:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")
  print(f"F1 Score: {f1:.3f}")'''