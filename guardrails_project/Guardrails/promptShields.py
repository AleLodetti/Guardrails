import requests
import time
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def runEvaluation(dataset, num_samples, text_column, label_column):
    """
    Runs the evaluation of the dataset using the Azure Content Safety API.
    
    Args:
        dataset: The dataset to evaluate.
        num_samples: The number of samples to evaluate.
        text_column: The column containing the text prompts.
        label_column: The column containing the ground truth label.
    """
    analyse_dataset_azure(dataset, num_samples, text_column, label_column)

def analyse_dataset_azure(dataset, num_samples, text_column, label_column):
  detected_cases = 0
  true_labels = []  # all 1s (attacks)
  predicted_labels = []  # predicted: 1 if attackDetected else 0
#note: you need to obtain an Azure subscription key and set it in the headers
  # for the API to work.
  url = 'https://guardrailazure.cognitiveservices.azure.com/contentsafety/text:shieldPrompt?api-version=2024-09-01'
  headers = {
      'Ocp-Apim-Subscription-Key': 'your_subscription_key_here',
      'Content-Type': 'application/json'
  }

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

    data = {"userPrompt": text, "documents": ["docs"]}
    response = requests.post(url, headers=headers, json=data)
    json_response = json.loads(response.text)

    try:
        detected = json_response['userPromptAnalysis']['attackDetected']
    except KeyError:
        print(f"Unexpected response structure on sample {i}: {json_response}")
        detected = False

    true_labels.append(true_label)
    if detected:
        predicted_labels.append(1)
        if true_label == 1:
            detected_cases += 1
    else:
        predicted_labels.append(0)

    time.sleep(0.1)

  accuracy = accuracy_score(true_labels, predicted_labels)
  precision = precision_score(true_labels, predicted_labels)
  recall = recall_score(true_labels, predicted_labels)
  f1 = f1_score(true_labels, predicted_labels)
  print(f"\nNumber of jailbreak cases detected: {detected_cases}")
  print(f"Accuracy: {accuracy:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")
  print(f"F1 Score: {f1:.3f}")