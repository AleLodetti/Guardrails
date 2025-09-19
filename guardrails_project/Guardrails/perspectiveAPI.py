from huggingface_hub import login
from guardrails_project.constants import TOKEN
login(TOKEN)
import requests
from datasets import load_dataset, concatenate_datasets
import time
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def runEvaluation(dataset, num_samples, text_column, label_column):
    """
    Runs the evaluation of the dataset using the Perspective API.
    
    Args:
        dataset: The dataset to evaluate.
        num_samples: The number of samples to evaluate.
        text_column: The column containing the text prompts.
        label_column: The column containing the ground truth label.
    """
    analyse_dataset_perspective(dataset, num_samples, text_column, label_column)
   

def check_prompt_perspective(text):
  attributes = ["TOXICITY", "SEVERE_TOXICITY", "THREAT", "INSULT", "OBSCENE", "IDENTITY_ATTACK", "SEXUALLY_EXPLICIT"]
  API_KEY = 'your_api_key_here'  # Replace with your actual API key
  url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={API_KEY}"
  data = {
      'comment': {'text': text},
      'languages': ['en'],
      'requestedAttributes': {attr: {} for attr in attributes}
  }
  response = requests.post(url, data=json.dumps(data))
  if response.status_code == 200:
      result = response.json()
      flag = False
      for attr in attributes:
          score = result["attributeScores"][attr]["summaryScore"]["value"]
          print(f"{attr:<20}: {score:.4f}\n")
          if score >= 0.5:
            flag = True
      return flag
  else:
      print("API request failed:", response.text)

def analyse_dataset_perspective(dataset, num_samples, text_column, label_column):
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

      detected = check_prompt_perspective(text)
      time.sleep(1.1)
      predicted_labels.append(1 if detected else 0)
      true_labels.append(true_label)
      if detected and true_label == 1:
          detected_cases += 1

  accuracy = accuracy_score(true_labels, predicted_labels)
  precision = precision_score(true_labels, predicted_labels)
  recall = recall_score(true_labels, predicted_labels)
  f1 = f1_score(true_labels, predicted_labels)
  print(f"\nNumber of jailbreak cases detected: {detected_cases}")
  print(f"Accuracy: {accuracy:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")
  print(f"F1 Score: {f1:.3f}")