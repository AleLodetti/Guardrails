import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from guardrails_project.constants import TOKEN


class LlamaGuard:
    def __init__(self):
        self.model = self.setModel()
        self.tokenizer = self.setTokenizer()
        self.classLabel = ["safe", "unsafe"]

    def getModelInfo(self) -> dict:
        """Returns information about the Llama model."""
        return {
            "name": "LlamaGuard",
            "version": "1.0",
            "description": "Llama model with guardrails for generating safe text responses."
        }
    
    def classifyPrompt(self, prompt: str) -> str:
        """Classifies the prompt as safe or unsafe."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictedClass = torch.argmax(logits, dim=1).item()

        return self.classLabel[predictedClass]
    
    def setModel(self):
        MODEL = "meta-llama/LlamaGuard-7b"
        token = TOKEN

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            token=token,
            device_map="auto",
            torch_dtype=torch.bfloat16  # use float16 or bfloat16 depending on your hardware
        )
        return model
    
    def setTokenizer(self):
        MODEL = "meta-llama/LlamaGuard-7b"
        token = TOKEN

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            token=token,
            use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def getTokenizer(self):
        """Returns the tokenizer associated with the LlamaGuard model."""
        return self.tokenizer
    
    def getModel(self):
        """Returns the model instance."""
        return self.model