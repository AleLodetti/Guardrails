from typing import List
from guardrails_project.LLMs.base_llm import BaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

from guardrails_project.constants import TOKEN

class Mistral(BaseLLM):
    """Mistral model for text generation.

    Args:
        BaseLLM (): Base class for language models.
    """
    def __init__(self, model=None, tokenizer=None):
        """Initialize the Mistral model."""
        super().__init__()
        if model is None and tokenizer is None:
            self.model = self.setModel()
            self.tokenizer = self.setTokenizer()
        elif model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer

    def get_model_info(self) -> dict:
        """
        Return information about the Mistral model.
        Returns:
            dict: A dictionary containing the model name, version, and description.
        """
        return {
            "name": "Mistral",
            "version": "1.0",
            "description": "Mistral model for generating text responses."
        }

    def generate_response(self, prompts: List[str], max_tokens: int = 30) -> List[str]:
        """
        Generate text based on the input prompt.
        Args:
            prompts (List[str]): The input texts to generate responses for.
            max_tokens (int): The maximum number of tokens to generate.
        Returns:
            List[str]: The generated text responses.
        """
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(next(self.model.parameters()).device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id
            )
        texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return texts
    
    
    def getTokenizer(self):
        """
        Return the tokenizer of the model.
        """
        return self.tokenizer
    
    def getModel(self):
        """
        Return the model instance.
        """
        return self.model   
    
    def setModel(self):
        """
        Set the model instance for Mistral.
        Returns:
            model: The Mistral model instance.
        """
        MODEL = 'mistralai/Mistral-7B-v0.3'
        token = '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'

        model_configs = {
            'torch_dtype': 'bfloat16',
            'device_map': 'cuda',
            'token': token,
            'quantization_config': BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16'
            )
        }
        model = AutoModelForCausalLM.from_pretrained(MODEL, **model_configs)
        return model
    

    def setTokenizer(self):
        """
        Set the tokenizer instance.
        Returns:
            tokenizer: The Mistral tokenizer instance.
        """
        MODEL = 'mistralai/Mistral-7B-v0.3'
        token = TOKEN
        tokeniser_configs = {'token': token}
        tokenizer = AutoTokenizer.from_pretrained(MODEL, **tokeniser_configs)
        tokenizer.pad_token_id = tokenizer.eos_token_id  # This is just to suppress a warning
        
        return tokenizer
