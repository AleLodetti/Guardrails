import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from guardrails_project.LLMs.base_guard import BaseGuard
from guardrails_project.constants import GUARDRAIL_TOKEN


class LlamaGuard(BaseGuard):
    def __init__(self, model = None, tokenizer = None):
        super().__init__
        if model is None and tokenizer is None:
            self.model = self.setModel()
            self.tokenizer = self.setTokenizer()
            self.classLabel = ["safe", "unsafe"]
        elif model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.classLabel = ["safe", "unsafe"]

    def get_model_info(self) -> dict:
        """Returns information about the Llama model."""
        return {
            "name": "LlamaGuard",
            "version": "1.0",
            "description": "Llama model with guardrails for generating safe text responses."
        }
    
    def validate_response(self, chat: dict) -> dict:
        """Classifies the prompt as safe or unsafe."""
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decodifica dell'output del classificatore
        raw_output = self.tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        status, reason = raw_output.split(",", 1) if "," in raw_output else (raw_output.strip(), None)

        return {
            "status": status.strip().lower(),          # es: "safe" o "unsafe"
            "reason": reason.strip() if reason else None,
            "raw_output": raw_output.strip(),
            "chat": chat
        }
    
    def setModel(self):
        MODEL = "meta-llama/LlamaGuard-7b"
        token = GUARDRAIL_TOKEN

        model_configs = {
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
            'token' : token,
            'quantization_config' : BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16'
            )
        }

        model = AutoModelForCausalLM.from_pretrained(MODEL, **model_configs)
        return model
    
    def setTokenizer(self):
        MODEL = "meta-llama/LlamaGuard-7b"
        token = GUARDRAIL_TOKEN

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