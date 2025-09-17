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
        print(self.model.hf_device_map)



    def get_model_info(self) -> dict:
        """Returns information about the Llama model."""
        return {
            "name": "LlamaGuard",
            "version": "1.0",
            "description": "Llama model with guardrails for generating safe text responses."
        }
    
    def validate_response(self, chat: dict) -> dict:
        """this method checks if the chat is safe from the point of view
        LlamaGuard, meaning that there is not attempt of jailbreaking 
        the LLM.

        Args:
            chat (dict): is a ditionare that contains the prompt and the 
            LLM response.

        Returns:
            dict: it return a dictionare of a format {
              "status": "unsafe",
              "reason": "S1: Hate Speech",
              "raw_output": "unsafe, S1: Hate Speech",
              "chat": "..."
            }
        """
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        # Decode the output of the classificator
        raw_output = self.tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        status, category, reason = self.parseRawOutput(raw_output) 

        return {
            "status": status.strip().lower(),          # es: "safe" o "unsafe"
            "category": category.strip().upper() if category else "unknown",
            "reason": reason.strip() if reason else None,
            #"chat": chat
        }

    def validate_responses(self, chats: list[dict]) -> list[dict]:
        """
        Validate multiple chats in batch using LlamaGuard.

        Args:
            chats (list[dict]): List of dictionaries with prompt and response.

        Returns:
            list[dict]: List of validation results.
        """
        # It applies the same chat template to all chats
        prompts = [
            self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            for chat in chats
        ]

        # Tokenizza in batch
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        results = []
        for i, output in enumerate(outputs):
            generated_tokens = output[inputs['input_ids'].shape[-1]:]
            if generated_tokens.numel() == 0:
                raw_output = ""
            else:
                raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            try:
                status, category, reason = self.parseRawOutput(raw_output)
            except Exception:
                status, category, reason = "unknown", "unknown", None

            results.append({
                "status": status.strip().lower(),
                "category": category.strip().upper() if category else "UNKNOWN",
                "reason": reason.strip() if reason else None,
                "raw_output": raw_output,
                "chat": chats[i]
            })

        return results

    def parseRawOutput(self, raw_output: str):
        raw_output = raw_output.strip()
        lines = raw_output.split("\n")

        status = lines[0].strip().lower()  # prima riga → "safe" o "unsafe"

        category = None
        reason = None

        if len(lines) > 1:
            line = lines[1]
            # es: "O1: Hate Speech" → categoria e motivo
            if ":" in line:
                category, reason = map(str.strip, line.split(":", 1))
            else:
                category = line.strip()

        return status, category, reason
    
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

        """tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            token=token,
            use_fast=True,
            trust_remote_code=True
        )"""

        token_config = {'token': token}

        tokenizer = AutoTokenizer.from_pretrained(MODEL, **token_config)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def getTokenizer(self):
        """Returns the tokenizer associated with the LlamaGuard model."""
        return self.tokenizer
    
    def getModel(self):
        """Returns the model instance."""
        return self.model