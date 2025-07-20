from guardrails_project.LLMs.base_llm import BaseLLM
from guardrails_project.constants import TOKEN
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

class Llama(BaseLLM):
    """Implementation of the Llama model."""
    def __init__(self, model = None, tokenizer = None ):
        super().__init__()
        print("Initializing Llama model...")
        if model is None and tokenizer is None:
            self.model = self.setModel()
            self.tokenizer = self.setTokenizer()
        elif model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer

    def get_model_info(self) -> dict:
        """Returns information about the Llama model."""
        return {
            "name": "Llama",
            "version": "3.0",
            "description": "Llama model for generating text responses."
        }

    def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        """Generates a response from the Llama model based on the input messages."""

        #message = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    def getTokenizer(self):
        """Returns the tokenizer associated with the Llama model."""
        return self.tokenizer

    def getModel(self):
        """Returns the model instance."""
        return self.model
    
    def setModel(self):
        """Set the model for Llama."""
        print("Loading Llama model...")
        try:
            MODEL = "meta-llama/Llama-3.1-8B"
            token= "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"

            model_configs = {
                'torch_dtype': 'bfloat16',
                'device_map': 'cuda',
                'token': token,
                'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype= 'bfloat16'
                )
            }

            model = AutoModelForCausalLM.from_pretrained(MODEL, **model_configs)
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
        return model

    def setTokenizer(self): 
        """
        Set the tokenizer for Llama.
        Initializes the tokenizer with specific configurations such as the model name and token.
        The tokenizer is loaded from the Hugging Face model hub using the specified model name and token.
        The tokenizer is set to use the end-of-sequence token as the padding token to suppress warnings.

        Returns:
            tokenizer: The initialized tokenizer instance.
        """
        MODEL = "meta-llama/Llama-3.1-8B"
        token= TOKEN
        tokeniser_configs = {'token': token}
        tokenizer = AutoTokenizer.from_pretrained(MODEL, **tokeniser_configs)
        tokenizer.pad_token = tokenizer.eos_token  # This is just to suppress a warning
        
        return tokenizer