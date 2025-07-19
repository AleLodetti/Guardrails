from guardrails_project.LLMs.base_llm import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig    
from transformers import SentenceTransformer, util
import torch

class Llama_chat(BaseLLM):
    """Implementation of the Llama chat model."""

    def __init__(self):
        super().__init__()
        self.model = self.setModel()
        self.tokenizer = self.setTokenizer()

    def __init__(self, model=None, tokenizer=None):
        """Initialize the Mistral model with custom model and tokenizer."""
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def get_model_info(self) -> dict:
        """
        Returns information about the Llama chat model.
        Returns:
            dict: A dictionary containing the model name, version, and description.
        """

        return {
            "name": "Llama Chat",
            "version": "3.0",
            "description": "Llama chat model for generating text responses."
        }


    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Generates a response from the Llama chat model based on the input messages.
        Args:
            messages (str): The input messages to the model.
            max_tokens (int): The maximum number of tokens to generate in the response.
        Returns:
            str: The generated response from the model. 
        """
        
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

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
        """
        Returns the tokenizer associated with the Llama chat model.
        """
        return self.tokenizer

    
    def getModel(self):
        """
        Returns the model instance associated with the Llama chat model.
        """
        return self.model
    
        
    def setModel(self): 
        """
        This function sets the model for Llama chat.
        It initializes the model with specific configurations such as dtype, device map, and quantization settings.
        The model is loaded from the Hugging Face model hub using the specified model name and token.
        The model is set to use 4-bit quantization for efficient memory usage and performance.
        The function returns the initialized model instance.
        """

        MODEL = "meta-llama/Llama-3.1-8B"
        token= "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"
        
        model_config = {
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
            'use_auth_token': token,
            'quantization_config': BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        }

        model = AutoModelForCausalLM.from_pretrained(MODEL, **model_config)
        return model
    

    def setTokenizer(self):
        """
        This function sets the tokenizer for Llama chat.
        It initializes the tokenizer using the specified model name and token.
        The tokenizer is configured to use the end-of-sequence token as the padding token to suppress warnings.
        The function returns the initialized tokenizer instance.
        """

        MODEL = "meta-llama/Llama-3.1-8B"
        token= "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"

        tokeniser_configs = {'token': token}
        tokenizer = AutoTokenizer.from_pretrained(MODEL, **tokeniser_configs)

        tokenizer.pad_token = tokenizer.eos_token  # This is just to suppress a warning

        return tokenizer