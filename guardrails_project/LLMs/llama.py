from guardrails_project.LLMs.base_llm import BaseLLM

class llama(BaseLLM):
    """Implementation of the Llama model."""

    def __init__(self):
        super().__init__()
        # Initialize the Llama model and tokenizer here
        # self.model = ...
        # self.tokenizer = ...
        self.model = setModel()
        self.tokenizer = setTokenizer()


    def get_model_info(self) -> dict:
        """Returns information about the Llama model."""
        return {
            "name": "Llama",
            "version": "3.0",
            "description": "Llama model for generating text responses."
        }

    def generate_response(self, messages: list) -> str:
        """Generates a response based on a list of messages."""
        # Placeholder implementation
        return "Generated response from Llama."

    def getTokenizer(self):
        """Returns the tokenizer associated with the Llama model."""
        # Placeholder implementation
        return "Llama tokenizer"

    def getModel(self):
        """Returns the model instance."""
        # Placeholder implementation
        return "Llama model instance" 
    
    def setModel():
        """Set the model for Llama."""
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
                bnb_4bit_compute_dtype='bfloat16'
            )
        }

        model = AutoModelForCausalLM.from_pretrained(MODEL, **model_configs)

        return model

    def setTokenizer(): 
        MODEL = "meta-llama/Llama-3.1-8B"
        token= "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"
        tokeniser_configs = {'token': token}
        tokenizer = AutoTokenizer.from_pretrained(MODEL, **tokeniser_configs)
        tokenizer.pad_token = tokenizer_LlamaV2.eos_token  # This is just to suppress a warning
        tokenizer = padding_side = "left"  # This is just to suppress a warning
        return tokenizer