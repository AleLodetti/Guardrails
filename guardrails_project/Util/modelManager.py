
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from guardrails_project.LLMs.llmsFactory import LLMsFactory

#con llama guard dovrebbe cambiare, soprattutto per il caricamente dove si usa
#AutoModelForSequenceClassification...

def saveModel(llm):
    """
    Save the model
    
    Args:
        model: The model to save.
    """
    name = llm.get_model_info()['name']
    if name == "LlamaGuard":
        folder_name = "llamaguard"
    elif name == "Llama":
        folder_name = "llama"
    elif name == "Llama Chat":
        folder_name = "llamaChat"
    elif name == "Mistral":
        folder_name = "mistral"

    path = os.path.join("savedModels", folder_name)

    if not os.path.exists(path):
        print(f"Creating directory {path} to save the model...")
        os.makedirs(path)

    if os.path.exists(path):
        llm.getModel().save_pretrained(path)
        llm.getTokenizer().save_pretrained(path)

        print(f"Model {name} saved successfully in {path}.")
    else:
        print(f"Error: The directory {path} does not exist. Please create it first.")


def loadModel(model_name):
    """
    Load the model
    
    Args:
        model_name: The name of the model to load.
    
    Returns:
        The loaded model.
    """

    if model_name.lower() == "llamachat":
        model_name = "llamaChat"
    
    path = os.path.join("savedModels", model_name)
        
    try:
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        llm = LLMsFactory.create_llm_custom(model_name, model=model, tokenizer=tokenizer)
        print(f"Model {model_name} loaded successfully.")
        return llm
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    