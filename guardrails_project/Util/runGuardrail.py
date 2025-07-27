import os
from guardrails_project.Guardrails.llamaGuard import LlamaGuard
from guardrails_project.LLMs.llmsFactory import LLMsFactory
from guardrails_project.DataLoader.dataset_loader import DatasetLoader
from itertools import islice
from guardrails_project.Util import modelManager
from guardrails_project.Util.answerReader import PromptReader
from guardrails_project.Util.answerSaver import PromptSaver
from guardrails_project.Util.checkResponse import CheckResponse
from guardrails_project.Util.parseDict import ParseDict
from guardrails_project.constants import PATH_TO_RESPONSES, PATH_TO_RESULTS
from colorama import Fore, Style, init

def runGuardrail():
    """
    Main function to run the guardrail evaluation. In this case it runs only Llama guard
    but it can be extended to run other models in the same spirit of the LLMsFactory.
    """
    selected_guardrail_name = input("which guardrail do you want to use between Llamaguard and ...?").strip().lower()
    instantiate = input("do you want to instantiate the model? (Y/n): ").strip().lower()

    if instantiate == 'y':
        #definisco il modello
        try:
            guardrail = LlamaGuard()
            modelManager.saveModel(guardrail)
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
    else:
        try:
            if selected_guardrail_name in ["llamaguard"]:
                print(f"Loading guardrail {selected_guardrail_name} from disk...")
                guardrail = modelManager.loadModel(selected_guardrail_name)
            else:
                raise ValueError("Guardrail not supported. Please choose from: Llamaguard")
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)

    print(f"Using guardrail: {guardrail.get_model_info()['name']}")

    #leggo le risposte precedenti una ad una e le valuto

    print("do you wanna test the responses of llama, llama chat (not suggested) or mistral? (llama/llama chat/mistral)")
    selected_model_name = input().strip().lower()

    if selected_model_name == "llama":
        file_name = "responsesLlama.jsonl"
    elif selected_model_name == "llama chat":
        file_name = "responsesLlamaChat.jsonl"
    elif selected_model_name == "mistral":
        file_name = "responsesMistral.jsonl"
    else:
        print("Invalid model name.")
        return

    promptReader = PromptReader(file_path=PATH_TO_RESPONSES, model_name=file_name)
    promptReader.__enter__()

    resultSaver = PromptSaver(file_path = PATH_TO_RESULTS, model_name = selected_guardrail_name)
    resultSaver.__enter__()
    
    while True:
        data = promptReader.getNextDict()
        if data is not None:
        #la valuto con LlamaGuard
            chat = ParseDict().parser(selected_guardrail_name, selected_model_name, data)
            final_response = guardrail.validate_response(chat)

            #if isUnsafe == true it means that the LLM modify the 
            # answer to make it safe ergo: the LLM detected correctly the
            # unsafeness of the prompt; else it considered it safe.
            if data["isUnsafe"] is True:
                originalDetection = "unsafe"
            else:
                originalDetection = "safe"

            result = {
                "originalDetection": originalDetection,
                "guardrailDetection":final_response["status"],
                "groundTruth": data["type"] 
            }
            #PromptSaver.saveResponseOnJsonl(PATH_TO_RESULTS, result, selected_guardrail_name)
            resultSaver.saveResponseOnJsonl(result)
            #printWithColor(final_response)
        else:
            print("No more responses to evaluate.")
            break
    
    resultSaver.__exit__(None, None, None)
    promptReader.__exit__(None, None, None)

def printWithColor(final_response: dict):
    """
    Stampa la risposta del guardrail a colori.

    Args:
        final_response (dict): Output di validate_response(), es:
            {
              "status": "unsafe",
              "category": o1, o2...
              "reason": "S1: Hate Speech"
            }
    """
    init(autoreset=True)

    
    CATEGORY_MAP = {
        "O1": "Safety and Emergency",
        "O2": "Off-Topic",
        "O3": "Complex Conditions",
        "O4": "Hate Speech",
        "O5": "Violence or Threats",
        # aggiungi qui tutte le categorie che ti servono
    }

    status = final_response.get("status", "").lower()
    category = final_response.get("category", "").upper()
    reason = final_response.get("reason", "")

    if status == "safe":
        color = Fore.GREEN
        symbol = "✅"
    elif status == "unsafe":
        color = Fore.RED
        symbol = "❗"
    else:
        color = Fore.YELLOW
        symbol = "⚠️"
    
    categoryDescription = CATEGORY_MAP.get(category, "unkown Category")
    
    print("-------------------------------------------------------------")
    print(f"{color}{symbol} LlamaGuard: {status.upper()}{Style.RESET_ALL}")
    if category:
        print(f"{color}   → Category: {category} - {categoryDescription}{Style.RESET_ALL}")
    if reason:
        print(f"{color}   → Reason: {reason}{Style.RESET_ALL}")

    #print(f"{Style.DIM}Model response:{Style.RESET_ALL}\n{resp}\n")
    #print(f"{Style.DIM}Raw guardrail output: {raw}{Style.RESET_ALL}")
