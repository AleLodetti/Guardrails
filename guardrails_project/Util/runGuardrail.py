import os
from guardrails_project import constants
from guardrails_project.Guardrails.llamaGuard import LlamaGuard
from guardrails_project.LLMs.llmsFactory import LLMsFactory
from guardrails_project.DataLoader.dataset_loader import DatasetLoader
from itertools import islice
from guardrails_project.Util import modelManager
from guardrails_project.Util.answerReader import PromptReader
from guardrails_project.Util.answerSaver import PromptSaver
from guardrails_project.Util.parseDict import ParseDict
from guardrails_project.constants import *

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
    
    with open("guardrails_project/constants.py", "w") as f:
        f.write(f'TOKEN = "{constants.TOKEN}"\n')
        f.write(f'GUARDRAIL_TOKEN = "{constants.GUARDRAIL_TOKEN}"\n')
        f.write(f'PATH_TO_RESPONSES = "{constants.PATH_TO_RESPONSES}"\n')
        f.write(f'CLASSIFIER_THRESHOLD = {constants.CLASSIFIER_THRESHOLD}\n')
        f.write(f'PATH_TO_RESULTS = "{constants.PATH_TO_RESULTS}"\n')
        f.write(f"CURRENT_GUARDRAIL = \"{guardrail.get_model_info()['name']}\"\n")
        f.write(f'CURRENT_LLM = "{selected_model_name}"\n') 

    promptReader = PromptReader(file_path=PATH_TO_RESPONSES, model_name=file_name)
    promptReader.__enter__()

    resultSaver = PromptSaver(file_path = PATH_TO_RESULTS, model_name = selected_guardrail_name)
    resultSaver.__enter__()
    
    """
    i = 0
    while True:
        data = promptReader.getNextDict()
        if data is not None:
            i = i + 1
        #la valuto con LlamaGuard
            chat = ParseDict().parser(selected_guardrail_name, selected_model_name, data)
            final_response = guardrail.validate_response(chat)

            #if isUnsafe == true it means that the LLM modify the 
            # answer to make it safe ergo: the LLM detected the
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
            if i == 20:
                i = 0
                print("other 20 prompts have been processed")
        else:
            print("No more responses to evaluate.")
            break
    """
    batch_size = 4
    max_token = 4500
    batch_chats = []
    batch_data = []
    i = 0

    while True:
        data = promptReader.getNextDict()
        if data is None:
            if batch_chats:  # processa eventuali rimanenze
                results = guardrail.validate_responses(batch_chats)
                for res, d in zip(results, batch_data):
                    originalDetection = "unsafe" if d["isUnsafe"] else "safe"
                    result = {
                        "originalDetection": originalDetection,
                        "guardrailDetection": res["status"],
                        "groundTruth": d["type"]
                    }
                    resultSaver.saveResponseOnJsonl(result)
            print("No more responses to evaluate.")
            break

        # prepara la chat per LlamaGuard
        chat = ParseDict().parser(selected_guardrail_name, selected_model_name, data)
        
        #controllo i token
        template = guardrail.getTokenizer().apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        chat_token = len(guardrail.getTokenizer()(template)["input_ids"])
        prospect = max_token - chat_token
        print(prospect)
        
        #in this case i need to skip
        if prospect < -700:
            results = guardrail.validate_responses(batch_chats)
            for res, d in zip(results, batch_data):
                originalDetection = "unsafe" if d["isUnsafe"] else "safe"
                result = {
                    "originalDetection": originalDetection,
                    "guardrailDetection": res["status"],
                    "groundTruth": d["type"]
                }
                resultSaver.saveResponseOnJsonl(result)

            batch_chats = []
            batch_data = []
            max_token = 4500

            print(f"{i} prompts processed so far.")
        
        batch_chats.append(chat)
        batch_data.append(data)
        max_token = prospect
        i += 1

        # processa il batch se pieno
        if max_token <= 0 and prospect >= -700:
            results = guardrail.validate_responses(batch_chats)
            for res, d in zip(results, batch_data):
                originalDetection = "unsafe" if d["isUnsafe"] else "safe"
                result = {
                    "originalDetection": originalDetection,
                    "guardrailDetection": res["status"],
                    "groundTruth": d["type"]
                }
                resultSaver.saveResponseOnJsonl(result)

            batch_chats = []
            batch_data = []
            max_token = 4500
            print(f"{i} prompts processed so far.")

    resultSaver.__exit__(None, None, None)
    promptReader.__exit__(None, None, None)