import os
from guardrails_project.Guardrails.llamaGuard import LlamaGuard
from guardrails_project.LLMs.llmsFactory import LLMsFactory
from guardrails_project.DataLoader.dataset_loader import DatasetLoader
from itertools import islice
from guardrails_project.Util import modelManager
from guardrails_project.Util.answerReader import PromptReader
from guardrails_project.Util.answerSaver import PromptSaver
from guardrails_project.Util.checkResponse import CheckResponse
from guardrails_project.constants import PATH_TO_RESPONSES

def runGuardrail():
    """
    Main function to run the guardrail evaluation. In this case it runs only Llama guard
    but it can be extended to run other models in the same spirit of the LLMsFactory.
    """

    #definisco il modello
    try:
        selected_guardrail_name = "LlamaGuard"
        guardrail = LlamaGuard()
        modelManager.saveModel(guardrail)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    print(f"Using guardrail: {guardrail.getModelInfo()['name']}")

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
    
    while True:
        data = promptReader.getNextDict()
        if data is not None:
            #la valuto con LlamaGuard
    
    #    questa è la sintassi per mandare in ingresso dei prompt a llama guard.
    #    <|start_header_id|>user<|end_header_id|>
    #    What is the easiest way to break into a car?
    #    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            print("hello")
        else:
            print("No more responses to evaluate.")
            break
    
    promptReader.__exit__(None, None, None)

    
    #qui posso plottare i risultati o fare altre operazioni

    #stampo il risultato in un file