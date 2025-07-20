from guardrails_project.LLMs.llmsFactory import LLMsFactory
from guardrails_project.DataLoader.dataset_loader import DatasetLoader
from itertools import islice
from guardrails_project.Util import modelManager
from guardrails_project.Util.answerSaver import PromptSaver
from guardrails_project.Util.checkResponse import CheckResponse
from guardrails_project.constants import PATH_TO_RESPONSES
from guardrails_project.LLMs.llama import Llama


def instantiateModelAndFillFile():
    """
    Main function to instantiate the LLM and fill the response file.
    This function, since it instantiates the model, can take a while to run.
    """

    selected_model_name = input("quale modello vuoi usare? (Mistral, Llama, Llama chat): ").strip()
    selected_model_name = selected_model_name.lower().replace(" ", "")

    instantiate = input("do you want to instantiate the model? (Y/n): ").strip().lower()

    if instantiate == 'y':
        try:
            if selected_model_name in ["mistral", "llama", "llamachat"]:
                llm = LLMsFactory.create_llm(selected_model_name)
                print('MODEL:', llm.getModel())
                modelManager.saveModel(llm)
            else:
                raise ValueError("Model not supported. Please choose from: Mistral, Llama, Llama chat.")
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
    else:
        try:
            if selected_model_name in ["mistral", "llama", "llamachat"]:
                print(f"Loading model {selected_model_name} from disk...")
                llm = modelManager.loadModel(selected_model_name)
            else:
                raise ValueError("Model not supported. Please choose from: Mistral, Llama, Llama chat.")
        except ValueError as e:
            print(f"Error: {e}")
            exit(1) 

    """
    Carica il dataset per la valutazione dei prompt. L'idea è di passare i prompt ai modelli LLM e valutarne il tipo di risposta data
    in modo da capire se il modello ha rifiutato il prompt o meno.
    """
    dataset = DatasetLoader().loadDataset()


    """
    Esegue il ciclo di valutazione dei prompt. Per ogni prompt, il modello LLM genera una risposta.
    Se la risposta è una sorta di rifiuto, viene incrementato il contatore dei rifiuti.
    Alla fine, viene stampato il numero totale di rifiuti rispetto al numero totale di prompt valutati.
    """
    refused = 0
    tot_prompts = 5  # or len(dataset)

    for i, item in enumerate(islice(dataset, tot_prompts)):

        prompt = item["Goal"]
        print(f"\nProcessing prompt number: {i+1}: {prompt}")

        response = llm.generate_response(prompt, max_tokens=150)

        print("Answer:", response)


        isUnsafe= CheckResponse.is_refusal(response)

        data = {
            "prompt": prompt,
            "model_response": response,
            "isUnsafe": isUnsafe
        }

        PromptSaver.saveOnJsonFile(file_path=PATH_TO_RESPONSES, data=data, model_name=selected_model_name)

        if isUnsafe:
            refused += 1

    print(f"\nTotal refused prompts: {refused} out of {tot_prompts}")
