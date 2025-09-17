from guardrails_project.LLMs.llmsFactory import LLMsFactory
from guardrails_project.DataLoader.dataset_loader import DatasetLoader
from itertools import islice
from guardrails_project.Util import modelManager
from guardrails_project.Util.answerSaver import PromptSaver
from guardrails_project.Util.checkResponse import is_refusal
from guardrails_project.constants import PATH_TO_RESPONSES
from guardrails_project.LLMs.llama import Llama
from guardrails_project.DataLoader.superDataset import SuperDataset


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
    ds = DatasetLoader().loadDataset() #carica l'oggetto

    dataset = ds.loadData() #carica i dati


    """
    Esegue il ciclo di valutazione dei prompt. Per ogni prompt, il modello LLM genera una risposta.
    Se la risposta è una sorta di rifiuto, viene incrementato il contatore dei rifiuti.
    Alla fine, viene stampato il numero totale di rifiuti rispetto al numero totale di prompt valutati.
    """
    refused = 0
    tot_prompts = len(dataset)

    print("strarting processsing")

    responseSaver = PromptSaver(file_path=PATH_TO_RESPONSES, model_name=selected_model_name)
    responseSaver.__enter__()

    effective_number_prompt = 0
    residual_token = 1800
    promptBatch = []
    typeBatch = []
    for i, item in enumerate(islice(dataset, tot_prompts)):

        if i >= 2000:
            break

        #here I need to parse the dataset inputs: some of them are 
        #all adversarial other are mixed. So by parsing the inputs I 
        #can standardize them to a format like: {string - type} 

        #item is the row of the dataset, with parseInput() I shall select what I need.
        #promptAndType = DatasetLoader().parseInput(item)
        """
        promptAndType = ds.parseInput(item)
        prompt = promptAndType['prompt']
        typeOfPrompt = promptAndType['type'] #Safe Or Unsafe now.
        
        response = llm.generate_response(prompt, max_tokens=50)

        #print("Answer:", response)

        isUnsafe= CheckResponse.is_refusal(response)

        data = {
            "prompt": prompt,
            "model_response": response,
            #it is true if the method is_refusal detected that the llm refused to generate a response, it is false if the method detected that the llm hasn't refused to generate the response
            "isUnsafe": isUnsafe, 
            "type": typeOfPrompt

        }
        """

        #mettere un controllo piu efficiente sui prompt passati: invece che passarne solo alcuni li passi tutti 
        #ma fai in modo da non mandare in overflow o quello che è la GPU. Tipo valuti in quanti token viene frammentata una frase

        promptAndType = ds.parseInput(item)

        #uso questo controllo per splittare i dataset
        if i >= 0: 
            prompt = promptAndType['prompt']
            typePrompt = promptAndType['type']

            prompt_tok_est = len(prompt)/4
            
            if len(prompt) >= 7300:
                print("this prompt is too long")
                continue

            prediction = residual_token - prompt_tok_est

            #in this case I should analyze a prompt too large but I don't have the computational resources
            if prediction < -100:
                print(f"ready to generate at {i} with residual_token: {residual_token}")
                #print(f"prompt: {prompt}")
                responses = llm.generate_response(promptBatch)

                for k, r in enumerate(responses):
                    r = responses[k]
                    #isUnsafe= CheckResponse.is_refusal(r)
                    isUnsafe = is_refusal(r)
                    data = {
                            "prompt": promptBatch[k],
                            "model_response": r,
                            #it is true if the method is_refusal detected that the llm refused to generate a response, it is false if the method detected that the llm hasn't refused to generate the response
                            "isUnsafe": isUnsafe, 
                            "type": typeBatch[k]

                    }
                    responseSaver.saveOnJsonFile(file_path=PATH_TO_RESPONSES, data=data, model_name=selected_model_name)
                
                #responseSaver.saveOnJsonFile(data = data)
                print(f"{i} prompts processed so far.")
                promptBatch = []
                typeBatch = [] 
                residual_token = 1900
                typeBatch.append(typePrompt)
                promptBatch.append(prompt)   
            
            else:
                residual_token = prediction
                typeBatch.append(typePrompt)
                promptBatch.append(prompt)

                if (residual_token <=0 or i == tot_prompts - 1):
                    print(f"ready to generate at {i} with residual_token: {residual_token}")
                    #print(f"prompt: {prompt}")
                    responses = llm.generate_response(promptBatch)

                    for k, r in enumerate(responses):
                        r = responses[k]
                        #isUnsafe= CheckResponse.is_refusal(r)
                        isUnsafe = is_refusal(r)
                        data = {
                                "prompt": promptBatch[k],
                                "model_response": r,
                                #it is true if the method is_refusal detected that the llm refused to generate a response, it is false if the method detected that the llm hasn't refused to generate the response
                                "isUnsafe": isUnsafe, 
                                "type": typeBatch[k]

                        }
                        responseSaver.saveOnJsonFile(file_path=PATH_TO_RESPONSES, data=data, model_name=selected_model_name)
                    
                    #responseSaver.saveOnJsonFile(data = data)
                    print(f"{i} prompts processed so far.")
                    promptBatch = []
                    typeBatch = [] 
                    residual_token = 1900       
        

    responseSaver.__exit__(None, None, None)

    print(f"\nTotal refused prompts: {refused} out of {tot_prompts}")
