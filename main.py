#a function to extract answers from llama chat
from guardrails_project.LLMs.llmsFactory import LLMsFactory


def extract_response(text):
    if "[/INST]" in text:
        return text.split("[/INST]")[-1].strip()
    return text.strip()


if __name__= "__main__":

    """
    In questo punto del codice, l'utente può scegliere il modello da utilizzare.
    Il modello viene creato utilizzando il factory pattern.

    Raises:
        ValueError: nel caso in cui il modello non sia supportato.
    """

    try:
        selected_model_name = input("quale modello vuoi usare? (Mistral, Llama, Llama chat): ").strip()
        selected_model_name = selected_model_name.lower()

        if selected_model_name in ["mistral", "llama", "llama chat"]:
            model, tokenizer = LLMsFactory.create_llm(selected_model_name)
        else:
            raise ValueError("Model not supported. Please choose from: Mistral, Llama, Llama chat.")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    """
    Carica il dataset per la valutazione dei prompt. L'idea è di passare i prompt ai modelli LLM e valutarne il tipo di risposta data
    in modo da capire se il modello ha rifiutato il prompt o meno.
    """
    loaddataset()


    """
    Esegue il ciclo di valutazione dei prompt. Per ogni prompt, il modello LLM genera una risposta.
    Se la risposta è una sorta di rifiuto, viene incrementato il contatore dei rifiuti.
    Alla fine, viene stampato il numero totale di rifiuti rispetto al numero totale di prompt valutati.
    """
    #the number of refused prompt: the idea is to count how many responses have activated
    #a sort of Guardrail in the LLM.
    refused = 0
    tot_prompts = 5  # or len(dataset)

    for i, item in enumerate(dataset):
        if i >= tot_prompts:
            break
        prompt = item["Goal"]
        print(f"\nPrompt {i+1}: {prompt}")

        response = generate_response(selected_model_name, model, tokenizer, prompt, max_tokens=150)

        print("Answer:", extract_response(response))

        if is_refusal(response):
            refused += 1

    print(f"\nTotal refused prompts: {refused} out of {tot_prompts}")
