TOKEN = "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"
GUARDRAIL_TOKEN = "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"
PATH_TO_RESPONSES = "responses"
CLASSIFIER_THRESHOLD = 0.7
PATH_TO_RESULTS = "results"
CURRENT_GUARDRAIL = "LlamaGuard"
CURRENT_LLM = "mistral"

def insert_token():
    import getpass
    global TOKEN
    TOKEN = getpass.getpass("Please enter your OpenAI API token: ").strip()

def insert_guardrail_token():
    import getpass
    global GUARDRAIL_TOKEN
    GUARDRAIL_TOKEN = getpass.getpass("Please enter your Guardrail API token: ").strip()