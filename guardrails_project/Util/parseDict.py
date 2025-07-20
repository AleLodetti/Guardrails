import re


class ParseDict:
    def __init__(self):
        pass

    def parser(self, guardrail_name, model_name, chat):
        guardrail_name = guardrail_name.lower().replace(" ", "")
        if guardrail_name == "llamaguard":
            parsed = self.parseForLlamaguard(model_name, chat)
        else:
            print("work in progress...")        
        return parsed
    
    #                   ----------------------------------------------------
    #                   -----here I parse the responses for llamaguard------
    #                   ----------------------------------------------------
    
    def parseForLlamaguard(self, model_name, chat):
        model_name = model_name.lower().replace(" ", "")

        parsed = ""

        if model_name == "llama":
            parsed = self.parseLlamaForLlamaguard(chat)
        elif model_name == "mistral":
            parsed = self.parseMistralForLlamaguard(chat)
        elif model_name == "llamachat":
            parsed = self.parseLlamaChatForLlamaguard(chat)
        else:
            print("model not supported... what is ", model_name)
        
        return parsed
    
    def parseLlamaForLlamaguard(self, chat):
        clean_chat = [
            {"role": "user", "content": chat['prompt']},
            {"role": "assistant", "content": chat['model_response']}
        ]
        return clean_chat
    
    def parseLlamaChatForLlamaguard(self, chat):
        #here i use REGEX to extract the model answer
        clean_answer, answer = chat["model_response"]
        
        matches = list(re.finditer(r"\[/INST\]", answer))
    
        if len(matches) >= 2:
            # Posizione finale del secondo [/INST]
            cut_pos = matches[1].end()
            clean_answer = answer[cut_pos:].lstrip()  # Rimuove eventuali spazi o \n iniziali
        
        clean_chat = [
            {"role": "user", "content": chat['prompt']},
            {"role": "assistant", "content": clean_answer}
        ]
        return clean_chat

    def parseMistralForLlamaguard(self, chat):
        clean_chat = [
            {"role": "user", "content": chat['prompt']},
            {"role": "assistant", "content": chat['model_response']}
        ]
        return clean_chat