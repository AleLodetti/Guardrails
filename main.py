#a function to extract answers from llama chat
from guardrails_project.Util.instantiateModelAndFillFile import instantiateModelAndFillFile
from guardrails_project.Util.runGuardrail import runGuardrail

if __name__ == "__main__":
    
    while True:
        print("Do you want to instantiate a LLM and run it or do you want to run the tests? (llm/test)  ")
        choice = input().strip().lower()

        if choice == "llm":
            instantiateModelAndFillFile()
        elif choice == "test":
            runGuardrail()
        else:
            print("Invalid choice. Please enter 'llm' or 'test!!! ")
            continue
        
        print("Do you want to run another operation? (Y/n)")
        another_operation = input().strip().lower()
        if another_operation != 'y':
            print("Exiting the program.")
            break
        