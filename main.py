#a function to extract answers from llama chat
from guardrails_project.Util import analyze
from guardrails_project.Util.instantiateModelAndFillFile import instantiateModelAndFillFile
from guardrails_project.Util.runGuardrail import runGuardrail
from guardrails_project.Util.analyze import analyzeMetrics

if __name__ == "__main__":
    """
    Main entry point for the script. It allows the user to choose between running a model or testing responses.
    The user can instantiate a model and fill a file with responses or run guardrail tests on existing responses.
    The script will continue to prompt the user for actions until they choose to exit.
    """

    
    while True:
        print("Do you want to instantiate a LLM and run it, do you want to run the tests or do you want to analyze the metrics? (llm/test/analyze)  ")
        choice = input().strip().lower()

        if choice == "llm":
            instantiateModelAndFillFile()
        elif choice == "test":
            runGuardrail()
        elif choice == "analyze":
            analyzeMetrics()
        else:
            print("Invalid choice. Please enter 'llm' or 'test!!! ")
            continue
        
        print("Do you want to run another operation? (Y/n)")
        another_operation = input().strip().lower()
        if another_operation != 'y':
            print("Exiting the program.")
            break

    print("Thank you for using the script!")
        