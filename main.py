#a function to extract answers from llama chat
from guardrails_project.Util import analyze
from guardrails_project.Util.instantiateModelAndFillFile import instantiateModelAndFillFile
from guardrails_project.Util.runGuardrail import runGuardrail
from guardrails_project.Util.analyze import analyzeMetrics
import torch

if __name__ == "__main__":
    """
    Main entry point for the script. It allows the user to choose between running a model or testing responses.
    The user can instantiate a model and fill a file with responses or run guardrail tests on existing responses.
    The script will continue to prompt the user for actions until they choose to exit.
    """

    #torch.cuda.empty_cache()

    while True:
        while True:
            print("Please select the guardrail system you want to inspect:")
            print("1. PerspectiveAPI")
            print("2. Detect Jailbreak")
            print("3. Prompt Shields")
            print("4. Llama Guard")
            choice = input("Enter the number corresponding to your choice: ").strip()

            if choice in ["1", "2", "3", "4"]:
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        if choice == "1":
            print("You have selected PerspectiveAPI.")
            runGuardrail(choice)
        elif choice == "2":
            print("You have selected Detect Jailbreak.")
            runGuardrail(choice)
        elif choice == "3":
            print("You have selected Prompt Shields.")
            runGuardrail(choice)
        elif choice == "4":
            while True:
                print("You have selected Llama Guard. Do you want to instantiate a LLM and run it, do you want to run the tests or do you want to analyze the metrics? (llm/test/analyze)  ")
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
        print("Do you want to use another guardrail? (Y/n)")
        another_operation = input().strip().lower()
        if another_operation != 'y':
            print("Exiting the program.")
            break
    print("Thank you for using the script!")
        