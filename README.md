# 🔐 Breaking the Guardrails

![Badge: Versione](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Badge: Licenza](https://img.shields.io/badge/license-MIT-green.svg)
![Badge: Stato](https://img.shields.io/badge/status-attivo-success)

This project aims to understand how effective various black-box guardrail systems for Large Language Models are. These systems aim to safeguard both the input and output channels of LLMs by dually detecting adversarial prompts and preventing the generation of unsafe responses.

## 🧩 Guardrails object of the study
We present a systematic, empirical evaluation of four ready-to-use guardrail systems: Google's Perspective API, Guardrails AI's Detect Jailbreak, Azure's Prompt Shields, and Meta's LlamaGuard. 

## 🔍 PerspectiveAPI
### Requirements:
* A Google account that includes Google Cloud
* A project created in Google Cloud
* The API key, which can be requested filling this [form](https://docs.google.com/forms/d/e/1FAIpQLSdhBBnVVVbXSElby-jhNnEj-Zwpt5toQSCFsJerGfpXW66CuQ/viewform). Remember to insert your API key in [perspectiveAPI.py](guardrails_project/Guardrails/perspectiveAPI.py).
You can find further detail in PerspectiveAPI [documentation](https://developers.perspectiveapi.com/s/docs-get-started?language=en_US).
## 🕵️‍♂️ Detect Jailbreak
### Requirements:
* Being registered on [Guardrails Hub](https://hub.guardrailsai.com/)
* Create an API key on your profile
* Install the requirements for the whole project (see below) which contain the **_guardrails-ai_ import**
Now you can type the following commands:
```bash
guardrails configure (and this is when you'll need to paste your API key)
guardrails hub install hub://guardrails/detect_jailbreak
```
Note: if **guardrails** is not recognized as a command, you may need to find the path to **guardrails.exe** and use that.
Further information can be found on Guardrails AI [blog](https://www.guardrailsai.com/blog/advanced-pii-and-jailbreak).

## 🛡️ Prompt Shields
### Requirements:
* An Azure subscription
* Content Safety resource in the Azure portal to get your key and endpoint
* cURL installed
You can find further detail in Azure [documentation](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak?pivots=programming-language-rest). Remember to insert your API key in [promptShields.py](guardrails_project/Guardrails/promptShields.py).
From the command prompt, run:

```bash
pip install -r requirments-dev.txt
```

## 🦙 LlamaGuard: 


- Downloading LLMs such as Llama version 2 7B, Mistral 7B, and Llama Chat 7B.
- Local saving of the aforementioned LLMs with the option of recovering the models from disk instead of downloading them.
- Downloading Guardrails such as Llamaguard 7B
- Local saving of the aforementioned Guardrails
- Option to select the preferred dataset (from those proposed)
- Saving and retrieving the responses given by the LLMs on a generic dataset
- Result estimations using a guardrail.



## 📦 Installation
To run, you need an Nvidia GPU. If you don't have one, that's your problem.
The installation on Windows is described here (on Linux, it's similar, only the part related to Microsoft C++ Build Tools should change).
Once you've overcome this huge obstacle, you'll need: Git, Python, and Microsoft C++ Build Tools.

### <img src="https://git-scm.com/images/logos/downloads/Git-Icon-1788C.png" alt="git logo" width="20"/> Git
Use this command to clone the repository:
```bash
git clone https://github.com/AleLodetti/Guardrails/tree/main/guardrails_project
cd your-repo
```

### <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="20"/> Python 
I'm using Python 3.11 (the latest version may not work). To install Python, go to https://www.python.org/downloads/ and during the installation, CHECK "add python to Path" (right at the beginning) and also install pip. To verify
```bash
python --version (o py --version)
pip --version
```

### <img src="https://visualstudio.microsoft.com/wp-content/uploads/2021/10/Product-Icon.svg" alt="msvc logo" width="20"/> Microsoft C++ Build Tools
Some packages that will be installed in the requirements require C++ to be compiled, so when installing this tool, you must check "Desktop development with C++" and "C++ CMake tools for Windows." It can be downloaded from this link: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Creating the Virtual Environment
Once the repo is pulled, navigate to the project root using PowerShell or the VSCode terminal and type:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1 #on PowerShell run as admin
```

## Installation of all the Requirements
When the virtual environment is activated, you can read (venv) on the terminal. Type the following commands:
```bash
pip install -r requirements-dev.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirments-gpu.txt
```

 # ▶️ Execution
To execute the project, type the following:
 ```bash
 python -m main
 ```
At this point, some commands to run from the terminal appear; easy peasy.

You may encounter errors related to unread libraries; in that case, simply pip uninstall library and pip install library, and it should fix it.

---

 # Developed
 Developed by:

📧 Mail: [lodetti.alessandro02@gmail.com]
🔗 GitHub: [@AleLodetti](https://github.com/AleLodetti)

 📧 Mail: [silvia.parolin@mail.polimi.it]
🔗 GitHub: [@silviaparolin](https://github.com/silviaparolin)

---

<p align="center">
  <i>Thanks for your time</i>
</p>
