# 🔐 Guardrails and stuff

![Badge: Versione](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Badge: Licenza](https://img.shields.io/badge/license-MIT-green.svg)
![Badge: Stato](https://img.shields.io/badge/status-attivo-success)

Questo progetto mira a comprendere quando siano rischiosi gli LLM senza un fine tuning legato alle conversazioni non tossiche e quanto siano efficaci i vari guardrail utilizzabili come una "black box".

## 🧩 Caratteristiche

- scaricamento di LLM quali Llama versione 2 7B, Mistral 7B e Llama Chat 7B. 
- salvataggio in locale dei suddetti LLM con la possibilità di recuperare i modelli dal disco anziche scaricarli
- scaricamento di Guardrail quali Llamaguard 7B
- salvataggio in locale dei suddetti Guardrail
- possibilità di scegliere il dataset che si preferisce (tra quelli proposti)
- salvataggio e recupero delle rispsote date dagli LLM sul un dataset generico
- stime dei risultati attraverso l'utilizzo di un guardrail.

## 📦 Installazione
Per poter funzionare necessità di una GPU Nvidia, se non ce l'hai problema tuo. 
Qui è descritta l'installazione su Windows (su linux è analoga, dovrebbe cambiare solo la parte legata a Microsoft C++ Build Tools).
Superato questo enorme scoglio, servono: Git, Python e Microsoft C++ Build Tools

### <img src="https://git-scm.com/images/logos/downloads/Git-Icon-1788C.png" alt="git logo" width="20"/> Git
Per clonare la repo:
```bash
git clone https://github.com/AleLodetti/Guardrails/tree/main/guardrails_project
cd tuo-repo
```

### <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="20"/> Python 
Io utilizzo Python 3.11 (l'utlima versione forse non va). Per installare Python vai su https://www.python.org/downloads/ e durante l'installazione SPUNTA "add python to Path" (subito all'inizio) ed installa anche pip. Per verificare
```bash
python --version (o py --version)
pip --version
```

### <img src="https://visualstudio.microsoft.com/wp-content/uploads/2021/10/Product-Icon.svg" alt="msvc logo" width="20"/> Microsoft C++ Build Tools
Alcuni pacchetti che verranno installati nei requirements richiedono C++ per essere compilati perciò durante l'installazione di questo tool è necessario spuntare "Desktop development with C++" e "C++ CMake tools for Windows". Può essere scricato a questo link https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Creazione dell'ambiente virtuale
Una volta pullata la repo, spostarsi nella root del progetto da PowerShell o da terminale VSCode e digitare:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1 #su PowerShell runnata come amministratiore
```

## Installazione dei Requirements
Ad ambiente virtuale attivo, si legge (venv) sul terminale, digitare sempre sul terminale:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirments-gpu.txt
```

 # Esecuzione
 Per eseguire il progetto digitare:
 ```bash
 python -m main
 ```
A questo punto compaiono dei comandi da eseguire da terminale; facile facile.

 Potrebberro verificarsi errori legati alle librerie non lette, in tal caso semplicemente pip uninstall libreria e pip install libreria e dovrebbe sistemarsi.

---

 # Developed
 Developed by:
 📧 Contatti: [lodetti.alessandro02@gmail.com]  
🔗 GitHub: [@AleLodetti](https://github.com/AleLodetti)

---

<p align="center">
  <i>Grazie per aver letto. Buon coding! 🚀</i>
</p>