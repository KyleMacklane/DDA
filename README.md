# 🧬 Drug Discovery AI Agent

This project is an **AI-powered drug discovery pipeline** built using [LangChain](https://www.langchain.com/), [Groq LLaMA-3](https://groq.com/), and [ArXiv](https://arxiv.org/) scientific literature. It automates the process from disease to drug candidate, including:

- 🔍 Target discovery
- 💊 Molecule generation
- 🔧 Optimization
- 🧪 ADMET profiling
- 🧬 Molecular docking integration **(AutoDock Vina)**

---

## 🚀 Features

- **Literature Search**: Uses ArXiv to find latest research on a given disease.
- **Target Identification**: Extracts druggable targets using QA chains.
- **Molecule Suggestion**: Generates small molecule inhibitors for the target.
- **Optimization**: Improves molecules for bioavailability and safety.
- **ADMET Analysis**: Provides a summary of absorption, distribution, metabolism, excretion, and toxicity.
- **Molecular Docking**: Automatically prepares receptors and ligands, then runs docking simulations using AutoDock Vina.
- 📦 **Caching**: Saves intermediate results (literature, targets, molecules) to skip recomputation.
- 🧠 **LLM-Driven Reasoning**: Powered by Groq's blazing-fast LLaMA-3 

---

## ⚙️ Requirements

- Python 3.10+
- AutoDock Vina installed and accessible via terminal (vina)

- OpenBabel (obabel) installed for ligand preparation
- Make sure you install AutoDock Vina and OpenBabel outside pip using system package managers or their respective installers:
 ```bash
 brew install vina open-babel (macOS)
 

sudo apt install autodock-vina openbabel (Linux)
```

Or download binaries from their official sites (Windows).

## 🔑 LLM Key Setup 
Create a `.env` file and replace this with your key
```bash 
GROQ_API_KEY=your_groq_api_key
```

## 📁 Project Structure
```bash
DDA/
├── docking/                # Receptor/ligand preparation and docking code
├── arxiv_cache.json         # Stores cached targets from ARXIV 
├── main.py                 # Pipeline entry point
├── requirements.txt        # Dependencies
└── README.md

```

## 📦 Installation

```bash
git clone https://github.com/KyleMacklane/DDA.git
cd DDA
pip install -r requirements.txt
python main.py
```

## 🧪 Example Run
Input Disease: Malaria
- → Searching ArXiv...
- → Extracting top protein targets...
- → Generating SMILES inhibitors...
- → Optimizing molecules...
- → Predicting ADMET...
- → Preparing ligands and receptors...
- → Docking using AutoDock Vina...
- → Output: Top scored compound + binding affinity

## 🖼️ Demo

Here's a screenshot of the AI pipeline in action:

![Pipeline Demo](images/docking.png)


## 📬 Coming Soon

- GUI frontend (streamlit or gradio)
- Multi-target scoring
- Lead compound ranking


