# LangChain + Groq LLaMA: Minimal Drug Discovery Agent Scaffold

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import ArxivLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import AnalyzeDocumentChain
from langchain.schema import Document
import os
from dotenv import load_dotenv



apiKey = os.getenv("GROQ_API_KEY")
if not apiKey:
    raise ValueError("GROQ_API_KEY not found in environment")


groq_api_key = os.getenv("GROQ_API_KEY", apiKey)

groq_llm = ChatGroq(
    api_key=groq_api_key,
    model="llama3-8b-8192",  
    temperature=0
)

# === Tool 1: Extract druggable targets from research ===
target_prompt = PromptTemplate(
    input_variables=["disease"],
    template="""You are a biomedical researcher. Based on the disease {disease}, identify the top 3 protein or gene targets in the pathogen that are druggable. Give a short reason for each."""
)
target_chain = LLMChain(llm=groq_llm, prompt=target_prompt)

def get_targets(disease):
    return target_chain.invoke({"disease": disease})

# === Tool 2: Suggest small molecules to inhibit a target ===
molecule_prompt = PromptTemplate(
    input_variables=["target"],
    template="""You are a drug discovery chemist. Suggest 2 small molecule inhibitors for the protein target: {target}. Include a brief explanation."""
)
molecule_chain = LLMChain(llm=groq_llm, prompt=molecule_prompt)

def suggest_molecules(target):
    return molecule_chain.invoke({"target": target})

# === Tool 3: Optimize the molecule ===
optimize_prompt = PromptTemplate(
    input_variables=["molecule"],
    template="""You are optimizing a drug candidate. Improve the molecule {molecule} for better bioavailability and lower toxicity. Provide a rationale."""
)
optimize_chain = LLMChain(llm=groq_llm, prompt=optimize_prompt)

def optimize_molecule(molecule):
    return optimize_chain.invoke({"molecule": molecule})

# === Tool 4: Predict ADMET profile ===
admet_prompt = PromptTemplate(
    input_variables=["molecule"],
    template="""Analyze the molecule {molecule} for its ADMET properties (Absorption, Distribution, Metabolism, Excretion, Toxicity). Give a concise profile."""
)
admet_chain = LLMChain(llm=groq_llm, prompt=admet_prompt)

def get_admet(molecule):
    return admet_chain.invoke({"molecule": molecule})

# === AGENT SETUP ===
tools = [
    Tool(name="Identify Targets", func=get_targets, description="Finds druggable targets from disease name"),
    Tool(name="Suggest Molecules", func=suggest_molecules, description="Suggests inhibitors for a target"),
    Tool(name="Optimize Molecule", func=optimize_molecule, description="Improves molecule properties"),
    Tool(name="Predict ADMET", func=get_admet, description="Predicts ADMET profile for a molecule"),
]

agent = initialize_agent(tools, groq_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# # === MAIN EXECUTION ===
# if __name__ == "__main__":
#     disease = "malaria"
#     print("\n--- Drug Discovery Pipeline for:", disease, "---")
#     targets = get_targets(disease)
#     print("\nTargets:\n", targets)

#     target = "Plasmodium falciparum ATPase"  # Replace dynamically later
#     molecules = suggest_molecules(target)
#     print("\nSuggested Molecules:\n", molecules)

#     molecule = "Artemisinin"  # Example molecule
#     optimized = optimize_molecule(molecule)
#     print("\nOptimized Molecule:\n", optimized)

#     admet = get_admet(molecule)
#     print("\nADMET Profile:\n", admet)
# === FULL AUTONOMOUS EXECUTION CHAIN ===

# Defines qa_chain for question-answering
qa_chain = load_qa_chain(llm=groq_llm, chain_type="stuff")

# === FULL AUTONOMOUS EXECUTION CHAIN ===
# this exceeds the token limit of the model which is 6000 reqs so we need to split the document
# into smaller chunks before passing it to the model

# def extract_targets_from_arxiv(disease_query):
#     print(f"🔎 Searching ArXiv for: {disease_query}")
#     loader = ArxivLoader(query=disease_query, max_results=3)
#     docs = loader.load()

#     response = qa_chain.invoke({
#         "input_documents": docs,
#         "question": f"What are the druggable protein targets in {disease_query}?"
#     })

#     return response["output_text"]

def extract_targets_from_arxiv(disease_query):
    print(f"🔎 Searching ArXiv for: {disease_query}")
    
    # Load metadata (usually includes abstract) instead of full paper
    loader = ArxivLoader(query=disease_query, load_max_docs=1)  # keep small
    raw_docs = loader.load()
    
    # Just use the abstract and metadata
    filtered_docs = []
    for doc in raw_docs:
        filtered_docs.append(doc)  # Keep this minimal — doc.page_content is typically abstract + metadata

    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(filtered_docs)

    # Use map-reduce chain to avoid hitting Groq token limits
    chain = load_qa_chain(llm=groq_llm, chain_type="map_reduce")

    response = chain.invoke({
        "input_documents": split_docs,
        "question": f"What are the most likely druggable targets for {disease_query}?"
    })

    return response["output_text"]

# def run_pipeline(disease):
#     print(f"\n🚀 Starting AI pipeline for: {disease}")

#     targets_result = get_targets(disease)
#     print("\n🎯 Targets Identified:\n", targets_result["text"])

#     # Let's assume the first target is the best
#     first_target = targets_result["text"].split("\n")[0]
#     molecules_result = suggest_molecules(first_target)
#     print("\n🧪 Suggested Molecules:\n", molecules_result["text"])

#     # Again, pick first molecule
#     first_molecule = molecules_result["text"].split("\n")[0]
#     optimized_result = optimize_molecule(first_molecule)
#     print("\n🔧 Optimized Molecule:\n", optimized_result["text"])

#     admet_result = get_admet(first_molecule)
#     print("\n🧬 ADMET Profile:\n", admet_result["text"])

def run_pipeline_from_arxiv(disease):
    print(f"\n🚀 Starting AI research pipeline for: {disease}")
    targets = extract_targets_from_arxiv(disease)
    print("\n📘 Extracted Targets:\n", targets)

    target_name = targets.split("\n")[0]
    mols = suggest_molecules(target_name)
    print("\n🧪 Suggested Molecules:\n", mols["text"])

    mol = mols["text"].split("\n")[0]
    opt = optimize_molecule(mol)
    print("\n🔧 Optimized Molecule:\n", opt["text"])

    admet = get_admet(mol)
    print("\n🧬 ADMET Profile:\n", admet["text"])
    print("\n🚀 Pipeline completed!")

# Trigger
if __name__ == "__main__":
    run_pipeline_from_arxiv("malaria")

# if __name__ == "__main__":
#    result = extract_targets_from_arxiv("malaria plasmodium")
# print("\n🧬 Druggable Targets:\n", result)


