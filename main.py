# === Imports ===
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import ArxivLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict
import json
import os
from langchain.schema import Document
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from dotenv import load_dotenv


from pdb_utils import search_pdb_structures, download_pdb_file

# === Init ===
load_dotenv()
set_llm_cache(InMemoryCache())

CACHE_FILE = "arxiv_cache.json"

# === Cache Helpers ===
def load_cached_result(disease):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        return cache.get(disease)
    return None

def save_cached_result(disease, result):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    cache[disease] = result
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# === API Setup ===
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

groq_llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", temperature=0)

# === Prompts and Chains ===
target_prompt = PromptTemplate(
    input_variables=["disease"],
    template="""You are a biomedical researcher. Based on the disease {disease}, identify the top 3 protein or gene targets in the pathogen that are druggable. Give a short reason for each."""
)
target_chain = LLMChain(llm=groq_llm, prompt=target_prompt)

molecule_prompt = PromptTemplate(
    input_variables=["target"],
    template="""You are a drug discovery chemist. Suggest 2 small molecule inhibitors for the protein target: {target}. Include a brief explanation."""
)
molecule_chain = LLMChain(llm=groq_llm, prompt=molecule_prompt)

optimize_prompt = PromptTemplate(
    input_variables=["molecule"],
    template="""You are optimizing a drug candidate. Improve the molecule {molecule} for better bioavailability and lower toxicity. Provide a rationale."""
)
optimize_chain = LLMChain(llm=groq_llm, prompt=optimize_prompt)

admet_prompt = PromptTemplate(
    input_variables=["molecule"],
    template="""Analyze the molecule {molecule} for its ADMET properties (Absorption, Distribution, Metabolism, Excretion, Toxicity). Give a concise profile."""
)
admet_chain = LLMChain(llm=groq_llm, prompt=admet_prompt)

qa_chain = load_qa_chain(llm=groq_llm, chain_type="map_reduce")

# === Drug Discovery State ===
class DrugDiscoveryState(TypedDict):
    disease: str
    targets: str
    selected_target: str
    molecules: str
    selected_molecule: str
    optimized_molecule: str
    admet_profile: str
    pdb_path: str 

# === Pipeline Nodes ===

def extract_targets(state: DrugDiscoveryState) -> DrugDiscoveryState:
    disease = state["disease"]
    search_query = f"{disease} drug targets"

    print(f"🔎 Searching ArXiv for: {search_query}")

    doc_cache_key = f"{disease}_docs"
    doc_cache = load_cached_result(doc_cache_key)

    if doc_cache:
        print("✅ Using cached documents")
        docs = [Document(page_content=doc) for doc in doc_cache]
    else:
        print("📡 Fetching documents from ArXiv...")
        loader = ArxivLoader(query=search_query, load_max_docs=3)
        docs = loader.load()
        save_cached_result(doc_cache_key, [doc.page_content for doc in docs])

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

    result = qa_chain.invoke({
        "input_documents": chunks,
        "question": f"What are the top protein or gene drug targets of the {disease} pathogen according to this paper?"
    })

    targets = result["output_text"]
    target_cache_key = f"{disease}_targets"
    save_cached_result(target_cache_key, targets)

    state["targets"] = targets
    state["selected_target"] = targets.split("\n")[0] if "\n" in targets else targets.strip()
    return state

def fetch_pdb_for_target(state: DrugDiscoveryState) -> DrugDiscoveryState:
    target = state["selected_target"]
    print(f"📦 Searching PDB for target: {target}")

    pdb_ids = search_pdb_structures(target, search_by_name=True)
    if not pdb_ids:
        print("❌ No PDB structures found.")
        state["pdb_path"] = "N/A"
        return state

    file_path = download_pdb_file(pdb_ids[0])
    print(f"📁 Downloaded PDB file: {file_path}")
    state["pdb_path"] = file_path
    return state

def generate_molecules(state: DrugDiscoveryState) -> DrugDiscoveryState:
    print(f"🧪 Suggesting molecules for target: {state['selected_target']}")
    result = molecule_chain.invoke({"target": state["selected_target"]})
    state["molecules"] = result["text"]
    state["selected_molecule"] = result["text"].split("\n")[0]
    return state

def optimize(state: DrugDiscoveryState) -> DrugDiscoveryState:
    print(f"🔧 Optimizing molecule: {state['selected_molecule']}")
    result = optimize_chain.invoke({"molecule": state["selected_molecule"]})
    state["optimized_molecule"] = result["text"]
    return state

def analyze_admet(state: DrugDiscoveryState) -> DrugDiscoveryState:
    print(f"🧬 Analyzing ADMET for: {state['selected_molecule']}")
    result = admet_chain.invoke({"molecule": state["selected_molecule"]})
    state["admet_profile"] = result["text"]
    return state

# === LangGraph Flow ===
builder = StateGraph(DrugDiscoveryState)
builder.add_node("FindTargets", extract_targets)
builder.add_node("FetchPDB", fetch_pdb_for_target)  
builder.add_node("SuggestMolecules", generate_molecules)
builder.add_node("OptimizeMolecule", optimize)
builder.add_node("ADMETAnalysis", analyze_admet)

builder.set_entry_point("FindTargets")
builder.add_edge("FindTargets", "FetchPDB")          
builder.add_edge("FetchPDB", "SuggestMolecules")     
builder.add_edge("SuggestMolecules", "OptimizeMolecule")
builder.add_edge("OptimizeMolecule", "ADMETAnalysis")
builder.add_edge("ADMETAnalysis", END)

graph = builder.compile()

# === Execution ===
if __name__ == "__main__":
    result = graph.invoke({"disease": "malaria"})

    print("\n🎯 Targets Identified:\n", result["targets"])
    print("\n📁 PDB Structure Path:\n", result["pdb_path"])
    print("\n🧪 Suggested Molecules:\n", result["molecules"])
    print("\n🔧 Optimized Molecule:\n", result["optimized_molecule"])
    print("\n🧬 ADMET Profile:\n", result["admet_profile"])
