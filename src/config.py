import os
from dotenv import load_dotenv

# Configuration pour Cloud Run
def setup_cloud_run_env():
    """Configure l'environnement pour Cloud Run"""
    
    # Configuration des répertoires temporaires
    temp_dirs = ['/tmp/crew', '/tmp/transformers', '/tmp/hf', '/tmp/cache']
    
    for temp_dir in temp_dirs:
        try:
            os.makedirs(temp_dir, exist_ok=True)
            os.chmod(temp_dir, 0o777)  # Permissions complètes
        except Exception as e:
            print(f"Warning: Could not create {temp_dir}: {e}")
    
    # Variables d'environnement pour CrewAI
    os.environ.setdefault('CREW_STORAGE_DIR', '/tmp/crew')
    os.environ.setdefault('HOME', '/tmp')
    os.environ.setdefault('TMPDIR', '/tmp')
    
    # Variables pour les modèles ML
    os.environ.setdefault('TRANSFORMERS_CACHE', '/tmp/transformers')
    os.environ.setdefault('HF_HOME', '/tmp/hf')
    os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', '/tmp/transformers')
    
    # Désactiver les barres de progression
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Appeler la configuration au début
setup_cloud_run_env()

# Charger les variables d'environnement
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Tuple, Optional, Type

#########################################################################################################
# formatage du json
def format_cv(document):
    def format_section(title, data, indent=0):
        prefix = "  " * indent
        lines = [f"{title}:"]
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}- {k.capitalize()}:")
                    lines.extend(format_section("", v, indent + 1))
                else:
                    lines.append(f"{prefix}- {k.capitalize()}: {v}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                lines.append(f"{prefix}- Élément {i + 1}:")
                lines.extend(format_section("", item, indent + 1))
        else:
            lines.append(f"{prefix}- {data}")
        return lines
    sections = []
    for section_name, content in document.items():
        title = section_name.replace("_", " ").capitalize()
        sections.extend(format_section(title, content))
        sections.append("") 
    return "\n".join(sections)


def read_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    cv_text = ""
    for page in pages:
        cv_text += page.page_content + "\n\n"
    return cv_text    

#########################################################################################################        
# modéles 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_openai = "gpt-4o"  

def crew_openai():
    """Configuration CrewAI pour Cloud Run"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=OPENAI_API_KEY
        )
        return llm
    except Exception as e:
        print(f"Error initializing CrewAI OpenAI: {e}")
        raise

def chat_openai():
    """Configuration Chat OpenAI pour Cloud Run"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.6,
            api_key=OPENAI_API_KEY
        )
        return llm
    except Exception as e:
        print(f"Error initializing Chat OpenAI: {e}")
        raise
