import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Tuple, Optional, Type
from crewai import LLM
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

"""GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
model_google = "gemini/gemma-3-27b-it"
def chat_gemini():
    llm = ChatGoogleGenerativeAI("gemini/gemma-3-27b-it")"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_openai = "gpt-4o"  

def crew_openai():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )
    return llm

def chat_openai():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.6,
        api_key=OPENAI_API_KEY
    )
    return llm
