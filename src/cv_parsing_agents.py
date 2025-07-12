import os
import json

from src.crew.crew_pool import analyse_cv
from src.config import load_pdf

def clean_dict_keys(data):
    if isinstance(data, dict):
        return {str(key): clean_dict_keys(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_dict_keys(element) for element in data]
    else:
        return data

class CvParserAgent:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def process(self) -> dict:
        """
        Traite le fichier PDF pour en extraire le contenu sous forme de JSON.
        Ne se connecte à aucune base de données.
        
        Retourne :
            Un dictionnaire contenant les données extraites du CV, ou None en cas d'erreur.
        """
        print(f"Début du traitement du CV : {self.pdf_path}")
        
        try:
            cv_text_content = load_pdf(self.pdf_path)
            crew_output = analyse_cv(cv_text_content)

            if not crew_output or not hasattr(crew_output, 'raw') or not crew_output.raw.strip():
                print("Erreur : L'analyse par le crew n'a pas retourné de résultat.")
                return None
            raw_string = crew_output.raw
            json_string_cleaned = raw_string
            if '```' in raw_string:
                json_part = raw_string.split('```json')[1].split('```')[0]
                json_string_cleaned = json_part.strip()
            profile_data = json.loads(json_string_cleaned)
            return clean_dict_keys(profile_data)

        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON : {e}")
            print(f"Données brutes reçues : {crew_output.raw}")
            return None
        except Exception as e:
            print(f"Une erreur inattendue est survenue dans CvParserAgent : {e}")
            return None