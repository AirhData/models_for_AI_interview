import os
import json
import logging

logger = logging.getLogger(__name__)

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
        Version sécurisée pour Cloud Run
        """
        logger.info(f"Début du traitement du CV : {self.pdf_path}")
        
        try:
            # Import avec gestion d'erreur
            from src.config import load_pdf
            cv_text_content = load_pdf(self.pdf_path)
            logger.info(f"Contenu extrait : {len(cv_text_content)} caractères")
            
            # Import sécurisé de crew_pool
            try:
                from src.crew.crew_pool import analyse_cv
                logger.info("Lancement de l'analyse par le crew...")
                crew_output = analyse_cv(cv_text_content)
            except Exception as crew_error:
                logger.error(f"Erreur de permission : {crew_error}")
                # Fallback en cas d'erreur CrewAI
                return self._create_fallback_response(cv_text_content)

            # Traitement du résultat
            if not crew_output:
                logger.warning("Crew n'a pas retourné de résultat")
                return self._create_fallback_response(cv_text_content)
            
            # Si c'est déjà un dictionnaire (cas d'erreur géré)
            if isinstance(crew_output, dict):
                return clean_dict_keys(crew_output)
            
            # Si c'est un objet avec .raw
            if hasattr(crew_output, 'raw') and crew_output.raw:
                raw_string = crew_output.raw.strip()
                
                # Nettoyage du JSON si nécessaire
                if '```' in raw_string:
                    try:
                        json_part = raw_string.split('```json')[1].split('```')[0]
                        raw_string = json_part.strip()
                    except:
                        # Si le parsing échoue, utiliser tel quel
                        pass
                
                try:
                    profile_data = json.loads(raw_string)
                    return clean_dict_keys(profile_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur JSON : {e}")
                    logger.error(f"Raw data: {raw_string[:500]}...")
                    return self._create_fallback_response(cv_text_content)
            
            # Si aucun format reconnu
            logger.warning("Format de sortie crew non reconnu")
            return self._create_fallback_response(cv_text_content)

        except Exception as e:
            logger.error(f"Erreur critique dans CvParserAgent : {e}", exc_info=True)
            return self._create_fallback_response("Erreur lors de la lecture du CV")

    def _create_fallback_response(self, cv_content: str) -> dict:
        """Crée une réponse de fallback en cas d'erreur"""
        return {
            "candidat": {
                "informations_personnelles": {
                    "nom": "Extraction automatique échouée",
                    "email": "Non extrait",
                    "numero_de_telephone": "Non extrait", 
                    "localisation": "Non extrait"
                },
                "compétences": {
                    "hard_skills": ["Analyse manuelle requise"],
                    "soft_skills": []
                },
                "expériences": [{
                    "Poste": "Analyse manuelle requise",
                    "Entreprise": "Voir CV original",
                    "start_date": "Non spécifié",
                    "end_date": "Non spécifié",
                    "responsabilités": ["Consulter le CV original"]
                }],
                "projets": {
                    "professional": [],
                    "personal": []
                },
                "formations": [{
                    "degree": "Analyse manuelle requise",
                    "institution": "Voir CV original",
                    "start_date": "Non spécifié",
                    "end_date": "Non spécifié"
                }],
                "raw_content_length": len(cv_content),
                "status": "fallback_mode",
                "message": "L'extraction automatique a échoué. Analyse manuelle recommandée."
            }
        }
