import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from src.cv_parsing_agents import CvParserAgent
from src.interview_simulator.entretient_version_prod import InterviewProcessor

app = FastAPI(
    title="API d'IA pour la RH",
    description="Une API pour le parsing de CV et la simulation d'entretiens.",
    version="1.0.0"
)
class InterviewRequest(BaseModel):
    cv_document: Dict[str, Any] = Field(..., example={"candidat": {"nom": "John Doe", "compétences": {"hard_skills": ["Python", "FastAPI"]}}})
    job_offer: Dict[str, Any] = Field(..., example={"poste": "Développeur Python", "description": "Recherche développeur expérimenté..."})
    messages: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]

class HealthCheck(BaseModel):
    status: str = Field(default="ok", example="ok")


@app.get("/", tags=["Status"], summary="Vérification de l'état de l'API")
def read_root() -> HealthCheck:
    """Vérifie que l'API est en cours d'exécution."""
    return HealthCheck(status="ok")

# --- Endpoint du parser de CV ---
@app.post("/parse-cv/")
async def parse_cv_endpoint(file: UploadFile = File(...)):
    # Utiliser un bloc "with" garantit que le fichier temporaire est bien supprimé après usage
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            # Écrire le contenu du fichier uploadé dans le fichier temporaire
            content = await file.read()
            temp_file.write(content)
            
            # Récupérer le chemin du fichier temporaire
            temp_file_path = temp_file.name

        print(f"Fichier temporaire créé à l'adresse : {temp_file_path}") # Log pour le débogage

        # Instancier l'agent AVEC le bon chemin
        cv_parser = CvParserAgent(file_path=temp_file_path) # C'EST LA LIGNE LA PLUS IMPORTANTE
        
        # Lancer le parsing
        parsed_data = cv_parser.parse() # Assurez-vous que la méthode s'appelle bien "parse"

        # Vérifier si le parsing a réussi
        if not parsed_data:
            raise HTTPException(status_code=500, detail="Le parsing du CV n'a retourné aucune donnée.")

        return parsed_data

    except Exception as e:
        # Cette partie est cruciale pour le débogage. Elle affichera la VRAIE erreur.
        print(f"Une erreur détaillée est survenue : {e}")
        raise HTTPException(status_code=500, ors de la suppression du fichier temporaire : {cleanup_error}")

# --- Endpoint de simulation d'entretien ---
@app.post("/simulate-interview/", tags=["Simulation d'Entretien"], summary="Gérer une conversation d'entretien")
async def simulate_interview_endpoint(request: InterviewRequest):
    try:
        logger.info("Création de l'instance InterviewProcessor.")
        processor = InterviewProcessor(
            cv_document=request.cv_document,
            job_offer=request.job_offer,
            conversation_history=request.conversation_history
        )
        logger.info("Lancement de la simulation dans un threadpool.")
        ai_response_object = await run_in_threadpool(processor.run, messages=request.messages)
        final_text_response = ""
        if isinstance(ai_response_object.get('messages'), list) and ai_response_object['messages']:
            last_message = ai_response_object['messages'][-1]
            if hasattr(last_message, 'content'):
                final_text_response = last_message.content       
        if not final_text_response:
            final_text_response = str(ai_response_object)
        logger.info(f"Simulation terminée. Réponse extraite : '{final_text_response}'")
        return {"response": final_text_response}
    except Exception as e:
        logger.error(f"Erreur interne dans /simulate-interview/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
