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
app.post("/parse-cv/", 
          response_model=ParsedCV, 
          tags=["CV Parsing"], 
          summary="Analyser un CV au format PDF")
async def parse_cv_endpoint(file: UploadFile = File(..., description="Le fichier CV au format PDF à analyser.")):
    """
    Recevez un fichier PDF, sauvegardez-le temporairement, analysez-le pour en
    extraire les informations pertinentes, puis retournez les données structurées.
    """
    if file.content_type != "application/pdf":
        logger.warning(f"Upload attempt with incorrect file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Le fichier doit être au format PDF.")
    
    tmp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name
            
        logger.info(f"Fichier temporaire créé pour le CV : {tmp_path}")
        cv_agent = CvParserAgent(pdf_path=tmp_path)
        parsed_data_dict = await run_in_threadpool(cv_agent.process)

        if not parsed_data_dict:
            logger.error("Le parsing du CV a échoué, aucune donnée n'a été retournée par l'agent.")
            raise HTTPException(status_code=500, detail="Échec du parsing du CV.")
        
        parsed_data = ParsedCV(**parsed_data_dict)
        
        logger.info("Parsing du CV réussi et données validées.")
        return parsed_data
        
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue dans l'endpoint de parsing : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"Fichier temporaire supprimé : {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Avertissement : Erreur lors de la suppression du fichier temporaire {tmp_path}: {cleanup_error}")

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

