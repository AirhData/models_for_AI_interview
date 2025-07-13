import tempfile
import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.cv_parsing_agents import CvParserAgent
from src.interview_simulator.entretient_version_prod import InterviewProcessor

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
TIMEOUT_SECONDS = 300  # 5 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie - VERSION CLOUD RUN"""
    logger.info("🚀 Démarrage Cloud Run...")
    
    try:
        os.environ.setdefault('CREW_STORAGE_DIR', '/tmp/crew')
        os.environ.setdefault('HOME', '/tmp')
        os.environ.setdefault('TMPDIR', '/tmp')
        os.makedirs('/tmp/crew', exist_ok=True)
        os.makedirs('/tmp/transformers', exist_ok=True)
        os.makedirs('/tmp/hf', exist_ok=True)
        logger.info("Vérification des imports...")
        import torch
        import transformers
        logger.info("✅ Dépendances ML disponibles")

        try:
            from crewai import Agent
            logger.info("✅ CrewAI disponible")
        except Exception as e:
            logger.warning(f"⚠️ CrewAI warning: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Avertissement au démarrage : {e}")
    logger.info("✅ Application prête")
    yield
    logger.info("🛑 Arrêt de l'application")

app = FastAPI(
    title="API d'IA pour la RH",
    description="Une API pour le parsing de CV et la simulation d'entretiens.",
    version="1.0.0",
    lifespan=lifespan
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

@app.get("/health", tags=["Status"], summary="Health check détaillé")
def health_check():
    """Health check pour Cloud Run avec status des modèles"""
    try:
        import torch
        import transformers
        models_status = {}
        if hasattr(app.state, 'model_analyzer') and app.state.model_analyzer:
            analyzer = app.state.model_analyzer
            models_status = {
                "sentiment_available": analyzer.sentiment_analyzer is not None,
                "similarity_available": analyzer.similarity_model is not None,
                "intent_available": analyzer.intent_classifier is not None,
                "models_loaded": analyzer.models_loaded
            }
        else:
            models_status = {"preloaded": False, "message": "Modèles non pré-chargés"}
        return {
            "status": "healthy",
            "pytorch_available": True,
            "transformers_available": True,
            "cuda_available": torch.cuda.is_available(),
            "models_status": models_status,
            "cache_dir": os.environ.get('TRANSFORMERS_CACHE', 'default')
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/parse-cv/", tags=["CV Parsing"], summary="Analyser un CV au format PDF")
async def parse_cv_endpoint(file: UploadFile = File(...)):
    """Version sécurisée pour Cloud Run"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Le fichier doit être au format PDF.")
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Fichier trop volumineux. Maximum: {MAX_FILE_SIZE} bytes")
    
    temp_file = None
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Fichier vide.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="cv_") as temp_file:
            temp_file.write(contents)
            temp_file.flush()
            temp_path = temp_file.name
        
        logger.info(f"Fichier temporaire créé : {temp_path}")
        cv_agent = CvParserAgent(pdf_path=temp_path)
        parsed_data = await asyncio.wait_for(
            run_in_threadpool(cv_agent.process),
            timeout=TIMEOUT_SECONDS
        )
        
        if not parsed_data:
            raise HTTPException(status_code=500, detail="Échec du parsing du CV.")
        
        logger.info("Parsing du CV réussi.")
        return parsed_data
        
    except asyncio.TimeoutError:
        logger.error("Timeout lors du parsing du CV")
        raise HTTPException(status_code=504, detail="Timeout lors du traitement du CV")
    except Exception as e:
        logger.error(f"Erreur lors du parsing du CV : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")
    finally:
        if temp_file and hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.info(f"Fichier temporaire supprimé : {temp_file.name}")
            except Exception as cleanup_error:
                logger.warning(f"Erreur lors de la suppression du fichier temporaire : {cleanup_error}")

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
        ai_response_object = await asyncio.wait_for(
            run_in_threadpool(processor.run, messages=request.messages),
            timeout=TIMEOUT_SECONDS
        )
        
        final_text_response = ""
        if isinstance(ai_response_object.get('messages'), list) and ai_response_object['messages']:
            last_message = ai_response_object['messages'][-1]
            if hasattr(last_message, 'content'):
                final_text_response = last_message.content       
        
        if not final_text_response:
            final_text_response = str(ai_response_object)
        
        logger.info(f"Simulation terminée. Réponse extraite : '{final_text_response[:100]}...'")
        return {"response": final_text_response}
        
    except asyncio.TimeoutError:
        logger.error("Timeout lors de la simulation d'entretien")
        raise HTTPException(status_code=504, detail="Timeout lors de la simulation")
    except Exception as e:
        logger.error(f"Erreur interne dans /simulate-interview/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
