import os
import json
import tempfile
import logging
from crewai import Crew, Process
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Type

# Import des agents et tâches
from .agents import (
    report_generator_agent, skills_extractor_agent, experience_extractor_agent, 
    project_extractor_agent, education_extractor_agent, ProfileBuilderAgent, 
    informations_personnelle_agent
)
from .tasks import (
    generate_report_task, task_extract_skills, task_extract_experience, 
    task_extract_projects, task_extract_education, task_build_profile, 
    task_extract_informations
)

logger = logging.getLogger(__name__)

def setup_safe_crew_environment():
    """Configure un environnement sécurisé pour CrewAI sur Cloud Run"""
    try:
        # Utiliser un répertoire temporaire sécurisé
        temp_dir = tempfile.mkdtemp(prefix='crew_', dir='/tmp')
        
        # Configuration CrewAI
        os.environ['CREW_STORAGE_DIR'] = temp_dir
        os.environ['CREW_TELEMETRY'] = 'false'  # Désactiver la télémétrie
        
        logger.info(f"CrewAI configuré avec répertoire temporaire: {temp_dir}")
        return temp_dir
        
    except Exception as e:
        logger.error(f"Erreur configuration CrewAI: {e}")
        # Fallback vers /tmp si possible
        fallback_dir = '/tmp/crew_fallback'
        try:
            os.makedirs(fallback_dir, exist_ok=True)
            os.environ['CREW_STORAGE_DIR'] = fallback_dir
            return fallback_dir
        except:
            # Dernière tentative - utiliser le répertoire courant
            os.environ['CREW_STORAGE_DIR'] = '.'
            return '.'

@tool
def interview_analyser(conversation_history: list, job_description_text: str) -> str:
    """
    Analyse l'entretien avec gestion d'erreurs pour Cloud Run
    """
    try:
        # Configuration sécurisée
        temp_dir = setup_safe_crew_environment()
        
        interview_crew = Crew(
            agents=[report_generator_agent],
            tasks=[generate_report_task],
            process=Process.sequential,
            verbose=False,
            telemetry=False
        )
        
        # Import avec gestion d'erreur
        try:
            from src.deep_learning_analyzer import MultiModelInterviewAnalyzer
            analyzer = MultiModelInterviewAnalyzer()
            structured_analysis = analyzer.run_full_analysis(conversation_history, job_description_text)
        except Exception as e:
            logger.error(f"Erreur analyzer ML: {e}")
            # Fallback sans analyse ML
            structured_analysis = {
                "overall_similarity_score": 0.5,
                "sentiment_analysis": [],
                "intent_analysis": [],
                "raw_transcript": conversation_history,
                "error": "ML analysis unavailable"
            }
        
        final_report = interview_crew.kickoff(inputs={
            'structured_analysis_data': json.dumps(structured_analysis, indent=2)
        })
        
        # Nettoyage du répertoire temporaire
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        return str(final_report)
        
    except Exception as e:
        logger.error(f"Erreur critique dans interview_analyser: {e}")
        return f"Erreur lors de l'analyse de l'entretien: {str(e)}"

def analyse_cv(cv_content: str) -> dict:
    """Analyse de CV avec configuration sécurisée pour Cloud Run"""
    try:
        # Configuration sécurisée
        temp_dir = setup_safe_crew_environment()
        
        logger.info("Début de l'analyse CV avec CrewAI")
        
        crew = Crew(
            agents=[            
                informations_personnelle_agent,
                skills_extractor_agent,
                experience_extractor_agent,
                project_extractor_agent,
                education_extractor_agent,
                ProfileBuilderAgent       
            ],
            tasks=[
                task_extract_informations,
                task_extract_skills,
                task_extract_experience,
                task_extract_projects,
                task_extract_education,
                task_build_profile     
            ],
            process=Process.sequential,
            verbose=False,
            telemetry=False
        )
        
        result = crew.kickoff(inputs={"cv_content": cv_content})
        
        # Nettoyage
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        logger.info("Analyse CV terminée avec succès")
        return result
        
    except Exception as e:
        logger.error(f"Erreur dans analyse_cv: {e}")
        # Retour d'urgence
        return {
            "candidat": {
                "error": f"Erreur lors de l'analyse: {str(e)}",
                "informations_personnelles": {"nom": "Erreur", "email": "", "numero_de_telephone": "", "localisation": ""},
                "compétences": {"hard_skills": [], "soft_skills": []},
                "expériences": [],
                "projets": {"professional": [], "personal": []},
                "formations": []
            }
        }
