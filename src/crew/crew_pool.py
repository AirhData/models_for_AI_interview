from crewai import Crew, Process
from langchain_core.tools import tool
import json
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Type
from .agents import report_generator_agent, skills_extractor_agent, experience_extractor_agent, project_extractor_agent, education_extractor_agent, ProfileBuilderAgent, informations_personnelle_agent
from .tasks import generate_report_task, task_extract_skills, task_extract_experience, task_extract_projects, task_extract_education, task_build_profile, task_extract_informations
from src.deep_learning_analyzer import MultiModelInterviewAnalyzer 
from langchain_core.tools import BaseTool

@tool
def interview_analyser(conversation_history: list, job_description_text: list) -> str:
    """
    Appelle cet outil à la toute fin d'un entretien d'embauche pour analyser
    l'intégralité de la conversation et générer un rapport de feedback.
    Ne l'utilise PAS pour répondre à une question normale, mais seulement pour conclure et analyser l'entretien.
    """
    interview_crew= Crew(
        agents=[report_generator_agent],
        tasks=[generate_report_task],
        process=Process.sequential,
        verbose=False,
        telemetry=False
    )
    analyzer = MultiModelInterviewAnalyzer()
    structured_analysis = analyzer.run_full_analysis(conversation_history, job_description_text)
    final_report = interview_crew.kickoff(inputs={
    'structured_analysis_data': json.dumps(structured_analysis, indent=2)
    })
    return final_report

'''
class EmptyInput(BaseModel):
    pass

class InterviewAnalysisTool(BaseTool):
    """
    Appelle cet outil à la toute fin d'un entretien d'embauche pour analyser
    l'intégralité de la conversation et générer un rapport de feedback.
    Ne l'utilise PAS pour répondre à une question normale, mais seulement pour conclure et analyser l'entretien.
    """
    name: str = "interview_analyser"
    description: str = (
        "Appelle cet outil à la toute fin d'un entretien d'embauche pour analyser "
        "l'intégralité de la conversation et générer un rapport de feedback. "
        "Ne l'utilise PAS pour répondre à une question normale, mais seulement pour conclure et analyser l'entretien."
    )
    args_schema: type[BaseModel] = EmptyInput
    job_offer: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]

    def _run(self) -> str:
        """Exécute l'analyse de l'entretien."""
        interview_crew = Crew(
            agents=[report_generator_agent],
            tasks=[generate_report_task],
            process=Process.sequential,
            verbose=False,
            telemetry=False
        )
        analyzer = MultiModelInterviewAnalyzer()
        structured_analysis = analyzer.run_full_analysis(self.conversation_history, self.job_offer)

        final_report = interview_crew.kickoff(inputs={
            'structured_analysis_data': json.dumps(structured_analysis, indent=2)
        })
        return final_report
'''
def analyse_cv(cv_content: str) -> json:
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
    return result
 
    