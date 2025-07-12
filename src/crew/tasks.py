from crewai import Task
from .agents import report_generator_agent, skills_extractor_agent, experience_extractor_agent, project_extractor_agent, education_extractor_agent, ProfileBuilderAgent, informations_personnelle_agent

generate_report_task = Task(
   description=(
       "Tu es un rédacteur expert en RH. Ta mission est de rédiger un rapport d'évaluation final."
       "Tu ne dois PAS analyser la conversation brute toi-même. "
       "Utilise EXCLUSIVEMENT les données structurées et pré-analysées fournies dans l'input '{structured_analysis_data}'. "
       "Ces données ont été générées par des modèles de Deep Learning spécialisés et sont considérées comme la source de vérité."
   ),
   expected_output=(
       "Un rapport final exceptionnel basé sur l'analyse fournie. Le rapport doit être structuré comme suit :\n"
       "1. **Résumé et Score d'Adéquation** : Synthétise le score de similarité sémantique et donne un aperçu global.\n"
       "2. **Analyse Comportementale** : Interprète les résultats de l'analyse de sentiment et d'intention pour décrire le comportement du candidat (stress, motivation, etc.). Cite les phrases analysées.\n"
       "3. **Adéquation Sémantique avec le Poste** : Explique ce que signifie le score de similarité.\n"
       "4. **Points Forts & Axes d'Amélioration** : Utilise toutes les données pour formuler des points concrets.\n"
       "5. **Recommandation Finale**."
   ),
   agent=report_generator_agent,
)

task_extract_skills = Task(
    description=(
        "Voici le contenu du CV :\n\n{cv_content}\n\n"
        "Extraire uniquement les compétences mentionnées explicitement dans le texte du CV. "
        "Séparer les hard skills (techniques) et les soft skills (comportementales) en analysant les listes ou phrases les contenant. "
        "Les hards skills doivent comprendre des compétences techniques, outils, langages de programmation, etc. "
        "Ne rien inventer. Ne pas déduire de compétences à partir d'un poste ou d'une expérience implicite. "
        "Identifie clairement les compétences, et n'en exclue aucune. "
        "\n\n**CONTRAINTES JSON STRICTES:**\n"
        "- Utiliser UNIQUEMENT des guillemets doubles (\") pour les chaînes\n"
        "- Aucune virgule finale dans les listes ou objets\n"
        "- Vérifier la syntaxe JSON avant de retourner le résultat\n"
        "- Échapper correctement les caractères spéciaux (\\, \", \\n, etc.)"
    ),
    agent=skills_extractor_agent,
    input_keys=["cv_content"],
    expected_output=(
        "Un dictionnaire JSON VALIDE 'Compétences' avec deux clés : 'hard_skills' et 'soft_skills', "
        "contenant uniquement des listes de compétences présentes dans le texte. "
        "FORMAT EXACT: {\"hard_skills\": [\"compétence1\", \"compétence2\"], \"soft_skills\": [\"compétence1\", \"compétence2\"]}"
    )
)

task_extract_experience = Task(
    description=(
        "Voici le contenu du CV :\n\n{cv_content}\n\n"
        """
        Extrais toutes les expériences professionnelles du CV. Pour chaque expérience, tu DOIS fournir les informations suivantes :
        - Poste: Le titre du poste.
        - Entreprise: Le nom de l'entreprise.
        - start_date: La date de début. Si non trouvée, retourne "Non spécifié".
        - end_date: La date de fin. Si le poste est actuel, utilise "Aujourd'hui". Si non trouvée, retourne "Non spécifié".
        - responsabilités: Une liste des tâches et missions.

        RÈGLES STRICTES :
        1.  NE JAMAIS laisser un champ vide (""). Si une information est introuvable, utilise la valeur "Non spécifié".
        2.  Analyse attentivement les dates. "Depuis 2023" signifie que la date de fin est "Aujourd'hui".
        """
    ),
    agent=experience_extractor_agent,
    input_keys=["cv_content"],
    expected_output=(
        "Un tableau JSON VALIDE d'objets 'Expérience Professionnelle' avec 5 clés par expérience : "
        "'Poste', 'Entreprise', 'start_date', 'end_date', 'responsabilités'. "
        "FORMAT EXACT: [{\"Poste\": \"titre\", \"Entreprise\": \"nom\", \"start_date\": \"année\", \"end_date\": \"année\", \"responsabilités\": [\"resp1\", \"resp2\"]}]"
    )
)

task_extract_projects = Task(
    description=(
        "Voici le contenu du CV :\n\n{cv_content}\n\n"
        """
        Identifie et extrais les PROJETS SPÉCIFIQUES mentionnés dans le CV.
        Un projet est distinct d'une expérience professionnelle générale. Il a un nom ou un objectif clair.

        RÈGLES STRICTES :
        1.  NE PAS extraire les responsabilités générales d'un poste en tant que projet. Par exemple, si le CV dit "Alternant chez Enedis où j'ai mené le projet 'Simulateur IA'", alors extrais 'Simulateur IA' comme projet. Ne copie pas toutes les tâches de l'alternance.
        2.  Si un projet est clairement lié à une expérience professionnelle, essaie de le noter, mais le plus important est de décrire le projet lui-même.
        """
    ),
    agent=project_extractor_agent,
    input_keys=["cv_content"],
    expected_output=(
        "Un dictionnaire JSON VALIDE 'Projets' avec deux clés : 'professional' et 'personal'. "
        "Chaque clé contient une liste de dictionnaires, chaque dictionnaire représentant un projet avec les clés 'title', 'role', 'technologies', et 'outcomes'. "
        "FORMAT EXACT: {\"professional\": [{\"title\": \"titre\", \"role\": \"rôle\", \"technologies\": [\"tech1\"], \"outcomes\": [\"résultat1\"]}], \"personal\": []}"
    )
)

task_extract_education = Task(
    description=(
        "Voici le contenu du CV :\n\n{cv_content}\n\n"
        """
        Extrais le parcours de formation et les certifications. Fais une distinction claire entre les types de formation.
        Pour chaque élément, fournis :
        - degree: Le nom du diplôme, du titre (ex: 'Titre RNCP niveau 6') ou de la certification (ex: 'Core Designer Certification').
        - institution: L'école, l'université ou la plateforme (ex: 'WILD CODE SCHOOL', 'DataIku', 'DataCamp').
        - start_date: La date de début. Si non trouvée, retourne "Non spécifié".
        - end_date: La date de fin. Si non trouvée, retourne "Non spécifié".

        RÈGLES STRICTES :
        1.  Si tu vois une certification comme "DataIku (core designer)", le diplôme est "Core Designer" et l'institution est "DataIku". NE PAS les mélanger.
        2.  NE PAS extraire une simple compétence (ex: 'Python') comme une formation.
        """
    ),
    agent=education_extractor_agent,
    input_keys=["cv_content"],
    expected_output=(
        "Un tableau JSON VALIDE d'objets 'Formation' avec les clés : 'degree', 'institution', 'start_date', 'end_date'. "
        "FORMAT EXACT: [{\"degree\": \"diplôme\", \"institution\": \"établissement\", \"start_date\": \"année\", \"end_date\": \"année\"}]"
    )
)

task_extract_informations = Task(
    description=(
        "Voici le contenu du CV :\n\n{cv_content}\n\n"
        "Votre tâche est d'extraire les informations de contact du candidat. Ces informations se trouvent généralement au début ou à la fin du CV, souvent sous une section intitulée 'CONTACT'.\n"
        "Extrayez précisément :\n"
        "- Le **Nom complet**.\n"
        "- L'**Adresse e-mail**.\n"
        "- Le **Numéro de téléphone**.\n"
        "- La **Localisation** (ville ou région).\n"
        "toutes les informations devront être normalisées, principalement le nom si il est en majuscule en titre. "
    ),
    agent=informations_personnelle_agent,
    input_keys=["cv_content"],
    expected_output=(
        "Un dictionnaire JSON VALIDE 'informations_personnelles' contenant le nom, l'email, le numéro de téléphone et la localisation du candidat. "
        "FORMAT EXACT: {\"nom\": \"nom\", \"email\": \"email\", \"numero_de_telephone\": \"tel\", \"localisation\": \"lieu\"}"
    )
)


task_build_profile = Task(
    description=(
        "Ta mission est d'agir comme un architecte de données. En utilisant les extractions des tâches précédentes, "
        "assemble un profil de candidat complet. "
        "Le résultat final doit être un unique objet JSON, parfaitement valide."
    ),
    agent=ProfileBuilderAgent,
    context=[
        task_extract_informations,
        task_extract_skills,
        task_extract_experience,
        task_extract_projects,
        task_extract_education
    ],
    expected_output=(
        "Retourner un unique objet JSON valide. Cet objet doit avoir une seule clé à la racine : 'candidat'. "
        "La valeur de cette clé sera un autre objet contenant toutes les informations assemblées. "
        "Assure-toi que la syntaxe est parfaite, que tous les guillemets sont des guillemets doubles et qu'il n'y a aucune virgule finale. "
        "Le JSON doit être immédiatement parsable par un programme.\n\n"
        "FORMAT EXACT:\n"
        "{\n"
        "    \"candidat\": {\n"
        "        \"informations_personnelles\": {\"nom\": \"...\", \"email\": \"...\", ...},\n"
        "        \"compétences\": {\"hard_skills\": [...], \"soft_skills\": [...]},\n"
        "        \"expériences\": [{\"Poste\": \"...\", ...}],\n"
        "        \"projets\": {\"professional\": [...], \"personal\": [...]},\n"
        "        \"formations\": [{\"degree\": \"...\", ...}]\n"
        "    }\n"
        "}"
    ),
)