import torch
import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class MultiModelInterviewAnalyzer:
    def __init__(self):
        """Initialisation sécurisée pour Cloud Run"""
        self.models_loaded = False
        self.sentiment_analyzer = None
        self.similarity_model = None
        self.intent_classifier = None
        
        try:
            self._load_models()
            self.models_loaded = True
            logger.info("Tous les modèles chargés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles : {e}")
            # Ne pas faire échouer l'initialisation, permettre le fallback

    def _load_models(self):
        """Chargement sécurisé des modèles"""
        # Force CPU usage - Cloud Run n'a pas de GPU
        device = -1  # Force CPU pour transformers
        
        try:
            # Sentiment analyzer avec gestion d'erreur
            self.sentiment_analyzer = pipeline(
                "text-classification",
                model="astrosbd/french_emotion_camembert",
                return_all_scores=True,
                device=device,  # Force CPU
                model_kwargs={"torch_dtype": torch.float32}  # Force float32 pour CPU
            )
            logger.info("Sentiment analyzer chargé")
        except Exception as e:
            logger.warning(f"Échec du chargement du sentiment analyzer : {e}")
            self.sentiment_analyzer = None

        try:
            # Similarity model
            self.similarity_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cpu'  # Force CPU
            )
            logger.info("Similarity model chargé")
        except Exception as e:
            logger.warning(f"Échec du chargement du similarity model : {e}")
            self.similarity_model = None

        try:
            # Intent classifier
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="joeddav/xlm-roberta-large-xnli",
                device=device,  # Force CPU
                model_kwargs={"torch_dtype": torch.float32}
            )
            logger.info("Intent classifier chargé")
        except Exception as e:
            logger.warning(f"Échec du chargement de l'intent classifier : {e}")
            self.intent_classifier = None

    def analyze_sentiment(self, messages):
        """Analyse de sentiment avec fallback"""
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        if not user_messages:
            return []
        
        if not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer non disponible, retour de données par défaut")
            return [{"label": "neutral", "score": 0.5} for _ in user_messages]
        
        try:
            sentiments = self.sentiment_analyzer(user_messages)
            return sentiments
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de sentiment : {e}")
            return [{"label": "error", "score": 0.0} for _ in user_messages]

    def compute_semantic_similarity(self, messages, job_requirements):
        """Calcul de similarité avec fallback"""
        if not self.similarity_model:
            logger.warning("Similarity model non disponible, retour de score par défaut")
            return 0.5
        
        try:
            user_answers = " ".join([msg['content'] for msg in messages if msg['role'] == 'user'])
            if not user_answers.strip():
                return 0.0
            
            embedding_answers = self.similarity_model.encode(user_answers, convert_to_tensor=True)
            embedding_requirements = self.similarity_model.encode(job_requirements, convert_to_tensor=True)
            cosine_score = util.cos_sim(embedding_answers, embedding_requirements)
            return float(cosine_score.item())
        except Exception as e:
            logger.error(f"Erreur lors du calcul de similarité : {e}")
            return 0.0

    def classify_candidate_intent(self, messages):
        """Classification d'intention avec fallback"""
        user_answers = [msg['content'] for msg in messages if msg['role'] == 'user']
        if not user_answers:
            return []
        
        if not self.intent_classifier:
            logger.warning("Intent classifier non disponible, retour de données par défaut")
            return [{"labels": ["unknown"], "scores": [0.5]} for _ in user_answers]
        
        try:
            candidate_labels = [
                "parle de son expérience technique",
                "exprime sa motivation", 
                "pose une question",
                "exprime de l'incertitude ou du stress"
            ]
            
            # Traitement batch plus efficace
            classifications = []
            for answer in user_answers:
                try:
                    result = self.intent_classifier(answer, candidate_labels, multi_label=False)
                    classifications.append(result)
                except Exception as e:
                    logger.warning(f"Erreur classification pour un message : {e}")
                    classifications.append({"labels": ["error"], "scores": [0.0]})
            
            return classifications
        except Exception as e:
            logger.error(f"Erreur lors de la classification d'intention : {e}")
            return [{"labels": ["error"], "scores": [0.0]} for _ in user_answers]

    def run_full_analysis(self, conversation_history, job_requirements):
        """Analyse complète avec gestion d'erreurs robuste"""
        try:
            # Validation des entrées
            if not conversation_history:
                conversation_history = []
            if not job_requirements:
                job_requirements = "Aucune exigence spécifiée"
            
            # Analyses avec fallback
            sentiment_results = self.analyze_sentiment(conversation_history)
            similarity_score = self.compute_semantic_similarity(conversation_history, job_requirements)
            intent_results = self.classify_candidate_intent(conversation_history)
            
            analysis_output = {
                "overall_similarity_score": round(similarity_score, 2),
                "sentiment_analysis": sentiment_results,
                "intent_analysis": intent_results,
                "raw_transcript": conversation_history,
                "models_status": {
                    "sentiment_available": self.sentiment_analyzer is not None,
                    "similarity_available": self.similarity_model is not None,
                    "intent_available": self.intent_classifier is not None,
                    "models_loaded": self.models_loaded
                }
            }
            
            return analysis_output
            
        except Exception as e:
            logger.error(f"Erreur critique dans run_full_analysis : {e}")
            # Retour d'urgence
            return {
                "overall_similarity_score": 0.0,
                "sentiment_analysis": [],
                "intent_analysis": [],
                "raw_transcript": conversation_history,
                "error": str(e),
                "models_status": {"error": True}
            }
