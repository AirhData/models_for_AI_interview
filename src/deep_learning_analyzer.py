import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

class MultiModelInterviewAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="astrosbd/french_emotion_camembert",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
        )
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli"
            #device=0 if torch.cuda.is_available() else -1,
        )

    def analyze_sentiment(self, messages):
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        if not user_messages:
            return []
        sentiments = self.sentiment_analyzer(user_messages)
        return sentiments

    def compute_semantic_similarity(self, messages, job_requirements):
        user_answers = " ".join([msg['content'] for msg in messages if msg['role'] == 'user'])
        embedding_answers = self.similarity_model.encode(user_answers, convert_to_tensor=True)
        embedding_requirements = self.similarity_model.encode(job_requirements, convert_to_tensor=True)
        cosine_score = util.cos_sim(embedding_answers, embedding_requirements)
        return cosine_score.item()

    def classify_candidate_intent(self, messages):
        user_answers = [msg['content'] for msg in messages if msg['role'] == 'user']
        if not user_answers:
            return []
        candidate_labels = [
            "parle de son expérience technique",
            "exprime sa motivation",
            "pose une question",
            "exprime de l’incertitude ou du stress"
        ]
        classifications = self.intent_classifier(user_answers, candidate_labels, multi_label=False)
        return classifications

    def run_full_analysis(self, conversation_history, job_requirements):
        sentiment_results = self.analyze_sentiment(conversation_history)
        similarity_score = self.compute_semantic_similarity(conversation_history, job_requirements)
        intent_results = self.classify_candidate_intent(conversation_history)
        analysis_output = {
            "overall_similarity_score": round(similarity_score, 2),
            "sentiment_analysis": sentiment_results,
            "intent_analysis": intent_results,
            "raw_transcript": conversation_history 
        }
        return analysis_output