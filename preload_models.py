import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preload_transformers_models():
    """Pré-charge les modèles Transformers"""
    try:
        from transformers import pipeline, AutoTokenizer, AutoModel
        
        logger.info("=== Pré-chargement des modèles Transformers ===")
        models_to_preload = [
            {
                "name": "astrosbd/french_emotion_camembert",
                "task": "text-classification",
                "description": "Sentiment analysis français"
            },
            {
                "name": "joeddav/xlm-roberta-large-xnli", 
                "task": "zero-shot-classification",
                "description": "Classification zero-shot multilingue"
            }
        ]
        
        for model_info in models_to_preload:
            try:
                logger.info(f"Téléchargement: {model_info['name']} ({model_info['description']})")
                tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
                model = AutoModel.from_pretrained(model_info["name"])
                pipe = pipeline(model_info["task"], model=model_info["name"], device=-1)
                logger.info(f"✅ {model_info['name']} pré-chargé avec succès")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors du pré-chargement de {model_info['name']}: {e}")
                continue
                
    except ImportError as e:
        logger.error(f"Transformers non disponible: {e}")
        return False
    
    return True

def preload_sentence_transformers():
    """Pré-charge les modèles Sentence Transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("=== Pré-chargement Sentence Transformers ===")
        model_name = 'all-MiniLM-L6-v2'
        logger.info(f"Téléchargement: {model_name}")
        model = SentenceTransformer(model_name)
        test_sentence = "Test de fonctionnement"
        embedding = model.encode(test_sentence)
        logger.info(f"✅ {model_name} pré-chargé avec succès (embedding shape: {embedding.shape})")
        return True
        
    except ImportError as e:
        logger.error(f"Sentence Transformers non disponible: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du pré-chargement Sentence Transformers: {e}")
        return False

def preload_torch():
    """Pré-charge PyTorch et vérifie la configuration"""
    try:
        import torch
        
        logger.info("=== Configuration PyTorch ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA disponible: {torch.cuda.is_available()}")
        logger.info(f"CPU threads: {torch.get_num_threads()}")
        test_tensor = torch.randn(10, 10)
        logger.info("✅ PyTorch configuré correctement")
        
        return True
        
    except ImportError as e:
        logger.error(f"PyTorch non disponible: {e}")
        return False

def verify_model_cache():
    """Vérifie que les modèles sont bien en cache"""
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/app/model_cache')
    cache_path = Path(cache_dir)
    
    if cache_path.exists():
        cached_models = list(cache_path.rglob("*"))
        logger.info(f"📁 Cache modèles: {len(cached_models)} fichiers dans {cache_dir}")
        model_dirs = [d for d in cache_path.iterdir() if d.is_dir()]
        for model_dir in model_dirs[:5]: 
            logger.info(f"  - {model_dir.name}")
            
        return len(cached_models) > 0
    else:
        logger.warning(f"❌ Répertoire de cache non trouvé: {cache_dir}")
        return False

def main():
    """Fonction principale de pré-chargement"""
    logger.info("🚀 Début du pré-chargement des modèles ML")
    
    success_count = 0
    total_steps = 4
    steps = [
        ("PyTorch", preload_torch),
        ("Transformers", preload_transformers_models), 
        ("Sentence Transformers", preload_sentence_transformers),
        ("Vérification cache", verify_model_cache)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n--- {step_name} ---")
        try:
            if step_func():
                success_count += 1
                logger.info(f"✅ {step_name} terminé avec succès")
            else:
                logger.warning(f"⚠️ {step_name} terminé avec des avertissements")
        except Exception as e:
            logger.error(f"❌ Erreur critique dans {step_name}: {e}")
    logger.info(f"\n🎯 Pré-chargement terminé: {success_count}/{total_steps} étapes réussies")
    
    if success_count >= 2:  
        logger.info("✅ Pré-chargement suffisant pour fonctionner")
        return 0
    else:
        logger.error("❌ Pré-chargement insuffisant - risque de problèmes au runtime")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
