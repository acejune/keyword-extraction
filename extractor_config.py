from transformers.pipelines import pipeline

TOP_N = 10


def configure_keybert(hf_model_name=None):
    config = {}
    if not hf_model_name:
        config["model_name"] = "all-MiniLM-L6-v2"
        config["hf_model"] = None
    else:
        config["model_name"] = hf_model_name  # --> https://huggingface.co/models
        config["hf_model"] = pipeline(
            model=config["model_name"], task="feature-extraction"
        )
    config["top_n"] = TOP_N
    config["use_maxsum"] = False  # diversification parameter
    # config["nr_candidates"]
    config["use_mmr"] = False  # diversification parameter
    # config["diversity"]
    config["stop_words"] = None
    return config


def configure_tfidf():
    config = {}
    config["top_n"] = TOP_N
    return config


def configure_yake():
    config = {}
    config["lan"] = "en"  # text language
    config["dedupLim"] = 0.8  # deduplication threshold
    config["windowsSize"] = 4  # number of keywords
    config["top"] = TOP_N
    return config
