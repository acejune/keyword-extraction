"""KeyBERT

https://maartengr.github.io/KeyBERT/index.html
"""


from keybert import KeyBERT

import utils
from extractor_config import configure_keybert

LOGGER = utils.set_logger("main")


def get_keyword(doc, n=1):
    config = configure_keybert()

    if config["hf_model"]:
        extract_model = KeyBERT(model=config["model_name"])
    else:
        extract_model = KeyBERT(model=config["hf_model"])

    keywords = extract_model.extract_keywords(
        docs=doc,
        stop_words=config["stop_words"],
        top_n=config["top_n"],
        use_maxsum=config["use_maxsum"],
        use_mmr=config["use_mmr"],
        keyphrase_ngram_range=(n, n),
    )
    return keywords


def extract(*docs):
    keyword_set = set()
    for doc in docs:
        for ngram_n in range(1, 4):
            try:
                keyword_set.update(get_keyword(doc, ngram_n))
            except Exception as e:
                LOGGER.warning("--- 'KeyBert n-gram' threw exception")
                LOGGER.warning(f"text: {doc}")
                LOGGER.exception("")

    keyword_set = utils.sort_keyword(keyword_set)
    return keyword_set
