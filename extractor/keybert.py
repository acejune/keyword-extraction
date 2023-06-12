"""KeyBERT

https://maartengr.github.io/KeyBERT/index.html
"""

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

import utils
from extractor_config import configure_keybert

LOGGER = utils.set_logger("main")


def get_keyword(doc):
    config = configure_keybert()

    if config["hf_model"]:
        extract_model = KeyBERT(model=config["model_name"])
    else:
        extract_model = KeyBERT(model=config["hf_model"])

    # pos_pattern: Noun == <N.*> / Adjective == <J.*>
    vectorizer = KeyphraseCountVectorizer(
        pos_pattern="<N.*>{1, 3} | <J.*><N.*>{1, 2} | <J.*>{2}<N.*>"
    )

    keywords = extract_model.extract_keywords(
        docs=doc,
        stop_words=config["stop_words"],
        top_n=config["top_n"],
        use_maxsum=config["use_maxsum"],
        use_mmr=config["use_mmr"],
        vectorizer=vectorizer,
    )
    return keywords


def extract(*docs):
    keyword_set = set()
    for doc in docs:
        try:
            keyword_set.update(get_keyword(doc))
        except Exception as e:
            LOGGER.warning("--- 'KeyBert' threw exception")
            LOGGER.warning(f"text: {doc}")
            LOGGER.exception("")

    keyword_set = utils.sort_keyword(keyword_set)
    return keyword_set
