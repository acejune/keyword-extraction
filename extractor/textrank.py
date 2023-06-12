import pytextrank
import spacy

import utils

LOGGER = utils.set_logger("main")


def get_keyword(doc):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    nlp_doc = nlp(doc)
    keywords = [(k.text, k.rank) for k in nlp_doc._.phrases if len(k.text.split()) <= 3]
    return keywords


def extract(*docs):
    keyword_set = set()
    for doc in docs:
        try:
            keyword_set.update(get_keyword(doc))
        except Exception as e:
            LOGGER.warning("--- 'TextRank' threw exception")
            LOGGER.warning(f"text: {doc}")
            LOGGER.exception("")

    keyword_set = utils.sort_keyword(keyword_set)
    return keyword_set
