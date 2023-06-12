"""RAKE (Rapid Automatic Keyword Extraction)"""


from rake_nltk import Rake

import utils

LOGGER = utils.set_logger("main")


def get_keyword(doc):
    rake = Rake()
    rake.extract_keywords_from_text(doc)
    keywords = []
    for score, keyword in rake.get_ranked_phrases_with_scores():
        if len(keyword.split()) <= 3:
            keywords.append((keyword, score))
    return keywords


def extract(*docs):
    keyword_set = set()
    for doc in docs:
        try:
            keyword_set.update(get_keyword(doc))
        except Exception as e:
            LOGGER.warning("--- 'RAKE' threw exception")
            LOGGER.warning(f"text: {doc}")
            LOGGER.exception("")

    keyword_set = utils.sort_keyword(keyword_set)
    return keyword_set
