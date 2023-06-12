import yake

import utils
from extractor_config import configure_yake

LOGGER = utils.set_logger("main")


def get_keyword(doc, n=1):
    config = configure_yake()
    kw_extractor = yake.KeywordExtractor(
        lan=config["lan"],
        n=n,
        dedupLim=config["dedupLim"],
        windowsSize=config["windowsSize"],
        top=config["top"],
    )

    keywords = kw_extractor.extract_keywords(doc)
    return keywords


def extract(*docs):
    keyword_set = set()
    for doc in docs:
        for ngram_n in range(1, 4):
            try:
                keyword_set.update(get_keyword(doc, ngram_n))
            except Exception as e:
                LOGGER.warning("--- 'YAKE' threw exception")
                LOGGER.warning(f"text: {doc}")
                LOGGER.exception("")

    keyword_set = utils.sort_keyword(keyword_set)
    return keyword_set
