from sklearn.feature_extraction.text import TfidfVectorizer

import utils
from extractor_config import configure_tfidf

LOGGER = utils.set_logger("main")


def get_keyword(doc, word_list, vectorizer):
    tf_idf_vector = vectorizer.transform([doc])

    # Sort with highest score
    coo_matrix = tf_idf_vector.tocoo()
    tuples = zip(coo_matrix.col, coo_matrix.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    # Get word & tf-idf score
    config = configure_tfidf()
    sorted_items = sorted_items[: config["top_n"]]
    output = {}
    for idx, score in sorted_items:
        output[word_list[idx]] = score

    return list(output.items())


def extract(*docs):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=(1, 3))
    vectorizer.fit_transform([docs[-1]])
    word_list = vectorizer.get_feature_names_out()

    keyword_set = set()
    for doc in docs:
        try:
            keyword_set.update(get_keyword(doc, word_list, vectorizer))
        except Exception as e:
            LOGGER.warning("--- 'TF-IDF' threw exception")
            LOGGER.warning(f"text: {doc}")
            LOGGER.exception("")

    keyword_set = utils.sort_keyword(keyword_set)
    return keyword_set
