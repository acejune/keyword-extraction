import os.path as osp
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import utils

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

LOGGER = utils.set_logger("main")


def change_to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    punct = string.punctuation
    return text.translate(str.maketrans("", "", punct))


def add_space_to_punctuation(text):
    punct_regex = r"([^\w\s])"
    text = re.sub(punct_regex, r" \1 ", text)
    return text


def remove_stopword(text):
    stopword = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stopword])


def lemmatize(text):
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    token_list = word_tokenize(text)
    pos_tagged_token_list = pos_tag(token_list)

    lemmatizer = WordNetLemmatizer()
    word_list = []
    for token, pos in pos_tagged_token_list:
        pos = get_wordnet_pos(pos)
        if pos:
            word_list.append(lemmatizer.lemmatize(token, pos=pos))
        else:
            word_list.append(lemmatizer.lemmatize(token))
    return " ".join(word_list)


def remove_html_tag(text):
    # html <sub> tag 제거
    html_regex = r"<sub>(.*?)<\/sub>"
    text = re.sub(html_regex, r"\1", text)
    return text


def get_preprocessed_paper_text(text):
    if not isinstance(text, str):
        return " "

    # html <sub> tag 제거
    text = remove_html_tag(text)

    # 화학 기호와 숫자 사이의 공백 제거
    chem_regex = r"[A-Z]+[a-z]*\s+\d+"
    text = re.sub(chem_regex, lambda x: x.group().replace(" ", ""), text)

    # 'Abstract null null'로 시작하면, 제거
    abstract_null_regex = "^Abstract null null"
    text = re.sub(abstract_null_regex, "", text)

    # 2번 이상 반복된 null 제거
    null_regex = r"\bnull\b( \bnull\b)+"
    text = re.sub(null_regex, "", text)

    # underbar 제거
    text = text.replace("_", "")

    # $ 제거
    text = text.replace("$", "")

    # 소문자로 변경
    text = change_to_lowercase(text)

    # # punctuation 제거
    # text = remove_punctuation(text)

    # stropword 제거
    text = remove_stopword(text)

    # lemmatize
    text = lemmatize(text)

    return text


def get_preprocessed_true_keyword(keyword_list):
    result = []
    if isinstance(keyword_list, list):
        for keyword in keyword_list:
            keyword = change_to_lowercase(keyword)
            keyword = remove_stopword(keyword)
            keyword = lemmatize(keyword)
            result.append(keyword)
        return result
    else:
        return None


def preprocess_data(dir_path, file_name):
    LOGGER.info("[ Data preprocessing ] Start")

    df = pd.read_csv(osp.join(dir_path, f"{file_name}.csv"))

    df["true_keyword"] = df["true_keyword"].str.split(",")

    df["ppd_true_keyword"] = None
    df["ppd_title"] = None
    df["ppd_abstract"] = None
    for row in df.itertuples():
        df.at[row.Index, "ppd_true_keyword"] = get_preprocessed_true_keyword(
            row.true_keyword
        )
        df.at[row.Index, "ppd_title"] = get_preprocessed_paper_text(row.title)
        df.at[row.Index, "ppd_abstract"] = get_preprocessed_paper_text(row.abstract)

    # Save dataframe
    save_path = osp.join(dir_path, f"{file_name}_preprocessed.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    LOGGER.info("[ Data preprocessing ] Complete")
