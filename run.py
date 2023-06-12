import argparse
import ast
import os.path as osp
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import extractor.keybert as keybert
import extractor.keybert_ngram as keybert_ngram
import extractor.rake as rake
import extractor.textrank as textrank
import extractor.tfidf as tfidf
import extractor.yake as yake
import preprocess
import utils
from evaluation import calculate_precision_recall_f1


def calculate_eval_score(keyword_score):
    keyword = [k for k, _ in keyword_score]
    return calculate_precision_recall_f1(keyword, ppd_true_keyword)


LOGGER = utils.set_logger("main")
LOGGER.info("=========================================")
LOGGER.info("        Keyword Extraction Start!        ")
LOGGER.info("=========================================")

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", required=True)
parser.add_argument("--load_fn", required=True)
parser.add_argument("--method", nargs="+", required=True)
args = parser.parse_args()

# Preprocss raw text
preprocess.preprocess_data(args.dir_path, args.load_fn)

# Load data
ppd_file_name = f"{args.load_fn}_preprocessed"
df = pd.read_csv(osp.join(args.dir_path, f"{ppd_file_name}.csv"))

df["ppd_true_keyword"] = df["ppd_true_keyword"].astype(str).values.tolist()

# Create columns
for col_name in args.method:
    df[f"kwrd_{col_name}"] = None

# Score
eval_dict = defaultdict(list)

# Extract keyword
for row in tqdm(df.itertuples(), total=df.shape[0]):
    idx = row.Index
    title = row.ppd_title
    abstract = row.ppd_abstract
    title_abstract = f"{title}. {abstract}"
    ppd_true_keyword = ast.literal_eval(row.ppd_true_keyword)

    if "keybert_ngram" in args.method:
        keyword_score = keybert_ngram.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_keybert_ngram"] = keyword_score
        eval_dict["keybert_ngram"].append(calculate_eval_score(keyword_score))

    if "keybert" in args.method:
        keyword_score = keybert.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_keybert"] = keyword_score
        eval_dict["keybert"].append(calculate_eval_score(keyword_score))

    if "rake" in args.method:
        keyword_score = rake.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_rake"] = keyword_score
        eval_dict["rake"].append(calculate_eval_score(keyword_score))

    if "textrank" in args.method:
        keyword_score = textrank.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_textrank"] = keyword_score
        eval_dict["textrank"].append(calculate_eval_score(keyword_score))

    if "tfidf" in args.method:
        keyword_score = tfidf.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_tfidf"] = keyword_score
        eval_dict["tfidf"].append(calculate_eval_score(keyword_score))

    if "yake" in args.method:
        keyword_score = yake.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_yake"] = keyword_score
        eval_dict["yake"].append(calculate_eval_score(keyword_score))

eval_score_mean_dict = {}
for key, val in eval_dict.items():
    eval_score_mean_dict[key] = [sum(x) / len(x) for x in zip(*val)]

# Evaluation score Dataframe
eval_df = pd.DataFrame.from_dict(data=eval_dict)
eval_mean_df = pd.DataFrame.from_dict(data=eval_score_mean_dict)

# Save dataframe
save_file_name = f"{args.load_fn}_complete.xlsx"
save_file_path = osp.join(args.dir_path, save_file_name)
with pd.ExcelWriter(save_file_path) as writer:
    df.to_excel(writer, index=False, sheet_name="keywords")
    eval_df.to_excel(writer, index=False, sheet_name="eval")
    eval_mean_df.to_excel(writer, index=False, sheet_name="avg. eval")

LOGGER.info("=========================================")
LOGGER.info("        Keyword Extraction End!        ")
LOGGER.info("=========================================")
