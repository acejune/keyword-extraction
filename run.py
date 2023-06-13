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
LOGGER.info("==========================================")
LOGGER.info("                  Start!                  ")
LOGGER.info("==========================================")

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", required=True)
parser.add_argument("--load_fn", required=True)
parser.add_argument("--method", nargs="+", required=True)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--no-eval", dest="eval", action="store_false")
args = parser.parse_args()
LOGGER.info(f"파일 이름: {args.load_fn}.csv")
LOGGER.info(f"사용할 Extractor: {args.method}")

# Preprocss raw text
preprocess.preprocess_data(args.dir_path, args.load_fn)

# Load data
ppd_file_name = f"{args.load_fn}_preprocessed"
df = pd.read_csv(osp.join(args.dir_path, f"{ppd_file_name}.csv"))

if args.eval:
    df = df[df["ppd_true_keyword"].notna()]  # Test set
    df["ppd_true_keyword"] = df["ppd_true_keyword"].astype(str).values.tolist()
    eval_dict = defaultdict(list)

# Create columns
for col_name in args.method:
    df[f"kwrd_{col_name}"] = None

LOGGER.info(f"사용할 논문 수: {df.shape[0]}")

# Extract keyword
for row in tqdm(df.itertuples(), total=df.shape[0]):
    idx = row.Index
    title = row.ppd_title
    abstract = row.ppd_abstract
    title_abstract = f"{title}. {abstract}"
    if args.eval:
        ppd_true_keyword = ast.literal_eval(row.ppd_true_keyword)

    if "keybert_ngram" in args.method:
        keyword_score = keybert_ngram.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_keybert_ngram"] = keyword_score
        if args.eval:
            eval_dict["keybert_ngram"].append(calculate_eval_score(keyword_score))

    if "keybert" in args.method:
        keyword_score = keybert.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_keybert"] = keyword_score
        if args.eval:
            eval_dict["keybert"].append(calculate_eval_score(keyword_score))

    if "rake" in args.method:
        keyword_score = rake.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_rake"] = keyword_score
        if args.eval:
            eval_dict["rake"].append(calculate_eval_score(keyword_score))

    if "textrank" in args.method:
        keyword_score = textrank.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_textrank"] = keyword_score
        if args.eval:
            eval_dict["textrank"].append(calculate_eval_score(keyword_score))

    if "tfidf" in args.method:
        keyword_score = tfidf.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_tfidf"] = keyword_score
        if args.eval:
            eval_dict["tfidf"].append(calculate_eval_score(keyword_score))

    if "yake" in args.method:
        keyword_score = yake.extract(title, abstract, title_abstract)
        df.at[idx, f"kwrd_yake"] = keyword_score
        if args.eval:
            eval_dict["yake"].append(calculate_eval_score(keyword_score))

if args.eval:
    eval_score_mean_dict = {}
    for key, val in eval_dict.items():
        eval_score_mean_dict[key] = [sum(x) / len(x) for x in zip(*val)]

    # Evaluation score Dataframe
    eval_df = pd.DataFrame.from_dict(data=eval_dict)
    eval_mean_df = pd.DataFrame.from_dict(data=eval_score_mean_dict)

# Save dataframe
save_file_name = f"{args.load_fn}_complete.xlsx"
save_file_path = osp.join(args.dir_path, save_file_name)
if args.eval:
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keywords")
        eval_df.to_excel(writer, index=False, sheet_name="eval")
        eval_mean_df.to_excel(writer, index=False, sheet_name="avg. eval")
else:
    with pd.ExcelWriter(save_file_path) as writer:
        df.to_excel(writer, index=False, sheet_name="keywords")

LOGGER.info("==========================================")
LOGGER.info("                   End!                   ")
LOGGER.info("==========================================")
