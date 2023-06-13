# keyword-extraction

Based on the title and abstract of the paper, try to find the keywords and research fields of the paper.

## Dependencies

```
pip install -r requirements.txt
```

or

```
conda env create -f environment.yaml
```

## Usage

```
python run.py --dir_path _dataset --load_fn sample_data --method keybert_ngram keybert rake textrank tfidf yake --eval
```

or

```
python run.py --dir_path _dataset --load_fn sample_data --method keybert_ngram keybert rake textrank tfidf yake --no-eval
```

## Issue

- text 전처리에서, stopword 제거할 때, 화학 원소 `In` (Indium)이 제거됨
