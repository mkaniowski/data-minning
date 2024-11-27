import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from transformers import AutoTokenizer

nltk.download('punkt')


def tokenize_data(df: pd.DataFrame, tokenizer: str, **kwargs) -> pd.DataFrame:
    tqdm.pandas(desc="Tokenizing")

    if tokenizer == 'nltk':
        df['tokens'] = df['content'].progress_apply(word_tokenize)

    elif tokenizer == 'nltk_sent':
        df['tokens'] = df['content'].progress_apply(sent_tokenize)

    elif tokenizer == 'treebank':
        treebank_tokenizer = TreebankWordTokenizer()
        df['tokens'] = df['content'].progress_apply(treebank_tokenizer.tokenize)

    elif tokenizer == 'tfidf':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['content'])
        df['tokens'] = tfidf_matrix.toarray().tolist()

    elif tokenizer == 'count':
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(df['content'])
        df['tokens'] = count_matrix.toarray().tolist()

    elif tokenizer == 'distilbert-base-uncased':
        model_name = kwargs.get('model_name', 'distilbert-base-uncased')
        auto_tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenized = auto_tokenizer(
            df['content'].tolist(),
            truncation=True,
            padding=True,
            max_length=kwargs.get('max_length', 512),
            return_tensors='pt'
        )

        # Add tokenized fields back to the DataFrame
        df['input_ids'] = tokenized['input_ids'].tolist()
        df['attention_mask'] = tokenized['attention_mask'].tolist()

    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    return df
