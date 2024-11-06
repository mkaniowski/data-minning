import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

nltk.download('punkt')


def tokenize_data(df: pd.DataFrame, tokenizer: str) -> pd.DataFrame:
    tqdm.pandas(desc="Tokenizing")

    if (tokenizer == 'nltk'):
        df['tokens'] = df['content'].progress_apply(word_tokenize)

    elif tokenizer == 'tfidf':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['content'])
        df['tokens'] = tfidf_matrix.toarray().tolist()

    return df