import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_data(df: pd.DataFrame, tokenizer: str, **kwargs) -> pd.DataFrame:
    tqdm.pandas(desc="Tokenizing")

    if tokenizer == 'count':
        vectorizer = CountVectorizer()
        df['content'] = df['content'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        count_matrix = vectorizer.fit_transform(df['content'])
        df['tokens'] = count_matrix.toarray().tolist()

    elif tokenizer == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=kwargs.get('max_features', 10000),
                                     max_df=kwargs.get('max_df', 0.95),
                                     min_df=kwargs.get('min_df', 1))

        df['content'] = df['content'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        tfidf_matrix = vectorizer.fit_transform(df['content'])
        df['tokens'] = tfidf_matrix.toarray().tolist()

    elif tokenizer == 'word2vec':
        # Ensure the content is tokenized into lists of words
        df['content'] = df['content'].apply(lambda x: x if isinstance(x, list) else x.split())

        # Train Word2Vec model
        word2vec_model = Word2Vec(sentences=df['content'].tolist(), vector_size=kwargs.get('vector_size', 100),
                                  window=kwargs.get('window', 5), min_count=kwargs.get('min_count', 1),
                                  workers=kwargs.get('workers', 4))

        # Generate average Word2Vec embeddings for each document
        def get_word2vec_embeddings(tokens):
            embeddings = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
            if embeddings:
                return sum(embeddings) / len(embeddings)
            else:
                return [0] * word2vec_model.vector_size

        df['tokens'] = df['content'].apply(get_word2vec_embeddings)

    elif tokenizer == 'distilbert-base-uncased':
        model_name = kwargs.get('model_name', 'distilbert-base-uncased')
        auto_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Join tokens back into a single string
        df['content'] = df['content'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

        tokenized = auto_tokenizer(df['content'].tolist(), truncation=True, padding=True,
                                   max_length=kwargs.get('max_length', 512), return_tensors='pt')

        # Add tokenized fields back to the DataFrame
        df['input_ids'] = tokenized['input_ids'].tolist()
        df['attention_mask'] = tokenized['attention_mask'].tolist()

    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    return df
