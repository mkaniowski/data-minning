import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_data(df: pd.DataFrame, tokenizer: str, **kwargs) -> pd.DataFrame:
    """
    Tokenize text data using specified tokenizer: 'count', 'tfidf', 'word2vec', or a transformer model like 'distilbert-base-uncased'.

    Args:
        df (pd.DataFrame): DataFrame containing a 'content' column with text data.
        tokenizer (str): Tokenizer to use ('count', 'tfidf', 'word2vec', or a HuggingFace model name).
        **kwargs: Additional keyword arguments for specific tokenizers.

    Returns:
        pd.DataFrame: DataFrame with tokenized data added as new columns.
    """
    tqdm.pandas(desc="Tokenizing")

    if 'content' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'content' column.")

    lowercase = kwargs.get('lowercase', True)

    if tokenizer == 'count':
        vectorizer = CountVectorizer(ngram_range=kwargs.get('ngram_range', (1, 2)), lowercase=lowercase,
                                     max_features=kwargs.get('max_features', 1024))
        df['content'] = df['content'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        count_matrix = vectorizer.fit_transform(df['content'])
        # Return sparse matrix directly or convert to dense if needed
        df['tokens'] = count_matrix.toarray().tolist()

    elif tokenizer == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=kwargs.get('max_features', 1024), max_df=kwargs.get('max_df', 0.95),
                                     min_df=kwargs.get('min_df', 1), ngram_range=kwargs.get('ngram_range', (1, 2)),
                                     lowercase=lowercase)
        df['content'] = df['content'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        tfidf_matrix = vectorizer.fit_transform(df['content'])
        # Return sparse matrix directly or convert to dense if needed
        df['tokens'] = tfidf_matrix.toarray().tolist()

    elif tokenizer == 'word2vec':
        # Ensure 'content' is tokenized into lists of words
        df['content'] = df['content'].apply(lambda x: x.split() if isinstance(x, str) else x)

        # Train Word2Vec model
        word2vec_model = Word2Vec(sentences=df['content'].tolist(), vector_size=kwargs.get('vector_size', 400),
            window=kwargs.get('window', 10), min_count=kwargs.get('min_count', 5),  # Include all words
            workers=kwargs.get('workers', 6), sg=kwargs.get('sg', 1),  # Skip-gram model
            epochs=kwargs.get('epochs', 15))

        # Compute TF-IDF weights for each word
        vectorizer = TfidfVectorizer(max_features=kwargs.get('max_features', 2048), lowercase=lowercase)
        df['content_str'] = df['content'].apply(lambda x: ' '.join(x))
        vectorizer.fit(df['content_str'])
        tfidf_weights = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

        # Function to get weighted Word2Vec embeddings
        def get_weighted_embeddings(tokens):
            embeddings = []
            for token in tokens:
                if token in word2vec_model.wv:
                    weight = tfidf_weights.get(token, 1.0)  # Default to 1.0 for OOV in TF-IDF
                    embeddings.append(word2vec_model.wv[token] * weight)
            if embeddings:
                return np.mean(embeddings, axis=0).tolist()
            else:
                # Return mean of all embeddings for empty cases
                return np.mean(word2vec_model.wv.vectors, axis=0).tolist()

        # Generate embeddings for each document
        df['tokens'] = df['content'].apply(get_weighted_embeddings)
        df.drop(columns=['content_str'], inplace=True)

    elif tokenizer.startswith('distilbert') or tokenizer.startswith('bert'):
        model_name = kwargs.get('model_name', tokenizer)
        auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
        df['content'] = df['content'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        try:
            tokenized = auto_tokenizer(df['content'].tolist(), truncation=True, padding=True,
                                       max_length=kwargs.get('max_length', 512), return_tensors='pt')
            # Store 'input_ids' and 'attention_mask' separately
            df['input_ids'] = tokenized['input_ids'].tolist()
            df['attention_mask'] = tokenized['attention_mask'].tolist()
            # Combine them into 'tokens' list
            df['tokens'] = df.apply(lambda row: [row['input_ids'], row['attention_mask']], axis=1)
        except Exception as e:
            # Handle tokenizer errors
            print(f"Tokenization error: {e}")
            return df

    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    return df
