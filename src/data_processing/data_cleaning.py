import nltk
import pandas as pd
import spacy
from stopwords import get_stopwords
from tqdm import tqdm

# Ensure necessary NLTK data is downloaded
nltk.download("wordnet")

# Load SpaCy model for Polish
nlp = spacy.load("pl_core_news_sm", disable=['ner', 'parser'])

# Define custom stop words
stop_words = set(get_stopwords("pl"))
to_add = ['w', 'i', 'z', 'ze', 'sie', 'o', 'przez', 'za', 'a', "'", "`", 's', 'fe', 'm', 'f', 'sa', 'za', 'po', 'tym',
          'ktore', 'ktora', 'tym', 'ale', 'juz', 'ktory', 'oraz', 'tez', 'aby', 'moze', 'który', 'byc', 'ktory',
          'która', 'które', 'być', 'co', 'na', 'to', 'jak', 'do', 'rok', 'które', 'miec', 'mieć']
stop_words.update(to_add)

# Polish to English character mapping
polish_to_english = str.maketrans(
    {'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z', })


def preprocess_text(texts):
    """
    Preprocess a list of text entries in batches.
    :param texts: list of str, input texts
    :return: List of preprocessed texts
    """
    docs = list(nlp.pipe(texts, batch_size=50, n_process=8))
    cleaned_texts = []
    for doc in docs:
        tokens = []
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma not in stop_words and len(token.text) > 1:
                tokens.append(lemma)
        cleaned_texts.append(' '.join(tokens))

    return cleaned_texts


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess text data in a DataFrame for text classification.
    :param data: pd.DataFrame, input DataFrame
    :return: pd.DataFrame, cleaned DataFrame
    """

    print(stop_words)

    tqdm.pandas()

    # Columns to process
    categories_to_transform = ['content', 'category', 'headline']

    data.dropna(inplace=True)

    # Preprocess content column in batches
    texts = data['content'].tolist()
    data['content'] = preprocess_text(texts)

    # Remove empty or very short content entries
    data = data[data['content'].progress_apply(lambda x: len(x) >= 10)]

    # Remove rows where content includes certain error-like patterns
    data = data[~data['content'].progress_apply(lambda x: all(word in x for word in ['error', 'not', 'found']))]

    # Polish to English mapping and clean other columns
    for category in categories_to_transform:
        data[category] = data[category].str.translate(polish_to_english)
        data[category] = data[category].replace(r'[^A-Za-z0-9ĄąĆćĘęŁłŃńÓóŚśŹźŻż\s]', '', regex=True)

    # Limit conent column to 200 characters
    data['content'] = data['content'].str.slice(0, 200)

    return data
