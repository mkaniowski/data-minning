import re

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from stopwords import get_stopwords
import spacy
from tqdm import tqdm

# Ensure you have downloaded the necessary NLTK dat
nltk.download('wordnet')
stop_words = set(get_stopwords("pl"))
to_add = ['w', 'i', 'z', 'ze', 'sie', 'o', 'przez', 'za', 'a', "'", "`", 's', 'fe', 'm', 'f', 'sa', 'za', 'po', 'tym', 'ktore', 'ktora', 'tym', 'ale', 'juz', 'ktory', 'oraz', 'tez', 'aby', 'moze']
stop_words.update(to_add)

nlp = spacy.load("pl_core_news_sm")

# TODO ograniczyc zbior i sprawdzac/debugowac
# TODO sprawdzic przetwarzanie danych
# TODO zbalansowane dane po 800
# /\/\/\/\/\/\/\ czy jest poprawa
# TODO stemming?
# TODO dluzszy tekst?

def preprocess_text(text):
    # lemmatizer = WordNetLemmatizer()
    # Tokenize the text
    # todo tokenizacja nltk/text blob
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [nlp(token) for token in tokens if token not in stop_words and len(token) > 1]
    return tokens


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas()
    print(stop_words)
    categories_to_transform = ['content', 'category', 'headline']

    for category in categories_to_transform:
        data[category] = data[category].astype(str)
        data[category] = data[category].str.lower()


    # TODO sprobowac bez lematyzacji i stop wrods/mapowanie polskich znakow na angielskie po preprocess_text

    data['content'] = data['content'].progress_apply(preprocess_text)

    data['content'] = data['content'].progress_apply(lambda x: x[:-1] if len(x) > 0 else x)

    data = data[data['content'].progress_apply(lambda x: len(x) >= 10)]

    data = data[~data['content'].progress_apply(lambda x: all(word in x for word in ['error', 'not', 'found']))]

    polish_to_english = str.maketrans(
        {'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z', })

    for category in categories_to_transform:
        data[category] = data[category].str.translate(polish_to_english)

    for category in categories_to_transform:
        data[category] = data[category].replace(r'[^a-zA-Z\s]', '', regex=True)

    return data
