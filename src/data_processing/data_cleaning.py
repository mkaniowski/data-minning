import re

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from stopwords import get_stopwords

# Ensure you have downloaded the necessary NLTK dat
nltk.download('wordnet')
stop_words = set(get_stopwords("pl"))
to_add = ['w', 'i', 'z', 'ze', 'sie', 'o', 'przez', 'za', 'a', "'", "`", 's', 'fe', 'm', 'f', 'sa', 'za', 'po', 'tym', 'ktore', 'ktora', 'tym', 'ale', 'juz', 'ktory', 'oraz', 'tez', 'aby', 'moze']
stop_words.update(to_add)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    # Tokenize the text
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]
    return tokens


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    print(stop_words)
    categories_to_transform = ['content', 'category', 'headline']

    for category in categories_to_transform:
        data[category] = data[category].astype(str)
        data[category] = data[category].str.lower()

    polish_to_english = str.maketrans(
        {'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z', })

    for category in categories_to_transform:
        data[category] = data[category].str.translate(polish_to_english)

    for category in categories_to_transform:
        data[category] = data[category].str.replace(r'[^a-zA-Z\s]', '', regex=True)

    data['content'] = data['content'].apply(preprocess_text)

    data['content'] = data['content'].apply(lambda x: x[:-1] if len(x) > 0 else x)

    data = data[data['content'].apply(lambda x: len(x) >= 10)]

    data = data[~data['content'].apply(lambda x: all(word in x for word in ['error', 'not', 'found']))]

    return data
