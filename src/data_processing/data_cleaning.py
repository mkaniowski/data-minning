import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    categories_to_transform = ['content', 'category', 'headline']

    for category in categories_to_transform:
        data[category] = data[category].str.lower()

    polish_to_english = str.maketrans(
        {'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z', })

    for category in categories_to_transform:
        data[category] = data[category].str.translate(polish_to_english)

    for category in categories_to_transform:
        data[category] = data[category].str.replace(r'[^a-zA-Z\s]', '', regex=True)

    return data