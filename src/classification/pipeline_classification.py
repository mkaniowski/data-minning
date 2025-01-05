import pandas as pd

from src.classification.classify_data import classify_data
from src.classification.train_model import train_model
from src.data_processing.make_dataset import make_dataset
from src.data_processing.tokenize_data import tokenize_data


def pipeline_classification(configs, return_df=False):
    # Load data
    df = pd.read_csv("E:\\Michal\\Dokumenty\\Studia\\DM\\data-minning\\data\\cleaned_data_v2.csv")

    # Optionally tokenize the content column as needed
    df['content'] = df['content'].apply(lambda x: x.split() if isinstance(x, str) else x)

    for config in configs:
        tokenizer = config['tokenizer']
        model_name = config['model']
        max_iter = config['max_iter'] if 'max_iter' in config else 1000
        padding = config['padding'] if 'padding' in config else True
        normalization = config['normalization'] if 'normalization' in config else True

        _df = df.copy()

        # Tokenize data
        _df = tokenize_data(df=_df, tokenizer=tokenizer)

        # Split the data
        X_train, X_test, y_train, y_test = make_dataset(df=_df, model_name=model_name, padding=padding,
                                                        normalization=normalization, tokenizer_name=tokenizer)

        # Train the model
        model = train_model(X_train, y_train, X_test, y_test, model_name, max_iter, tokenizer)

        # Classify data
        classify_data(model_name, model, X_test, y_test)

        if return_df:
            return _df
