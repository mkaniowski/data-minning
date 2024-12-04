import pandas as pd

from src.data_processing.make_dataset import make_dataset
from src.classification.train_model import train_model
from src.classification.classify_data import classify_data
from src.data_processing.tokenize_data import tokenize_data


def pipeline_classification(configs):
    # Load data
    df = pd.read_csv("E:\\Michal\\Dokumenty\\Studia\\DM\\data-minning\\data\\cleaned_data.csv")

    for config in configs:
        tokenizer = config['tokenizer']

        model_name = config['model']

        _df = df.copy()

        _df = tokenize_data(df=_df, tokenizer=tokenizer)

        # Split the data
        X_train, X_test, y_train, y_test = make_dataset(df=_df, model_name=model_name)

        # Train the model
        if model_name in ['logistic_regression', 'svm', 'naive_bayes', 'bert']:
            model = train_model(X_train, y_train, X_test, y_test, model_name)

            classify_data(model_name, model, X_test, y_test)
        elif model_name == 'distilbert-base-uncased':
            trainer = train_model(X_train, y_train, X_test, y_test, model_name)

            classify_data(model_name, trainer, X_test, y_test)
