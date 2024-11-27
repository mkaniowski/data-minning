import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_processing.make_dataset import make_dataset
from src.classification.train_model import train_model
from src.classification.classify_data import classify_data
from src.data_processing.tokenize_data import tokenize_data
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def normalize_tokens(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def pipeline_classification(configs):
    # Load data
    df = pd.read_csv("E:\\Michal\\Dokumenty\\Studia\\DM\\data-minning\\data\\cleaned_data.csv")

    for config in configs:
        tokenizer = config['tokenizer']
        model_name = config['model']

        _df=df

        # Tokenize the data
        if model_name in ['logistic_regression', 'svm', 'naive_bayes']:
            _df = tokenize_data(df=df, tokenizer=tokenizer)

        # Split the data
        X_train, X_test, y_train, y_test = make_dataset(df=_df, model_name=model_name)

        # Normalize tokens if necessary
        if model_name in ['logistic_regression', 'svm', 'naive_bayes']:
            X_train, X_test = normalize_tokens(X_train, X_test)

        # Train the model
        if model_name in ['logistic_regression', 'svm', 'naive_bayes', 'bert']:
            model = train_model(X_train, y_train, model_name)
        elif model_name == 'distilbert-base-uncased':
            # Load the tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = DistilBertForSequenceClassification.from_pretrained(model_name)

            # Tokenize the input text
            train_encodings = tokenizer(list(df.loc[X_train.index, 'content']), truncation=True, padding=True,
                                        max_length=512)
            test_encodings = tokenizer(list(df.loc[X_test.index, 'content']), truncation=True, padding=True,
                                       max_length=512)

            # Convert to torch Dataset
            train_dataset = TextDataset(train_encodings, y_train.tolist())
            test_dataset = TextDataset(test_encodings, y_test.tolist())

            # Check dataset lengths
            assert len(train_dataset) == len(y_train), "Mismatch in train dataset length"
            assert len(test_dataset) == len(y_test), "Mismatch in test dataset length"

            # Define training arguments
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
            )

            # Initialize the Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )

            # Train the model
            trainer.train()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Classify the data
        classify_data(model, X_test, y_test)