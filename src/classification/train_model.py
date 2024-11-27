import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, \
    DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch


def train_model(X_train, y_train, model_name):
    if model_name == 'logistic_regression':
        model = LogisticRegression(random_state=42, verbose=True, max_iter=1000)
        model.fit(X_train, y_train)
    elif model_name == 'svm':
        model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
    elif model_name == 'naive_bayes':
        model = MultinomialNB()
        model.fit(X_train, y_train)
    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
        train_dataset = torch.utils.data.Dataset(train_encodings, y_train)
        training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
    elif model_name == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
        train_dataset = torch.utils.data.Dataset(train_encodings, y_train)
        training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model