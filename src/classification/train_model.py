import optuna
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification, \
    AutoTokenizer
from transformers import Trainer, TrainingArguments

from src.classification.TextDataset import TextDataset


def train_model(X_train, y_train, X_test, y_test, model_name, max_iter, tokenizer):
    # Exclude text columns
    text_columns = ['headline', 'content', 'category']
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.drop(columns=text_columns)
        X_test = X_test.drop(columns=text_columns)

    # PostgreSQL Optuna storage
    storage_url = "postgresql+psycopg2://studies:studies@localhost:5432/studies"

    if model_name == 'logistic_regression':
        def objective(trial):
            params = {'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']), }
            model = LogisticRegression(max_iter=max_iter, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)

        study = optuna.create_study(storage=storage_url, study_name=f"{model_name}_{tokenizer}", direction='maximize',
                                    load_if_exists=True)
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        model = LogisticRegression(max_iter=max_iter, **best_params)
        model.fit(X_train, y_train)

    elif model_name == 'svm':
        def objective(trial):
            params = {'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']), }
            model = SVC(max_iter=max_iter, random_state=42, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)

        study = optuna.create_study(storage=storage_url, study_name=f"{model_name}_{tokenizer}", direction='maximize',
                                    load_if_exists=True)
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        model = SVC(max_iter=max_iter, random_state=42, **best_params)
        model.fit(X_train, y_train)

    elif model_name == 'naive_bayes':
        model = MultinomialNB()
        model.fit(X_train, y_train)

    elif model_name == 'random_forest':
        def objective(trial):
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)

        study = optuna.create_study(storage=storage_url, study_name=f"{model_name}_{tokenizer}", direction='maximize',
                                    load_if_exists=True)
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)

    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
        train_dataset = TextDataset(train_encodings, y_train)
        training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()

    elif model_name == 'distilbert-base-uncased':
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                    num_labels=len(set(y_train)))
        train_dataset = TextDataset(X_train, y_train)
        test_dataset = TextDataset(X_test, y_test)

        training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=32,
                                          per_device_eval_batch_size=32, warmup_steps=500, weight_decay=0.01,
                                          logging_dir='./logs', logging_steps=10, evaluation_strategy="epoch",
                                          save_strategy="epoch", fp16=True, gradient_accumulation_steps=2)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
        trainer.train()

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model
