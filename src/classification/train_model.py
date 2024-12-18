import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification, \
    AutoTokenizer
from transformers import Trainer, TrainingArguments

from src.classification.TextDataset import TextDataset


def train_model(X_train, y_train, X_test, y_test, model_name):
    # Exclude text columns
    text_columns = ['headline', 'content', 'category']
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.drop(columns=text_columns)
        X_test = X_test.drop(columns=text_columns)

    if model_name == 'logistic_regression':
        params = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear'], 'class_weight': ['balanced']}
        model = LogisticRegression(max_iter=750)
        grid = GridSearchCV(model, params, scoring='accuracy', cv=5, verbose=1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    elif model_name == 'svm':
        # params = {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'class_weight': ['balanced'], 'gamma': ['scale', 'auto']}
        model = SVC(max_iter=750, random_state=42)
        # grid = GridSearchCV(model, params, scoring='accuracy', cv=5)
        model.fit(X_train, y_train)
        # model = grid.best_estimator_
    elif model_name == 'naive_bayes':
        model = MultinomialNB()
        model.fit(X_train, y_train)
    elif model_name == 'random_forest':
        params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        model = RandomForestClassifier()
        grid = GridSearchCV(model, params, scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
        train_dataset = TextDataset(train_encodings, y_train)
        training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
    elif model_name == 'distilbert-base-uncased':
        print(torch.cuda.is_available())  # Should return True
        print(torch.cuda.device_count())  # Number of GPUs available
        print(torch.cuda.current_device())  # Currently active GPU ID
        print(torch.cuda.get_device_name(0))  # Name of the GPU


        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                    num_labels=len(set(y_train)))

        # Wrap the datasets using the TextDataset class
        train_dataset = TextDataset(X_train, y_train)
        test_dataset = TextDataset(X_test, y_test)

        # Set up training arguments
        training_args = TrainingArguments(output_dir='./results',  # Directory to save results
                                          num_train_epochs=3,  # Number of epochs
                                          per_device_train_batch_size=32,  # Batch size
                                          per_device_eval_batch_size=32, warmup_steps=500,
                                          # Warmup steps for learning rate scheduler
                                          weight_decay=0.01,  # Weight decay
                                          logging_dir='./logs',  # Directory to save logs
                                          logging_steps=10, evaluation_strategy="epoch", save_strategy="epoch",
                                          fp16=True,  # Use mixed precision training
                                          gradient_accumulation_steps=2)

        # Initialize the Trainer
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)

        if torch.cuda.is_available():
            model.cuda()

        device = next(model.parameters()).device
        print(f"Model is running on: {device}")

        trainer.train()
        return trainer
        raise ValueError(f"Unknown model_name: {model_name}")

    return model
