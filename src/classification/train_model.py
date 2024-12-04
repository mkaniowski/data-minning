import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, \
    DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch


def train_model(X_train, y_train, X_test, y_test, model_name):
    if model_name == 'logistic_regression':
        model = LogisticRegression(random_state=42, verbose=True, max_iter=1000)
        model.fit(X_train, y_train)
    elif model_name == 'svm':
        model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
    elif model_name == 'naive_bayes':
        model = MultinomialNB()
        model.fit(X_train, y_train)
    elif model_name == 'random_forest':
        model = RandomForestClassifier()
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
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(set(y_train))
        )

        # Wrap the datasets using the TextDataset class
        train_dataset = TextDataset(X_train, y_train)
        test_dataset = TextDataset(X_test, y_test)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./results',  # Directory to save results
            num_train_epochs=3,  # Number of epochs
            per_device_train_batch_size=16,  # Batch size
            per_device_eval_batch_size=16,
            warmup_steps=500,  # Warmup steps for learning rate scheduler
            weight_decay=0.01,  # Weight decay
            logging_dir='./logs',  # Directory to save logs
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch"
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        trainer.evaluate()

        # Save model
        model.save_pretrained(f"./results/{model_name}")
        trainer.save_model('./model')

        return trainer
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model