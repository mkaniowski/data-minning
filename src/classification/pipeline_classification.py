import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

from src.data_processing.make_dataset import make_dataset
from src.classification.train_model import train_model
from src.classification.classify_data import classify_data
from src.data_processing.tokenize_data import tokenize_data
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

        _df=df.copy()


        _df = tokenize_data(df=_df, tokenizer=tokenizer)

        # Split the data
        X_train, X_test, y_train, y_test = make_dataset(df=_df, model_name=model_name)

        # Normalize tokens if necessary
        if model_name in ['logistic_regression', 'svm', 'naive_bayes']:
            X_train, X_test = normalize_tokens(X_train, X_test)

        # Train the model
        if model_name in ['logistic_regression', 'svm', 'naive_bayes', 'bert']:
            model = train_model(X_train, y_train, model_name)

            # Create predictions
            classify_data(model, X_test, y_test)
        elif model_name == 'distilbert-base-uncased':
            # Ensure X_train and X_test are dictionaries with input_ids and attention_mask
            # Initialize DistilBERT tokenizer and model
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

            # Predict on the test dataset
            predictions = trainer.predict(test_dataset)

            # Extract logits and convert to predicted classes
            logits = predictions.predictions
            y_pred = np.argmax(logits, axis=1)

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)

            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(set(y_test))),
                        yticklabels=range(len(set(y_test))))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()





