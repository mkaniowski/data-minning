import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV

from src.classification.TextDataset import TextDataset


def classify_data(model_name, model, X_test, y_test):
    # Exclude text columns for prediction
    text_columns = ['headline', 'content', 'category']
    if isinstance(X_test, pd.DataFrame):
        X_test_for_prediction = X_test.drop(columns=text_columns)
    else:
        X_test_for_prediction = X_test

    # Check if the model was a GridSearchCV instance and print best params
    if isinstance(model, GridSearchCV):
        print("Best Parameters from GridSearchCV:")
        print(model.best_params_)
        model = model.best_estimator_  # Use the best model for predictions

    if model_name == "distilbert-base-uncased":
        test_dataset = TextDataset(X_test_for_prediction, y_test)

        # Predict on the test dataset
        predictions = model.predict(test_dataset)

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
    else:
        # Predict the labels for the test set
        y_pred = model.predict(X_test_for_prediction)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # Plot confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    # Save to csv rows that were wrongly classified
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)


