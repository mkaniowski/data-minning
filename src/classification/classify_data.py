from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def classify_data(model_name, model, X_test, y_test):
    if (model_name = "distilbert-base-uncased"):
        test_dataset = TextDataset(X_test, y_test)

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
    else:
        # Predict the labels for the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # Plot confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
