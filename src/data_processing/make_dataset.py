import ast
from typing import Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        seq = seq[:maxlen]  # Trim if too long
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences.astype(int)


def normalize_tokens(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def balance_dataset(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced


def make_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, maxlen: int = 5000,
                 model_name: str = "generic") -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[dict, dict, np.ndarray, np.ndarray]]:
    y = df['category']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    print("Label mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {i}")

    out = None

    if model_name == "distilbert-base-uncased":
        input_ids = pad_sequences(df['input_ids'].tolist(), 512)
        attention_mask = pad_sequences(df['attention_mask'].tolist(), 512)

        # Balance the dataset
        input_ids, y = balance_dataset(input_ids, y)
        attention_mask, _ = balance_dataset(attention_mask, y)

        ids_train, ids_test, mask_train, mask_test, y_train, y_test = train_test_split(input_ids, attention_mask, y,
                                                                                       test_size=test_size,
                                                                                       random_state=random_state)

        out = (
        {"input_ids": ids_train, "attention_mask": mask_train}, {"input_ids": ids_test, "attention_mask": mask_test},
        y_train, y_test,)

    else:
        X = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        X = pad_sequences(X.tolist(), maxlen)

        X, y = balance_dataset(X, y)
        # Split the datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                            stratify=y)


        out = X_train, X_test, y_train, y_test

    # Normalize tokens if necessary
    if model_name in ['logistic_regression', 'svm', 'naive_bayes']:
        X_train, X_test, y_train, y_test = out
        X_train, X_test = normalize_tokens(X_train, X_test)
        out = X_train, X_test, y_train, y_test

    return out
