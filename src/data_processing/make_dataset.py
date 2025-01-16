import ast
from typing import Tuple, Union, Any

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
    return padded_sequences.astype(float)


def normalize_tokens(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def balance_dataset(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced


def make_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, maxlen: int = 1024,
                 model_name: str = "generic", tokenizer_name: str = "count", padding = True, normalization = True) -> \
        tuple[tuple[dict[str, Any], dict[str, Any], Any, Any], LabelEncoder] | tuple[Any, Any, Any, Any, LabelEncoder]:
    y = df['category']

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    print("Label mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {i}")

    if model_name == "dkleczek/bert-base-polish-uncased-v1":
        # Handle DistilBERT inputs
        input_ids = pad_sequences(df['input_ids'].tolist(), 512)
        attention_mask = pad_sequences(df['attention_mask'].tolist(), 512)

        # Split data with stratification
        ids_train, ids_test, mask_train, mask_test, y_train, y_test = train_test_split(input_ids, attention_mask, y,
                                                                                       test_size=test_size,
                                                                                       random_state=random_state,
                                                                                       stratify=y, )

        return (
        {"input_ids": ids_train, "attention_mask": mask_train}, {"input_ids": ids_test, "attention_mask": mask_test},
        y_train, y_test, label_encoder)

    else:
        # Handle generic models (TF-IDF, Word2Vec, etc.)
        X = df['tokens'].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(str(x)))

        if (padding):
            X = pad_sequences(X.tolist(), maxlen)
        else:
            X = np.array(X.tolist())


        # Balance dataset (optional)
        try:
            X, y = balance_dataset(X, y)
        except ValueError as e:
            print(f"SMOTE failed: {e}. Skipping balancing.")

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                            stratify=y)

        # Normalize for specific models
        if normalization:
            X_train, X_test = normalize_tokens(X_train, X_test)

        if tokenizer_name == 'word2vec':
            x_train_min = X_train.min()
            x_test_min = X_test.min()

            if x_train_min < 0:
                X_train = X_train - X_train.min()

            if x_test_min < 0:
                X_test = X_test - X_test.min()


        return X_train, X_test, y_train, y_test, label_encoder
