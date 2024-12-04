from typing import Tuple, Union
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import ast


def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        seq = seq[:maxlen]  # Trim if too long
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences.astype(int)



def make_dataset(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        maxlen: int = 5000,
        model_name: str = "generic",
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[dict, dict, np.ndarray, np.ndarray]
]:
    y = df['category']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if model_name == "distilbert-base-uncased":
        input_ids = pad_sequences(df['input_ids'].tolist(), 512)
        attention_mask = pad_sequences(df['attention_mask'].tolist(), 512)

        ids_train, ids_test, mask_train, mask_test, y_train, y_test = train_test_split(
            input_ids, attention_mask, y, test_size=test_size, random_state=random_state
        )

        return (
            {"input_ids": ids_train, "attention_mask": mask_train},
            {"input_ids": ids_test, "attention_mask": mask_test},
            y_train,
            y_test,
        )

    else:
        X = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        X = pad_sequences(X.tolist(), maxlen)

        # Split the datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test
