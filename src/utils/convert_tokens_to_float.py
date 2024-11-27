import pandas as pd
import ast

def convert_tokens_to_float(df: pd.DataFrame) -> pd.DataFrame:
    df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df
