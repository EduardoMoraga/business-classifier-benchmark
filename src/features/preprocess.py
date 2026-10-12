import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(df: pd.DataFrame, target: str):
    features = df.drop(columns=[target])
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in features.columns if col not in numeric]
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    numeric, categorical = split_feature_types(df, target)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
