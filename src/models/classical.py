from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocess import build_preprocessor


def build_dummy_model():
    return DummyClassifier(strategy="prior")


def build_logistic_model(train_df, target: str):
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(train_df, target)),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    l1_ratio=1,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
