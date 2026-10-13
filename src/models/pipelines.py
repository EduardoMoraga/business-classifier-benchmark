from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features.preprocess import build_preprocessor

TARGET = "Attrition"


def build_model_pipeline(
    train_x: pd.DataFrame,
    model,
    features: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    selector=None,
    memory=None,
) -> Pipeline:
    """features (optional) -> one-hot + scaling -> selector (optional) -> model."""
    steps = []
    sample = train_x
    if features is not None:
        steps.append(("features", FunctionTransformer(features)))
        sample = features(train_x)
    steps.append(("preprocess", build_preprocessor(sample.assign(**{TARGET: 0}), TARGET)))
    if selector is not None:
        steps.append(("boruta", selector))
    steps.append(("model", model))
    return Pipeline(steps, memory=memory)


def engineered_frame(model: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    """The columns the preprocessor sees: raw columns plus any engineered ones."""
    if "features" in model.named_steps:
        return model.named_steps["features"].transform(x)
    return x


def contributions(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row log-odds contribution of each transformed feature, and the feature names.

    Logistic: coefficient x transformed value. XGBoost: exact TreeSHAP (pred_contribs).
    """
    from xgboost import DMatrix, XGBClassifier

    preprocessor = model.named_steps["preprocess"]
    transformed = preprocessor.transform(engineered_frame(model, x))
    names = preprocessor.get_feature_names_out()
    estimator = model.named_steps["model"]
    if isinstance(estimator, XGBClassifier):
        matrix = DMatrix(np.asarray(transformed, dtype=float))
        return estimator.get_booster().predict(matrix, pred_contribs=True)[:, :-1], names
    return transformed * estimator.coef_[0], names
