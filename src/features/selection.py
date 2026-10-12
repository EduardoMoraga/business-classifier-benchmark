import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def numeric_correlation_report(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    corr = df.select_dtypes(include=["number", "bool"]).corr().abs()
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if value >= threshold:
                pairs.append({"feature_a": left, "feature_b": right, "abs_corr": value})
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False)


def vif_report(x_numeric: pd.DataFrame) -> pd.DataFrame:
    clean = x_numeric.dropna().astype(float)
    return pd.DataFrame(
        {
            "feature": clean.columns,
            "vif": [variance_inflation_factor(clean.values, i) for i in range(clean.shape[1])],
        }
    ).sort_values("vif", ascending=False)
