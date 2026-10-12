from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/ibm_hr_attrition.csv")


def load_attrition_csv(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the dataset and keep source/provenance in docs/dataset-card.md."
        )
    return pd.read_csv(path)
