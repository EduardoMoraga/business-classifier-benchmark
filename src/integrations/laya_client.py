import os

import requests

LAYA_API_KEY = os.getenv("LAYA_API_KEY")


def classify_attrition_with_laya(state: str) -> dict:
    if not LAYA_API_KEY:
        raise RuntimeError("LAYA_API_KEY is not configured")

    payload = {
        "state": state,
        "questions": {
            "attrition_risk": {
                "type": "noul",
                "instructions": "Is this employee likely to leave the company?"
            }
        },
    }

    response = requests.post(
        "https://api.laya-ai.com/v1/systemone",
        headers={
            "Authorization": f"Bearer {LAYA_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
