import os

from typesafe_sdk import Noul, TypeSafeClient

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")


def classify_attrition_with_jev(state: str) -> dict:
    """Call Jev / TypeSafe using the official Python SDK."""
    if not TYPESAFE_API_KEY:
        raise RuntimeError("TYPESAFE_API_KEY is not configured")

    client = TypeSafeClient(api_key=TYPESAFE_API_KEY)
    response = client.system_one(
        state=state,
        questions={
            "attrition_risk": Noul(
                instructions="Is this employee likely to leave the company?"
            )
        },
    )
    answer = response.answers["attrition_risk"]
    return {"attrition_risk": getattr(answer, "noul", None)}
