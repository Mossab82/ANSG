"""Legacy ArabicNLP client example; network access is opt-in.

This helper is retained for research provenance, not required by ANSG training.
Do not store tokens in source control.
"""
import os
import requests


def analyze_text(text: str, endpoint: str | None = None) -> requests.Response:
    """Submit Arabic text only when explicitly invoked.

    ANSG_ARABICNLP_ENDPOINT defaults to HTTPS; replace for your provider if needed.
    ANSG_ARABICNLP_API_TOKEN is optional and never logged.
    """
    url = endpoint or os.environ.get("ANSG_ARABICNLP_ENDPOINT", "https://arabicnlp.pro/alp/")
    token = os.environ.get("ANSG_ARABICNLP_API_TOKEN")
    headers = {"Accept-Language": "ar-AR"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.post(url, json={"text": text}, headers=headers, timeout=30)
    response.raise_for_status()
    return response


if __name__ == "__main__":
    print("Legacy client: import analyze_text() to invoke; no request sent by default.")
