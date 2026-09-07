"""
api_client.py
==============
Every call the Streamlit app makes to the FastAPI backend goes through
this one file. If the backend URL changes (e.g. moving from localhost
to a deployed Render URL), this is the ONLY place to update.
"""

import os

import requests
import streamlit as st

# Reads from a Streamlit secret in production, falls back to localhost
# for local development. Set this in frontend/.streamlit/secrets.toml
# (never commit that file) or as an environment variable.
# st.secrets raises (not just returns None) when no secrets.toml exists at
# all, which is the normal case for local dev — so this has to be a
# try/except, not a plain .get().
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def check_backend_health() -> dict | None:
    """Returns the /health response, or None if the backend is unreachable."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def get_prediction(employee_data: dict) -> dict:
    """
    Sends one employee's form data to POST /predict.
    Raises a RuntimeError with a readable message on failure, so the
    calling page can just try/except and st.error() it.
    """
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=employee_data, timeout=10)
        if response.status_code == 422:
            # Pydantic validation error — extract the field-level messages
            details = response.json().get("detail", [])
            messages = [f"{'.'.join(str(x) for x in d['loc'])}: {d['msg']}" for d in details]
            raise RuntimeError("Invalid input — " + "; ".join(messages))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Can't reach the prediction API at {BACKEND_URL}. "
            "Is the FastAPI backend running?"
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Prediction request failed: {exc}")
