"""Project settings and secret loading.

Secrets are read, in this order, from:
  1. Google Colab Secrets (the key icon in Colab's left sidebar), when running in Colab
  2. A local .env file in the repository root (see .env.example)
  3. An environment variable that is already set
They are never printed, logged or written to disk by this project.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_MERGED = ROOT / "data" / "merged_hourly.csv"
RESULTS = ROOT / "results"

# Study settings (from the proposal)
LAT, LON = 31.5204, 74.3587          # Lahore
TZ = "Asia/Karachi"
CAMS_START = "2022-08-01"            # CAMS global history on Open-Meteo starts here
SEARCH_RADIUS_M = 25000
HORIZONS = (24, 48, 72)              # forecast horizons in hours
EVENT_THRESHOLD = 125.5              # US EPA "Very Unhealthy", 24-h mean, ug/m3


def _from_colab(name):
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name)
    except Exception:
        return None


def _load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def get_secret(name):
    value = _from_colab(name)
    if not value:
        _load_dotenv()
        value = os.getenv(name)
    if not value or value.startswith("your_"):
        raise RuntimeError(
            f"{name} is not set. Add it to Colab Secrets, or copy .env.example to .env "
            f"and fill it in. Never paste the key into code."
        )
    return value
