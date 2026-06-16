import json
from pathlib import Path

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"
CONDITIONS_PATH = DATA_PATH / "conditions.json"
STOPWORDS_PATH = DATA_PATH / "stopwords.txt"
CACHE_PATH = PROJECT_ROOT / "cache"


def load_conditions() -> list[dict]:
    with open(CONDITIONS_PATH, "r") as f:
        data = json.load(f)
    return data["conditions"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
