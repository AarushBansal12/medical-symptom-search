"""
api.py — FastAPI backend for Medical Symptoms Search Engine
Run with: uv run uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_conditions

load_dotenv()

app = FastAPI(title="Medical Symptoms Search API", version="1.0.0")

# Allow React dev server and deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://medical-symptom-search-8nlq.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading variables
conditions = None
search_engine = None
CATEGORIES = None

SEVERITIES = ["Emergency", "Severe", "Chronic-Severe", "Moderate-Severe",
              "Moderate", "Mild-Moderate", "Mild-Severe", "Mild", "Chronic"]


def get_search_engine():
    global conditions, search_engine, CATEGORIES
    if search_engine is None:
        print("Loading search engine...")
        conditions = load_conditions()
        search_engine = HybridSearch(conditions)
        CATEGORIES = sorted(set(c["category"] for c in conditions))
        print(f"✅ Ready — {len(conditions)} conditions indexed.")
    return search_engine, conditions, CATEGORIES


@app.get("/")
def root():
    return {"status": "ok", "message": "Medical Symptoms Search API"}


@app.get("/categories")
def get_categories_endpoint():
    _, _, categories = get_search_engine()
    return {"categories": categories}


@app.get("/severities")
def get_severities_endpoint():
    return {"severities": SEVERITIES}


@app.get("/search")
def search(
    q: str = Query(..., description="Symptom query"),
    category: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=20),
    k: int = Query(60),
):
    try:
        engine, _, _ = get_search_engine()
        results = engine.rrf_search(query=q, k=k, limit=50)

        # Apply filters
        filtered = []
        for r in results:
            doc = r["doc"]

            if category and doc.get("category") != category:
                continue

            if severity and doc.get("severity") != severity:
                continue

            filtered.append(r)

        filtered = filtered[:limit]

        return {
            "query": q,
            "total": len(filtered),
            "results": [
                {
                    "id": r["doc"].get("id"),
                    "title": r["doc"].get("title"),
                    "category": r["doc"].get("category"),
                    "severity": r["doc"].get("severity"),
                    "symptoms": r["doc"].get("symptoms"),
                    "causes": r["doc"].get("causes"),
                    "what_to_do": r["doc"].get("what_to_do"),
                    "description": r["doc"].get("description"),
                    "rrf_score": r.get("rrf_score"),
                    "bm25_rank": r.get("bm25_rank"),
                    "semantic_rank": r.get("semantic_rank"),
                }
                for r in filtered
            ]
        }

    except Exception:
        import traceback
        traceback.print_exc()
        raise