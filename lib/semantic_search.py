import math
import numpy as np
import os
import json
import re
import requests
import time

from lib.search_utils import CACHE_PATH


def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 1) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not re.search(r"[.!?]$", text):
        sentences = [text]
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    step = max_chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if not chunk_sentences:
            break
        chunk = " ".join(chunk_sentences).strip()
        if chunk:
            chunks.append(chunk)
        if len(chunk_sentences) < max_chunk_size:
            break
    return chunks


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.token = os.environ.get("HF_TOKEN")
        
        if not self.token:
            print("Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.")
            
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def _query_hf_api(self, texts: list[str]) -> np.ndarray:
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts})
        
        # Retry logic if the model is currently loading on HF's servers
        if response.status_code == 503:
            estimated_time = response.json().get("estimated_time", 10.0)
            print(f"Model is loading on Hugging Face API. Waiting {estimated_time} seconds...")
            time.sleep(estimated_time)
            response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts})
            
        if response.status_code != 200:
            raise ValueError(f"Hugging Face API Error: {response.text}")
            
        return np.array(response.json())

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return self._query_hf_api([text])[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        
        # Process in batches of 100 to avoid API payload limits
        batch_size = 100
        all_embeddings = []
        
        print(f"Generating embeddings for {len(texts)} documents via Hugging Face API...")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._query_hf_api(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
        self.embeddings = np.array(all_embeddings)
        os.makedirs(CACHE_PATH, exist_ok=True)
        np.save(str(CACHE_PATH / "condition_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        cache_file = CACHE_PATH / "condition_embeddings.npy"
        if cache_file.exists():
            self.embeddings = np.load(str(cache_file))
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_or_create_embeddings first.")
        q_emb = self.generate_embedding(query)
        results = []
        for i, emb in enumerate(self.embeddings):
            score = cosine_similarity(q_emb, emb)
            results.append((score, self.documents[i]))
        results.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": s, "title": d["title"], "description": d["description"]}
            for s, d in results[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        all_chunks = []
        chunk_metadata = []
        for cond_idx, doc in enumerate(documents):
            text = doc["description"]
            if not text:
                continue
            chunks = semantic_chunk_text(text, max_chunk_size=4, overlap=1)
            all_chunks.extend(chunks)
            for chunk_idx, _ in enumerate(chunks):
                chunk_metadata.append(
                    {"condition_idx": cond_idx, "chunk_idx": chunk_idx, "total_chunks": len(chunks)}
                )
        # Process in batches of 100 to avoid API payload limits
        batch_size = 100
        all_embeddings = []
        
        print(f"Generating embeddings for {len(all_chunks)} chunks via Hugging Face API...")
        for i in range(0, len(all_chunks), batch_size):
            batch_texts = all_chunks[i:i + batch_size]
            batch_embeddings = self._query_hf_api(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
        self.chunk_embeddings = np.array(all_embeddings)
        self.chunk_metadata = chunk_metadata
        os.makedirs(CACHE_PATH, exist_ok=True)
        np.save(str(CACHE_PATH / "chunk_embeddings.npy"), self.chunk_embeddings)
        with open(str(CACHE_PATH / "chunk_metadata.json"), "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        emb_file = CACHE_PATH / "chunk_embeddings.npy"
        meta_file = CACHE_PATH / "chunk_metadata.json"
        if emb_file.exists() and meta_file.exists():
            self.chunk_embeddings = np.load(str(emb_file))
            with open(str(meta_file)) as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        q_emb = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            score = cosine_similarity(q_emb, chunk_emb)
            meta = self.chunk_metadata[i]
            chunk_scores.append({"condition_idx": meta["condition_idx"], "score": score})
        # Best chunk score per condition
        cond_scores = {}
        for cs in chunk_scores:
            cidx = cs["condition_idx"]
            if cidx not in cond_scores or cs["score"] > cond_scores[cidx]:
                cond_scores[cidx] = cs["score"]
        sorted_conds = sorted(cond_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cond_idx, score in sorted_conds[:limit]:
            doc = self.documents[cond_idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, 4),
                    "metadata": {},
                }
            )
        return results


def verify_model():
    s = SemanticSearch()
    print(f"Model: {s.model!s}")
    print(f"Max sequence length: {s.model.max_seq_length}")


def embed_text(text: str):
    s = SemanticSearch()
    emb = s.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dims: {emb[:3]}")
    print(f"Dimensions: {emb.shape[0]}")


def embed_query_text(query: str):
    s = SemanticSearch()
    emb = s.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dims: {emb[:3]}")
    print(f"Shape: {emb.shape}")


def verify_embeddings(documents: list[dict]):
    s = SemanticSearch()
    embs = s.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embs.shape[0]} vectors in {embs.shape[1]} dimensions")
