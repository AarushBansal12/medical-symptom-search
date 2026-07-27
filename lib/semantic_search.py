from fastembed import TextEmbedding
import math
import numpy as np
import os
import json
import re

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
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return np.array(list(self.model.embed([text]))[0])

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = np.array(list(self.model.embed(texts)))
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
        print("Step 1: Building chunks...")

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
        print(f"Step 2: Total chunks = {len(all_chunks)}")
        print("Step 3: Starting embedding generation...")
        self.chunk_embeddings = np.array(list(self.model.embed(all_chunks)))
        print("Step 4: Embeddings generated")
        self.chunk_metadata = chunk_metadata
        os.makedirs(CACHE_PATH, exist_ok=True)
        np.save(str(CACHE_PATH / "chunk_embeddings.npy"), self.chunk_embeddings)
        print("Step 5: Saved embeddings")
        with open(str(CACHE_PATH / "chunk_metadata.json"), "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f)
        print("Step 6: Finished build_chunk_embeddings")
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
        print("No cache found. Building embeddings...")
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
    print(s.model)


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
