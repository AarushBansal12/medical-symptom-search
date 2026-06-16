import os
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if min_s == max_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * semantic_score + (1 - alpha) * bm25_score


def rrf_score(rank: int, k: int) -> float:
    return 1 / (rank + k)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not self.idx.index_path.exists():
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        norm_bm25 = normalize([r["score"] for r in bm25_results])
        norm_sem = normalize([r["score"] for r in semantic_results])

        docs = {}
        for result, score in zip(bm25_results, norm_bm25):
            doc_id = result["doc_id"]
            docs[doc_id] = {"doc": self.document_map[doc_id], "bm25": score, "semantic": 0.0}

        for result, score in zip(semantic_results, norm_sem):
            doc_id = result["id"]
            if doc_id not in docs:
                docs[doc_id] = {"doc": self.document_map[doc_id], "bm25": 0.0, "semantic": score}
            else:
                docs[doc_id]["semantic"] = score

        results = []
        for info in docs.values():
            info["hybrid_score"] = hybrid_score(info["bm25"], info["semantic"], alpha)
            results.append(info)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit)
        semantic_results = self.semantic_search.search_chunks(query, limit)

        docs = {}
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["doc_id"]
            docs[doc_id] = {
                "doc": self.document_map[doc_id],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            score = rrf_score(rank, k)
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc": self.document_map[doc_id],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": score,
                }
            else:
                docs[doc_id]["semantic_rank"] = rank
                docs[doc_id]["rrf_score"] += score

        results = list(docs.values())
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:limit]
