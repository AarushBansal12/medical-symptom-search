from lib.search_utils import load_conditions, load_stopwords, CACHE_PATH, BM25_K1, BM25_B
import string
import pickle
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import os
import math

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"
        self.doc_lengths_path = CACHE_PATH / "doc_lengths.pkl"

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def get_document(self, term):
        return sorted(list(self.index[term]))

    def _get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 token")
        return self.term_frequencies[str(doc_id)][token[0]]

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        doc_length = self.doc_lengths[str(doc_id)]
        avg_dl = self._get_avg_doc_length()
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_dl)))

    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 token")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 token")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log(((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5)) + 1)

    def get_bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int = 5) -> list[dict]:
        query_tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.get_bm25(doc_id, token)
            scores[doc_id] = score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = sorted_scores[:limit]
        formatted = []
        for doc_id, score in results:
            title = self.docmap[doc_id]["title"]
            formatted.append({"doc_id": doc_id, "title": title, "score": score})
        return formatted

    def build(self):
        conditions = load_conditions()
        for cond in conditions:
            doc_id = cond["id"]
            # Index title + symptoms + description for rich keyword coverage
            text = f"{cond['title']} {cond['symptoms']} {cond['description']}"
            self.docmap[doc_id] = cond
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = clean_text(text)
    stopwords = load_stopwords()
    res = []
    for tok in text.split():
        tok = tok.strip("\n")
        if tok and tok not in stopwords:
            tok = stemmer.stem(tok)
            res.append(tok)
    return res


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print(f"Index built with {len(idx.docmap)} conditions.")


def bm25search_command(query: str, limit: int = 5) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tfidf(doc_id, term)
    print(f"TF-IDF score of '{term}' in doc '{doc_id}': {tf_idf:.2f}")


def idf_command(term: str):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def search_command(query: str, n_results: int = 5) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    seen, res = set(), []
    query_tokens = tokenize_text(query)
    for qt in query_tokens:
        for matching_doc_id in idx.get_document(qt):
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            res.append(idx.docmap[matching_doc_id])
            if len(res) >= n_results:
                return res
    return res
