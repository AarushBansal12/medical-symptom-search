# 🏥 Medical Symptoms Search Engine

A **hybrid BM25 + Semantic** search engine over 80 medical conditions, built with the exact same architecture as the Netflix search engine — just swapped from movies to medicine.

---

## Architecture

```
User query (plain English symptoms)
        │
        ├──► BM25 keyword scorer     (InvertedIndex, Porter stemming)
        │
        └──► Chunked Semantic scorer (all-MiniLM-L6-v2, cosine sim)
                  │
        RRF or Weighted fusion
                  │
        Top-K conditions
                  │
        Gemini 2.0 Flash  ──►  Natural language RAG response
```

---

## Project Structure

```
.
├── lib/
│   ├── __init__.py
│   ├── search_utils.py          # Paths, BM25 constants, load_conditions()
│   ├── keyword_search.py        # InvertedIndex, BM25, TF-IDF
│   ├── semantic_search.py       # SemanticSearch, ChunkedSemanticSearch
│   └── hybrid_search.py         # HybridSearch (weighted + RRF)
│
├── data/
│   ├── conditions.json          # 80 medical conditions dataset
│   ├── golden_dataset.json      # Evaluation test cases
│   └── stopwords.txt            # Custom stopword list
│
├── cache/                       # Auto-generated index files
│
├── keyword_search_cli.py        # BM25 search commands
├── semantic_search_cli.py       # Semantic search commands
├── hybrid_search_cli.py         # Hybrid search + query enhancement + reranking
├── augmented_generation_cli.py  # Gemini RAG (rag, summarize, citations, question)
├── evaluation_cli.py            # Precision/Recall/F1 evaluation
├── pyproject.toml
└── .env                         # GEMINI_API_KEY=...
```

---

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Create .env
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Build BM25 index (run once)
python keyword_search_cli.py build

# 4. Build semantic embeddings (run once)
python semantic_search_cli.py embed_chunks
```

---

## Usage

### Keyword Search (BM25)
```bash
python keyword_search_cli.py bm25search "chest pain shortness of breath"
python keyword_search_cli.py search "fever headache"
python keyword_search_cli.py bm25idf "fever"
python keyword_search_cli.py tf 1 "fever"
```

### Semantic Search
```bash
python semantic_search_cli.py search "I feel really tired and my joints hurt every morning"
python semantic_search_cli.py search_chunked "pain behind my eyes and high fever"
python semantic_search_cli.py verify
```

### Hybrid Search (RRF)
```bash
# Basic RRF search
python hybrid_search_cli.py rrf-search "severe headache fever stiff neck"

# With query spelling correction
python hybrid_search_cli.py rrf-search "chets pain brething difficultie" --enhance spell

# With query rewrite (vague → medical)
python hybrid_search_cli.py rrf-search "my tummy hurts after eating" --enhance rewrite

# With query expansion
python hybrid_search_cli.py rrf-search "yellow skin fatigue" --enhance expand

# With batch reranking + evaluation
python hybrid_search_cli.py rrf-search "joint pain morning" --rerank-method batch --evaluate

# Weighted search (alpha: 0=pure BM25, 1=pure semantic)
python hybrid_search_cli.py weighted-search "persistent cough blood" --alpha 0.6
```

### RAG with Gemini
```bash
# Full RAG response
python augmented_generation_cli.py rag "I have a bad headache, fever and my neck feels really stiff"

# Summarize results
python augmented_generation_cli.py summarize "chest feels tight and I can't breathe"

# Answer with citations
python augmented_generation_cli.py citations "yellow skin and pain in my right side"

# Conversational question
python augmented_generation_cli.py question "What could cause me to be really tired all the time?"
```

### Evaluation
```bash
python evaluation_cli.py --limit 5
```

---

## Example Queries

```
"I have a bad headache, fever and my neck feels really stiff"  → Meningitis (Emergency)
"feeling very tired, yellow skin, pain in my right side"       → Gallstones, Hepatitis
"chest feels tight and I can't breathe when walking upstairs"  → Angina, Heart Failure, Asthma
"my joints are swollen and painful especially in the mornings" → Rheumatoid Arthritis, Gout
"I've been really thirsty and peeing a lot, also losing weight"→ Type 1 Diabetes
```

---

## Same Architecture as Netflix Engine

| Component | Netflix | Medical |
|-----------|---------|---------|
| Data loader | `load_movie()` | `load_conditions()` |
| Dataset | `movies.json` | `conditions.json` |
| Index class | `InvertedIndex` | `InvertedIndex` |
| Semantic class | `ChunkedSemanticSearch` | `ChunkedSemanticSearch` |
| Fusion class | `HybridSearch` | `HybridSearch` |
| Embedding model | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |
| BM25 params | k1=1.5, b=0.75 | k1=1.5, b=0.75 |
| RRF default k | 60 | 60 |
| Query enhancement | spell/rewrite/expand | spell/rewrite/expand |

---

## ⚠️ Disclaimer

This tool is **for informational and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.
