import argparse
import json
import time
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_conditions
import os
from dotenv import load_dotenv
from google import genai


# ── Query enhancement helpers (mirrors Netflix engine) ────────────────────────

def enhance_query(query: str) -> str:
    """Spell-correct a medical symptom query."""
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""Fix any spelling errors in the medical symptom search query below.
Correct only clear, high-confidence typos. Do not add, remove, or rewrite anything.
If there are no errors, return the original query unchanged.
Output only the final query text.

User query: "{query}"
""",
    )
    return response.text.strip()


def rewrite_query(query: str) -> str:
    """Rewrite a vague symptom description into a more specific medical search query."""
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""Rewrite the medical symptom search query below to be more specific and medically precise.

Consider:
- Use proper medical terminology where appropriate
- Mention body system or organ if implied
- Keep the rewritten query concise (under 12 words)
- It should be a search-style query, not a sentence

If you cannot improve it, output the original unchanged.
Output only the rewritten query text.

User query: "{query}"
""",
    )
    return response.text.strip()


def expand_query(query: str) -> str:
    """Add related medical terms to the query."""
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""Expand the medical symptom search query with related terms.

Add synonyms, related symptoms, and medical terms that may appear in condition descriptions.
Output only the additional terms to append to the original query.

Examples:
- "chest pain breathing" -> "chest pain breathing pleurisy pneumothorax dyspnea"
- "joint swelling morning" -> "joint swelling morning stiffness arthritis rheumatoid"

User query: "{query}"
""",
    )
    return response.text.strip()


def rerank_batch(query: str, results: list[dict]) -> list[int]:
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    doc_list = "\n".join(
        f"ID: {r['doc']['id']}\nCondition: {r['doc']['title']}\nSymptoms: {r['doc'].get('symptoms', '')[:200]}"
        for r in results
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""Rank these medical conditions by relevance to the symptom query below.

Query: "{query}"

Conditions:
{doc_list}

Return the condition IDs as a raw JSON array of strings, best match first.
Do not wrap in Markdown or add explanations.

Example: ["12", "3", "7", "1", "5"]

Ranking:
""",
    )
    return json.loads(response.text.strip())


def evaluate_results(query: str, results: list[dict]) -> list[int]:
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    formatted = "\n".join(
        f"{r['doc']['title']} - Symptoms: {r['doc'].get('symptoms', '')[:150]}"
        for r in results
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""Rate how relevant each medical condition is to this symptom query on a 0-3 scale.

Query: "{query}"

Results:
{formatted}

Scale: 3=Highly relevant, 2=Relevant, 1=Marginally relevant, 0=Not relevant

Return ONLY a JSON array of integers, e.g. [3, 1, 0, 2, 1]
""",
    )
    return json.loads(response.text.strip())


# ── Main CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    sp = subparsers.add_parser("normalize")
    sp.add_argument("scores", nargs="*", type=float)

    sp = subparsers.add_parser("rrf-search")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=60)
    sp.add_argument("--limit", type=int, default=5)
    sp.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"])
    sp.add_argument("--rerank-method", type=str, choices=["batch"])
    sp.add_argument("--evaluate", action="store_true")

    sp = subparsers.add_parser("weighted-search")
    sp.add_argument("query")
    sp.add_argument("--alpha", type=float, default=0.5)
    sp.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    def _normalize(scores):
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if lo == hi:
            return [1.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    match args.command:
        case "normalize":
            for s in _normalize(args.scores):
                print(f"* {s:.4f}")

        case "weighted-search":
            conditions = load_conditions()
            search = HybridSearch(conditions)
            results = search.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results, 1):
                doc = r["doc"]
                print(f"{i}. {doc['title']}")
                print(f"   Hybrid: {r['hybrid_score']:.3f}  BM25: {r['bm25']:.3f}  Semantic: {r['semantic']:.3f}")
                print(f"   {doc['description'][:100]}")
                print()

        case "rrf-search":
            query = args.query
            print(f"Original Query: {query}")

            if args.enhance == "spell":
                enhanced = enhance_query(query)
                print(f"Enhanced (spell): '{query}' -> '{enhanced}'")
                query = enhanced
            elif args.enhance == "rewrite":
                enhanced = rewrite_query(query)
                print(f"Enhanced (rewrite): '{query}' -> '{enhanced}'")
                query = enhanced
            elif args.enhance == "expand":
                enhanced = expand_query(query)
                print(f"Enhanced (expand): '{query}' -> '{enhanced}'")
                query = enhanced

            conditions = load_conditions()
            search = HybridSearch(conditions)
            search_limit = args.limit * 5 if args.rerank_method else args.limit
            results = search.rrf_search(query, args.k, search_limit)

            print("\nRRF Results:")
            for i, r in enumerate(results[:5], 1):
                print(f"  {i}. {r['doc']['title']} (RRF={r['rrf_score']:.3f})")

            if args.rerank_method == "batch":
                print(f"\nRe-ranking {len(results)} results using batch method...")
                ranked_ids = rerank_batch(query, results)
                rank_map = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, 1)}
                for r in results:
                    r["rerank_rank"] = rank_map.get(r["doc"]["id"], 999999)
                results.sort(key=lambda x: x["rerank_rank"])

            print("\nFinal Results:")
            for i, r in enumerate(results[: args.limit], 1):
                doc = r["doc"]
                sev_icon = {"Emergency": "🚨", "Severe": "🔴", "Moderate-Severe": "🟠",
                            "Moderate": "🟡", "Mild-Moderate": "🟡", "Mild": "🟢",
                            "Chronic": "🔵", "Chronic-Severe": "🔴"}.get(doc.get("severity", ""), "⚪")
                print(f"\n{i}. {sev_icon} {doc['title']} [{doc.get('severity', '?')}]")
                if args.rerank_method == "batch":
                    print(f"   Re-rank: #{r['rerank_rank']}")
                print(f"   RRF Score: {r['rrf_score']:.3f}  BM25 Rank: {r['bm25_rank']}  Semantic Rank: {r['semantic_rank']}")
                print(f"   Symptoms: {doc.get('symptoms', '')[:100]}")
                print(f"   Action: {doc.get('what_to_do', '')[:100]}")

            if args.evaluate:
                scores = evaluate_results(query, results[: args.limit])
                print("\nEvaluation Report:")
                for i, (r, s) in enumerate(zip(results[: args.limit], scores), 1):
                    print(f"  {i}. {r['doc']['title']}: {s}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
