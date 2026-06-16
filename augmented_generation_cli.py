from lib.hybrid_search import HybridSearch
from lib.search_utils import load_conditions
from google import genai
from dotenv import load_dotenv
import os
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    sp = subparsers.add_parser("rag", help="Retrieval Augmented Generation")
    sp.add_argument("query", type=str)

    sp = subparsers.add_parser("summarize", help="Summarize search results")
    sp.add_argument("query", type=str)
    sp.add_argument("--limit", type=int, default=5)

    sp = subparsers.add_parser("citations", help="Answer with citations")
    sp.add_argument("query", type=str)
    sp.add_argument("--limit", type=int, default=5)

    sp = subparsers.add_parser("question", help="Answer symptom questions")
    sp.add_argument("question", type=str)
    sp.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    DISCLAIMER = "\n⚠️  DISCLAIMER: This is informational only. Always consult a qualified healthcare professional."

    def _build_docs_str(results):
        out = ""
        for r in results:
            doc = r["doc"]
            out += (
                f"Condition: {doc['title']}\n"
                f"Severity: {doc.get('severity', 'Unknown')}\n"
                f"Symptoms: {doc.get('symptoms', '')}\n"
                f"Causes: {doc.get('causes', '')}\n"
                f"What to do: {doc.get('what_to_do', '')}\n\n"
            )
        return out

    match args.command:
        case "rag":
            conditions = load_conditions()
            search = HybridSearch(conditions)
            results = search.rrf_search(query=args.query, k=60, limit=5)
            docs = _build_docs_str(results)

            prompt = f"""You are a RAG agent for MedSearch, a medical symptom information service.

Your task is to provide a natural-language answer to the user's symptom query based on retrieved medical conditions.

Provide a comprehensive, empathetic answer. Always include a clear disclaimer at the end.
If any condition is an Emergency, make it very prominent.

Query: {args.query}

Retrieved Conditions:
{docs}

Answer:"""

            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            print("Search Results:")
            for r in results:
                print(f"- {r['doc']['title']}")
            print("\nRAG Response:")
            print(response.text)
            print(DISCLAIMER)

        case "summarize":
            conditions = load_conditions()
            search = HybridSearch(conditions)
            results = search.rrf_search(query=args.query, k=60, limit=args.limit)
            docs = _build_docs_str(results)

            prompt = f"""Provide a concise information summary for the symptom query below based on the retrieved conditions.

Goal: Help the user understand what conditions match their symptoms and what they should do next.
Be information-dense, empathetic, and clear. Mention severity levels. 3-4 sentences max.

Query: {args.query}

Retrieved Conditions:
{docs}

Summary:"""

            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            print("Search Results:")
            for r in results:
                print(f"- {r['doc']['title']}")
            print("\nSummary:")
            print(response.text)
            print(DISCLAIMER)

        case "citations":
            conditions = load_conditions()
            search = HybridSearch(conditions)
            results = search.rrf_search(query=args.query, k=60, limit=args.limit)

            documents = ""
            for i, r in enumerate(results, 1):
                doc = r["doc"]
                documents += (
                    f"[{i}] Condition: {doc['title']} | Severity: {doc.get('severity', '?')}\n"
                    f"Symptoms: {doc.get('symptoms', '')}\n"
                    f"What to do: {doc.get('what_to_do', '')}\n\n"
                )

            prompt = f"""Answer the symptom query below using the provided medical condition documents.

Query: {args.query}

Documents:
{documents}

Instructions:
- Cite sources as [1], [2], etc.
- Mention severity clearly for each condition
- If not enough information, say so
- Be direct and informative

Answer:"""

            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            print("Search Results:")
            for r in results:
                print(f"- {r['doc']['title']}")
            print("\nAnswer with Citations:")
            print(response.text)
            print(DISCLAIMER)

        case "question":
            conditions = load_conditions()
            search = HybridSearch(conditions)
            results = search.rrf_search(query=args.question, k=60, limit=args.limit)
            docs = _build_docs_str(results)

            prompt = f"""Answer the user's medical question based on the retrieved conditions.

Question: {args.question}

Conditions:
{docs}

Instructions:
- Answer directly and concisely
- Be clear about severity and urgency
- Sound like a knowledgeable, empathetic friend - not a clinical robot
- Do not diagnose, only inform

Answer:"""

            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            print("Search Results:")
            for r in results:
                print(f"- {r['doc']['title']}")
            print("\nAnswer:")
            print(response.text)
            print(DISCLAIMER)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
