import argparse
import json
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    with open("data/golden_dataset.json", "r") as f:
        dataset = json.load(f)

    conditions = load_conditions()
    search = HybridSearch(conditions)

    print(f"k={args.limit}")

    for test_case in dataset["test_cases"]:
        query = test_case["query"]
        results = search.rrf_search(query=query, k=60, limit=args.limit)

        retrieved_titles = [r["doc"]["title"] for r in results]
        relevant_titles = test_case["relevant_docs"]

        matches = sum(1 for t in retrieved_titles if t in relevant_titles)
        precision = matches / args.limit
        recall = matches / len(relevant_titles) if relevant_titles else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"\n- Query: {query}")
        print(f"  - Precision@{args.limit}: {precision:.4f}")
        print(f"  - Recall@{args.limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")


if __name__ == "__main__":
    main()
