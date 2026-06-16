#!/usr/bin/env python3
from lib.keyword_search import (
    search_command, build_command, tf_command, idf_command,
    tfidf_command, bm25_idf_command, bm25_tf_command,
    bm25search_command,
)
from lib.search_utils import BM25_K1, BM25_B
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Build BM25 index cache")

    sp = subparsers.add_parser("search", help="Inverted-index search")
    sp.add_argument("query", type=str)

    sp = subparsers.add_parser("bm25search", help="Full BM25 search")
    sp.add_argument("query", type=str)

    sp = subparsers.add_parser("tf", help="Term frequency")
    sp.add_argument("doc_id", type=int)
    sp.add_argument("term", type=str)

    sp = subparsers.add_parser("idf", help="Inverse document frequency")
    sp.add_argument("term", type=str)

    sp = subparsers.add_parser("tfidf", help="TF-IDF score")
    sp.add_argument("doc_id", type=int)
    sp.add_argument("term", type=str)

    sp = subparsers.add_parser("bm25idf", help="BM25 IDF score")
    sp.add_argument("term", type=str)

    sp = subparsers.add_parser("bm25tf", help="BM25 TF score")
    sp.add_argument("doc_id", type=int)
    sp.add_argument("term", type=str)
    sp.add_argument("k1", type=float, nargs="?", default=BM25_K1)
    sp.add_argument("b", type=float, nargs="?", default=BM25_B)

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            results = search_command(args.query, 5)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']}")
        case "bm25search":
            print(f"Searching for: {args.query}")
            results = bm25search_command(args.query)
            for i, r in enumerate(results, 1):
                print(f"{i}. ({r['doc_id']}) {r['title']} - Score: {r['score']:.2f}")
        case "tf":
            print(tf_command(args.doc_id, args.term))
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            print(f"BM25 IDF of '{args.term}': {bm25_idf_command(args.term):.2f}")
        case "bm25tf":
            print(f"BM25 TF of '{args.term}' in doc {args.doc_id}: {bm25_tf_command(args.doc_id, args.term, args.k1, args.b):.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
