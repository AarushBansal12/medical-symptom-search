#!/usr/bin/env python3
import argparse
import json
import re
from lib.semantic_search import (
    verify_model, embed_text, verify_embeddings,
    embed_query_text, SemanticSearch, ChunkedSemanticSearch,
)
from lib.search_utils import load_conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify the embedding model")
    subparsers.add_parser("verify_embeddings", help="Verify condition embeddings")
    subparsers.add_parser("embed_chunks", help="Generate chunk embeddings")

    sp = subparsers.add_parser("embed_text")
    sp.add_argument("text", type=str)

    sp = subparsers.add_parser("embed_query")
    sp.add_argument("query", type=str)

    sp = subparsers.add_parser("search")
    sp.add_argument("query", type=str)
    sp.add_argument("--limit", type=int, default=5)

    sp = subparsers.add_parser("search_chunked")
    sp.add_argument("query", type=str)
    sp.add_argument("--limit", type=int, default=5)

    sp = subparsers.add_parser("chunk")
    sp.add_argument("text", type=str)
    sp.add_argument("--chunk-size", type=int, default=200)
    sp.add_argument("--overlap", type=int, default=0)

    sp = subparsers.add_parser("semantic_chunk")
    sp.add_argument("text", type=str)
    sp.add_argument("--max-chunk-size", type=int, default=4)
    sp.add_argument("--overlap", type=int, default=0)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings(load_conditions())
        case "search":
            docs = load_conditions()
            engine = SemanticSearch()
            engine.load_or_create_embeddings(docs)
            results = engine.search(args.query, args.limit)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"   {r['description'][:100]}")
                print()
        case "embed_chunks":
            docs = load_conditions()
            search = ChunkedSemanticSearch()
            embs = search.load_or_create_chunk_embeddings(docs)
            print(f"Generated {len(embs)} chunked embeddings")
        case "search_chunked":
            docs = load_conditions()
            search = ChunkedSemanticSearch()
            search.load_or_create_chunk_embeddings(docs)
            results = search.search_chunks(args.query, args.limit)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"    {r['document']}...")
        case "chunk":
            words = args.text.split()
            step = args.chunk_size - args.overlap
            chunks = []
            for i in range(0, len(words), step):
                cw = words[i : i + args.chunk_size]
                if not cw:
                    break
                chunks.append(" ".join(cw))
                if len(cw) < args.chunk_size:
                    break
            print(f"Chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {c}")
        case "semantic_chunk":
            sentences = re.split(r"(?<=[.!?])\s+", args.text.strip())
            step = args.max_chunk_size - args.overlap
            chunks = []
            for i in range(0, len(sentences), step):
                cs = sentences[i : i + args.max_chunk_size]
                if not cs:
                    break
                chunks.append(" ".join(cs))
                if len(cs) < args.max_chunk_size:
                    break
            print(f"Semantically chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {c}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
