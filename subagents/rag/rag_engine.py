#!/usr/bin/env python3
"""
RAG Engine - Hybrid retrieval with ChromaDB (dense) + BM25 (sparse) + RRF fusion.

Architecture:
  - Dense retrieval: ChromaDB with BGE-M3 sentence-transformer embeddings (1024d, cosine)
  - Sparse retrieval: rank_bm25 + jieba (Chinese tokenization)
  - Fusion: Reciprocal Rank Fusion (RRF, k=60)

Tools exposed (selected via argv[1]):
  - rag_search  : Hybrid search over the knowledge base
  - rag_ingest  : Ingest files/directories (md/txt/pdf/docx/csv) with chunking + dedup
  - rag_status  : Report collection statistics

Usage:
  echo '{"query":"GPU算力","top_k":5}' | python3 rag_engine.py rag_search --data-dir ~/.prometheus/data
  echo '{"path":"~/notes","recursive":true}' | python3 rag_engine.py rag_ingest --data-dir ~/.prometheus/data
  echo '{}' | python3 rag_engine.py rag_status --data-dir ~/.prometheus/data

Input is JSON on stdin; output is JSON on stdout (compatible with MTClaw FR execute_tool).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag_engine")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("RAG_MODEL_NAME", "BAAI/bge-m3")
MODEL_LOCAL_PATH = os.environ.get("RAG_MODEL_PATH", "")  # if set, load from local dir
CHUNK_SIZE_DEFAULT = 512  # tokens (~350 Chinese chars)
CHUNK_OVERLAP_DEFAULT = 64
MIN_CHUNK_CHARS = 30
MAX_CHUNK_CHARS = 4096
RRF_K = 60  # RRF smoothing parameter
BM25_OVERFETCH = 3  # fetch top_k * BM25_OVERFETCH from each retriever before fusion
DENSE_OVERFETCH = 3

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".csv"}

# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_DEFAULT,
               overlap: int = CHUNK_OVERLAP_DEFAULT) -> list[str]:
    """Split text into overlapping chunks.

    Uses a simple character-based splitter with paragraph/heading awareness.
    For Chinese text, ~0.7 chars per token, so chunk_size tokens ≈ chunk_size*0.7 chars.
    """
    if not text or not text.strip():
        return []

    char_limit = int(chunk_size * 0.7)  # approximate tokens -> chars
    char_limit = max(char_limit, MIN_CHUNK_CHARS)
    char_limit = min(char_limit, MAX_CHUNK_CHARS)
    overlap_chars = int(overlap * 0.7)
    overlap_chars = min(overlap_chars, char_limit // 2)

    # Try splitting by markdown headings first (## or ###)
    heading_pattern = re.compile(r'^(#{1,3}\s+.+)$', re.MULTILINE)
    sections: list[str] = []
    parts = heading_pattern.split(text)

    current_section = ""
    for part in parts:
        if heading_pattern.match(part):
            if current_section.strip():
                sections.append(current_section.strip())
            current_section = part
        else:
            current_section += part
    if current_section.strip():
        sections.append(current_section.strip())

    # If no headings found, split by double newlines (paragraphs)
    if len(sections) <= 1:
        paragraphs = re.split(r'\n\s*\n', text)
        sections = [p.strip() for p in paragraphs if p.strip()]

    # Further split long sections by character limit
    chunks: list[str] = []
    for section in sections:
        if len(section) <= char_limit:
            chunks.append(section)
        else:
            # Slide a window over the section
            start = 0
            while start < len(section):
                end = start + char_limit
                chunk = section[start:end]
                # Try to break at a sentence boundary
                if end < len(section):
                    last_period = max(
                        chunk.rfind('。'), chunk.rfind('!'),
                        chunk.rfind('？'), chunk.rfind('.'),
                        chunk.rfind('\n'),
                    )
                    if last_period > char_limit // 2:
                        chunk = chunk[:last_period + 1]
                        end = start + len(chunk)
                chunks.append(chunk.strip())
                if end >= len(section):
                    break
                start = end - overlap_chars

    # Filter out tiny chunks
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


# ---------------------------------------------------------------------------
# BM25 Index (JSON-persisted)
# ---------------------------------------------------------------------------

class BM25Index:
    """BM25 sparse retrieval index using rank_bm25 + jieba.

    Persisted as JSON (doc_ids, documents, tokenized_docs).
    Rebuilt on each ingest; loaded on search startup.
    """

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.doc_ids: list[str] = []
        self.documents: list[str] = []
        self.tokenized_docs: list[list[str]] = []
        self._bm25 = None
        self._jieba = None

    def _get_jieba(self):
        if self._jieba is None:
            import jieba
            # Add common domain terms
            for word in ["GPU", "CUDA", "MTClaw", "Function Router", "ChromaDB",
                         "BGE-M3", "RRF", "BM25", "Prometheus"]:
                jieba.add_word(word)
            self._jieba = jieba
        return self._jieba

    def _tokenize(self, text: str) -> list[str]:
        jieba = self._get_jieba()
        return [t for t in jieba.cut(text) if t.strip()]

    def add_documents(self, doc_ids: list[str], documents: list[str]) -> None:
        """Add documents and rebuild the BM25 index."""
        for doc_id, doc in zip(doc_ids, documents):
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)
                self.documents.append(doc)
                self.tokenized_docs.append(self._tokenize(doc))

        self._rebuild()
        self.save()

    def _rebuild(self) -> None:
        if not self.tokenized_docs:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return [(doc_id, score), ...] sorted by BM25 score descending."""
        if self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [(self.doc_ids[i], float(score)) for i, score in ranked if score > 0]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "doc_ids": self.doc_ids,
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
        }
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.doc_ids = data.get("doc_ids", [])
            self.documents = data.get("documents", [])
            self.tokenized_docs = data.get("tokenized_docs", [])
            self._rebuild()
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load BM25 index: %s", e)

    def count(self) -> int:
        return len(self.doc_ids)


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def rrf_fusion(dense_results: list[tuple[str, float]],
               sparse_results: list[tuple[str, float]],
               top_k: int = 5,
               k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of dense and sparse retrieval results.

    RRF_score(doc) = Σ 1/(k + rank_i(doc))

    Args:
        dense_results: [(doc_id, score), ...] from ChromaDB
        sparse_results: [(doc_id, score), ...] from BM25
        top_k: number of results to return
        k: smoothing parameter (default 60)

    Returns:
        [(doc_id, rrf_score), ...] sorted descending
    """
    scores: dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(dense_results, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    for rank, (doc_id, _) in enumerate(sparse_results, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def read_file_content(filepath: str) -> str:
    """Read text content from a file based on its extension."""
    ext = Path(filepath).suffix.lower()

    if ext in (".md", ".txt"):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if ext == ".pdf":
        try:
            import pymupdf  # PyMuPDF
            doc = pymupdf.open(filepath)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            try:
                import fitz  # older import name
                doc = fitz.open(filepath)
                text = "\n\n".join(page.get_text() for page in doc)
                doc.close()
                return text
            except ImportError:
                logger.warning("pymupdf not installed, skipping PDF: %s", filepath)
                return ""

    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(filepath)
            return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except ImportError:
            logger.warning("python-docx not installed, skipping DOCX: %s", filepath)
            return ""

    if ext == ".csv":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    return ""


def file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def discover_files(path: str, recursive: bool = True) -> list[str]:
    """Find all supported files under a path."""
    p = Path(path).expanduser()
    if not p.exists():
        return []

    if p.is_file():
        return [str(p)] if p.suffix.lower() in SUPPORTED_EXTENSIONS else []

    files = []
    if recursive:
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(f))
    else:
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(f))
    return sorted(files)


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """Hybrid RAG engine: ChromaDB dense + BM25 sparse + RRF fusion."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_path = self.data_dir / "chroma"
        self.bm25_path = self.data_dir / "bm25_index.json"
        self.file_hashes_path = self.data_dir / "file_hashes.json"

        # Lazy-loaded resources
        self._client = None
        self._collection = None
        self._embed_fn = None
        self._bm25_index = None

    # -- Lazy initialization --

    def _get_embed_fn(self):
        if self._embed_fn is None:
            from chromadb.utils import embedding_functions
            model_path = MODEL_LOCAL_PATH if MODEL_LOCAL_PATH else MODEL_NAME
            self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_path,
                device="cpu",
            )
        return self._embed_fn

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = self._client.get_or_create_collection(
                name="documents",
                embedding_function=self._get_embed_fn(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_bm25(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index(str(self.bm25_path))
            self._bm25_index.load()
        return self._bm25_index

    def _load_file_hashes(self) -> dict[str, str]:
        if self.file_hashes_path.exists():
            try:
                with self.file_hashes_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_file_hashes(self, hashes: dict[str, str]) -> None:
        with self.file_hashes_path.open("w", encoding="utf-8") as f:
            json.dump(hashes, f, ensure_ascii=False)

    # -- Search --

    def search(self, query: str, top_k: int = 5, file_type: str = "all",
               source_dir: str | None = None, min_score: float = 0.0) -> dict[str, Any]:
        """Hybrid search: dense (ChromaDB) + sparse (BM25) + RRF fusion."""
        collection = self._get_collection()
        bm25 = self._get_bm25()

        if collection.count() == 0 and bm25.count() == 0:
            return {"query": query, "matches": [], "total": 0,
                    "message": "knowledge base is empty, ingest documents first"}

        # Build metadata filter for ChromaDB
        where_clauses: list[dict] = []
        if file_type and file_type != "all":
            where_clauses.append({"file_type": file_type})
        if source_dir:
            where_clauses.append({"source_dir": source_dir})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        fetch_k = top_k * DENSE_OVERFETCH

        # Dense retrieval via ChromaDB
        dense_results: list[tuple[str, float]] = []
        try:
            query_params: dict[str, Any] = {
                "query_texts": [query],
                "n_results": fetch_k,
                "include": ["metadatas", "documents", "distances"],
            }
            if where:
                query_params["where"] = where
            raw = collection.query(**query_params)
            ids = raw.get("ids", [[]])[0]
            distances = raw.get("distances", [[]])[0]
            # ChromaDB returns cosine distance (1 - similarity), so score = 1 - distance
            dense_results = [(doc_id, 1.0 - dist) for doc_id, dist in zip(ids, distances)]
        except Exception as e:
            logger.warning("Dense retrieval failed: %s", e)

        # Sparse retrieval via BM25
        sparse_results: list[tuple[str, float]] = []
        try:
            sparse_results = bm25.search(query, top_k=fetch_k)
        except Exception as e:
            logger.warning("BM25 retrieval failed: %s", e)

        # RRF fusion
        fused = rrf_fusion(dense_results, sparse_results, top_k=top_k * 2)

        # Filter by min_score and retrieve full metadata
        matches: list[dict] = []
        if fused:
            # Batch-fetch metadata from ChromaDB
            doc_ids = [doc_id for doc_id, _ in fused if doc_id not in
                       {d.get("doc_id") for d in matches}]
            try:
                meta_data = collection.get(ids=doc_ids, include=["metadatas", "documents"])
            except Exception:
                meta_data = {"metadatas": [], "documents": []}

            meta_map: dict[str, dict] = {}
            if meta_data.get("ids"):
                for i, did in enumerate(meta_data["ids"]):
                    meta_map[did] = {
                        "metadata": meta_data["metadatas"][i] if meta_data.get("metadatas") else {},
                        "document": meta_data["documents"][i] if meta_data.get("documents") else "",
                    }

            for doc_id, rrf_score in fused:
                if rrf_score < min_score:
                    continue
                info = meta_map.get(doc_id, {})
                meta = info.get("metadata", {})
                matches.append({
                    "doc_id": doc_id,
                    "content": info.get("document", ""),
                    "score": round(rrf_score, 6),
                    "source_path": meta.get("source_path", ""),
                    "file_type": meta.get("file_type", ""),
                    "title": meta.get("title", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                })
                if len(matches) >= top_k:
                    break

        return {
            "query": query,
            "matches": matches,
            "total": len(matches),
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
        }

    # -- Ingest --

    def ingest(self, path: str, recursive: bool = True,
               chunk_size: int = CHUNK_SIZE_DEFAULT,
               overlap: int = CHUNK_OVERLAP_DEFAULT,
               force: bool = False) -> dict[str, Any]:
        """Ingest files from a path into the knowledge base."""
        files = discover_files(path, recursive=recursive)
        if not files:
            return {"status": "no_files", "path": path, "message": "no supported files found"}

        collection = self._get_collection()
        bm25 = self._get_bm25()
        file_hashes = self._load_file_hashes()

        ingested = 0
        skipped = 0
        failed = 0
        total_chunks = 0
        errors: list[str] = []

        for filepath in files:
            try:
                fp = os.path.expanduser(filepath)
                sha = file_sha256(fp)

                if not force and fp in file_hashes and file_hashes[fp] == sha:
                    skipped += 1
                    continue

                content = read_file_content(fp)
                if not content.strip():
                    skipped += 1
                    continue

                chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    skipped += 1
                    continue

                file_name = Path(fp).stem
                file_ext = Path(fp).suffix.lstrip(".").lower()
                source_dir = Path(fp).parent.name
                base_id = hashlib.md5(fp.encode()).hexdigest()[:12]
                now_iso = datetime.now(timezone.utc).isoformat()

                chunk_ids = [f"{base_id}_{i}" for i in range(len(chunks))]
                chunk_metadatas = [{
                    "source_path": fp,
                    "file_type": file_ext,
                    "source_dir": source_dir,
                    "title": file_name,
                    "chunk_index": i,
                    "ingested_at": now_iso,
                    "file_hash": sha[:16],
                } for i in range(len(chunks))]

                # Add to ChromaDB (upsert to handle force re-ingest)
                collection.upsert(
                    ids=chunk_ids,
                    documents=chunks,
                    metadatas=chunk_metadatas,
                )

                # Add to BM25 index
                bm25.add_documents(chunk_ids, chunks)

                file_hashes[fp] = sha
                ingested += 1
                total_chunks += len(chunks)

            except Exception as e:
                failed += 1
                errors.append(f"{filepath}: {e}")
                logger.error("Failed to ingest %s: %s", filepath, e)

        self._save_file_hashes(file_hashes)

        return {
            "status": "completed",
            "path": path,
            "files_found": len(files),
            "files_ingested": ingested,
            "files_skipped": skipped,
            "files_failed": failed,
            "total_chunks": total_chunks,
            **({"errors": errors[:10]} if errors else {}),
        }

    # -- Status --

    def status(self) -> dict[str, Any]:
        """Return knowledge base statistics."""
        try:
            collection = self._get_collection()
            chroma_count = collection.count()
        except Exception:
            chroma_count = 0

        bm25 = self._get_bm25()
        bm25_count = bm25.count()

        # ChromaDB storage size
        chroma_size = 0
        if self.chroma_path.exists():
            for f in self.chroma_path.rglob("*"):
                if f.is_file():
                    chroma_size += f.stat().st_size

        bm25_size = self.bm25_path.stat().st_size if self.bm25_path.exists() else 0

        file_hashes = self._load_file_hashes()

        return {
            "status": "ok",
            "chroma": {
                "document_count": chroma_count,
                "storage_bytes": chroma_size,
                "storage_mb": round(chroma_size / (1024 * 1024), 2),
                "path": str(self.chroma_path),
            },
            "bm25": {
                "document_count": bm25_count,
                "storage_bytes": bm25_size,
                "path": str(self.bm25_path),
            },
            "files": {
                "total_tracked": len(file_hashes),
            },
            "model": MODEL_LOCAL_PATH if MODEL_LOCAL_PATH else MODEL_NAME,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Engine for MTClaw Function Router")
    parser.add_argument("tool", choices=["rag_search", "rag_ingest", "rag_status"],
                        help="Tool to execute")
    parser.add_argument("--data-dir", default=os.environ.get("RAG_DATA_DIR", "~/.prometheus/data"),
                        help="Data directory for ChromaDB and BM25 index")
    args = parser.parse_args()

    # Read JSON input from stdin
    try:
        raw_input = sys.stdin.read()
        params = json.loads(raw_input) if raw_input.strip() else {}
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid JSON input: {e}"}))
        sys.exit(1)

    engine = RAGEngine(args.data_dir)

    try:
        if args.tool == "rag_search":
            result = engine.search(
                query=params["query"],
                top_k=params.get("top_k", 5),
                file_type=params.get("file_type", "all"),
                source_dir=params.get("source_dir"),
                min_score=params.get("min_score", 0.0),
            )
        elif args.tool == "rag_ingest":
            result = engine.ingest(
                path=params["path"],
                recursive=params.get("recursive", True),
                chunk_size=params.get("chunk_size", CHUNK_SIZE_DEFAULT),
                overlap=params.get("overlap", CHUNK_OVERLAP_DEFAULT),
                force=params.get("force", False),
            )
        elif args.tool == "rag_status":
            result = engine.status()
        else:
            result = {"error": f"unknown tool: {args.tool}"}

        print(json.dumps(result, ensure_ascii=False))

    except KeyError as e:
        print(json.dumps({"error": f"missing required parameter: {e}"}))
        sys.exit(1)
    except Exception as e:
        logger.exception("Engine error")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("RAG_LOG_LEVEL", "WARNING"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    main()
