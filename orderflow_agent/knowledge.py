"""Operational dual-RAG service with source and timing traces."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from .models import ToolStep
from .runtime.rag import DualRAGEngine, RAGError, build_grounded_prompt, extract_document_text, make_chunks
from .storage import SQLiteStorageAdapter


class KnowledgeService:
    def __init__(self, storage: SQLiteStorageAdapter, provider: object | None = None) -> None:
        self.storage = storage
        self.provider = provider

    def seed(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            return
        data = source.read_bytes()
        text = extract_document_text(source.name, data)
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = [
            row
            for row in self.storage.list_knowledge_documents()
            if row["source_name"] == source.name
        ]
        if any(row["checksum"] == checksum for row in existing):
            return
        for row in existing:
            self.storage.delete_knowledge_document(row["id"])
        self.ingest(source.name, "text/markdown", data, title="OrderFlow operating knowledge")

    def ingest(self, filename: str, mime_type: str, data: bytes, *, title: str = "") -> tuple[str, bool, int]:
        text = extract_document_text(filename, data)
        document_id = str(uuid4())
        chunks = make_chunks(document_id, title.strip() or Path(filename).stem, filename, text)
        stored_id, created = self.storage.save_knowledge_document(
            title=title.strip() or Path(filename).stem,
            source_name=Path(filename).name,
            mime_type=mime_type or "application/octet-stream",
            text=text,
            chunks=chunks,
        )
        return stored_id, created, len(chunks)

    def answer(self, query: str) -> tuple[str, tuple[ToolStep, ...]] | None:
        chunks = self.storage.list_knowledge_chunks()
        if not chunks:
            return None
        embedder = self.provider if getattr(getattr(self.provider, "capabilities", None), "embeddings", False) else None
        engine = DualRAGEngine(chunks, embedder=embedder)
        result = engine.retrieve(query, top_k=4)
        if not result.hits:
            return None
        generated = False
        if self.provider is not None:
            try:
                prompt = build_grounded_prompt(query, result.hits)
                answer = self.provider.generate(
                    "Answer only from the supplied operational evidence. Preserve source markers.",
                    (("user", prompt),),
                )
                generated = True
            except Exception:
                answer = self._extractive_answer(result)
        else:
            answer = self._extractive_answer(result)
        provider_id = getattr(self.provider, "provider_id", "local") if generated else "local-extractive"
        self.storage.save_retrieval_trace(result, provider_id=provider_id, answer_generated=generated)
        return answer, (
            ToolStep(
                "dual_rag",
                "passed",
                f"Fused {result.trace.lexical_lane} and {result.trace.vector_lane}; selected {len(result.hits)} source(s).",
            ),
            ToolStep("source_guard", "passed", "Answer is tied to stored source chunks."),
        )

    @staticmethod
    def _extractive_answer(result) -> str:
        blocks = []
        for index, hit in enumerate(result.hits[:3], start=1):
            excerpt = " ".join(hit.text.split())[:520]
            blocks.append(f"[S{index}] **{hit.title}**: {excerpt}")
        return "\n\n".join(blocks)


__all__ = ["KnowledgeService", "RAGError"]
