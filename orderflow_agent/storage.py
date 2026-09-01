"""Explicit persistence boundary for sessions, orders, handovers, RAG, and exports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Protocol
from uuid import uuid4

from .models import AgentResponse, ConversationSession, OrderRecord, TranscriptTurn, utc_now


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "orderflow.db"


class StorageAdapter(Protocol):
    def ensure_session(self, session: ConversationSession) -> None: ...

    def record_exchange(
        self,
        session: ConversationSession,
        user_message: str,
        response: AgentResponse,
        channel: str = "text",
    ) -> None: ...

    def append_turn(self, turn: TranscriptTurn) -> None: ...
    def save_order(self, order: OrderRecord) -> None: ...
    def list_orders(self) -> list[dict[str, Any]]: ...
    def find_order(self, reference: str) -> dict[str, Any] | None: ...
    def list_turns(self, session_id: str) -> list[TranscriptTurn]: ...
    def list_tool_traces(self, session_id: str | None = None) -> list[dict[str, Any]]: ...
    def record_signal(self, session_id: str, signal: Any) -> None: ...
    def save_handover(self, handover: Any) -> None: ...
    def get_handover(self, case_id: str) -> dict[str, Any] | None: ...
    def handover_queue_position(self, case_id: str) -> int | None: ...
    def append_handover_message(self, case_id: str, *, role: str, content: str) -> dict[str, str]: ...
    def replace_latest_ai_response(self, session_id: str, content: str) -> bool: ...


class SQLiteStorageAdapter:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("ORDERFLOW_DB_PATH", DEFAULT_DB_PATH))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id TEXT PRIMARY KEY,
                    agent_mode TEXT NOT NULL,
                    strictness INTEGER NOT NULL CHECK(strictness BETWEEN 0 AND 100),
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    repair_requests INTEGER NOT NULL DEFAULT 0,
                    successful_repairs INTEGER NOT NULL DEFAULT 0,
                    compliance_failures INTEGER NOT NULL DEFAULT 0,
                    confirmed_orders INTEGER NOT NULL DEFAULT 0,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    validation_failures INTEGER NOT NULL DEFAULT 0,
                    unsupported_attempts INTEGER NOT NULL DEFAULT 0,
                    confirmation_failures INTEGER NOT NULL DEFAULT 0,
                    handover_active INTEGER NOT NULL DEFAULT 0,
                    handover_case_id TEXT NOT NULL DEFAULT '',
                    fulfilment TEXT NOT NULL DEFAULT 'undecided',
                    delivery_address TEXT NOT NULL DEFAULT '',
                    menu_context_json TEXT NOT NULL DEFAULT '[]',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_traces (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id),
                    user_turn_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_signals (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id),
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    source_turns_json TEXT NOT NULL,
                    method TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id),
                    status TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    lines_json TEXT NOT NULL,
                    fulfilment TEXT NOT NULL DEFAULT 'pickup',
                    delivery_address TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS handovers (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES agent_sessions(id),
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    context_carryover_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_name, checksum)
                );
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS retrieval_traces (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    answer_generated INTEGER NOT NULL,
                    trace_json TEXT NOT NULL,
                    hit_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_signals_session ON conversation_signals(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);
                CREATE INDEX IF NOT EXISTS idx_handovers_status ON handovers(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON knowledge_chunks(document_id, ordinal);
                """
            )
            self._migrate_agent_sessions(connection)
            self._migrate_orders(connection)
            self._migrate_handovers(connection)

    @staticmethod
    def _migrate_agent_sessions(connection: sqlite3.Connection) -> None:
        """Upgrade early local databases without discarding sessions or orders."""
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(agent_sessions)").fetchall()
        }
        if "prompt_mode" in columns and "agent_mode" not in columns:
            connection.execute("ALTER TABLE agent_sessions RENAME COLUMN prompt_mode TO agent_mode")
            columns.remove("prompt_mode")
            columns.add("agent_mode")

        additions = {
            "failed_attempts": "INTEGER NOT NULL DEFAULT 0",
            "validation_failures": "INTEGER NOT NULL DEFAULT 0",
            "unsupported_attempts": "INTEGER NOT NULL DEFAULT 0",
            "confirmation_failures": "INTEGER NOT NULL DEFAULT 0",
            "handover_active": "INTEGER NOT NULL DEFAULT 0",
            "handover_case_id": "TEXT NOT NULL DEFAULT ''",
            "fulfilment": "TEXT NOT NULL DEFAULT 'undecided'",
            "delivery_address": "TEXT NOT NULL DEFAULT ''",
            "menu_context_json": "TEXT NOT NULL DEFAULT '[]'",
        }
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE agent_sessions ADD COLUMN {name} {declaration}")

    @staticmethod
    def _migrate_orders(connection: sqlite3.Connection) -> None:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(orders)").fetchall()}
        additions = {
            "fulfilment": "TEXT NOT NULL DEFAULT 'pickup'",
            "delivery_address": "TEXT NOT NULL DEFAULT ''",
        }
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE orders ADD COLUMN {name} {declaration}")

    @staticmethod
    def _migrate_handovers(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(handovers)").fetchall()
        }
        if "continuity_score" in columns and "context_carryover_score" not in columns:
            connection.execute(
                "ALTER TABLE handovers RENAME COLUMN continuity_score TO context_carryover_score"
            )

    def ensure_session(self, session: ConversationSession) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_sessions
                   (id, agent_mode, strictness, turn_count, repair_requests, successful_repairs,
                    compliance_failures, confirmed_orders, failed_attempts, validation_failures,
                    unsupported_attempts, confirmation_failures, handover_active, handover_case_id,
                    fulfilment, delivery_address, menu_context_json, started_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     agent_mode=excluded.agent_mode,
                     strictness=excluded.strictness,
                     turn_count=excluded.turn_count,
                     repair_requests=excluded.repair_requests,
                     successful_repairs=excluded.successful_repairs,
                     compliance_failures=excluded.compliance_failures,
                     confirmed_orders=excluded.confirmed_orders,
                     failed_attempts=excluded.failed_attempts,
                     validation_failures=excluded.validation_failures,
                     unsupported_attempts=excluded.unsupported_attempts,
                     confirmation_failures=excluded.confirmation_failures,
                     handover_active=excluded.handover_active,
                     handover_case_id=excluded.handover_case_id,
                     fulfilment=excluded.fulfilment,
                     delivery_address=excluded.delivery_address,
                     menu_context_json=excluded.menu_context_json,
                     updated_at=excluded.updated_at""",
                (
                    session.session_id,
                    session.agent_mode.value,
                    session.strictness,
                    session.turn_count,
                    session.repair_requests,
                    session.successful_repairs,
                    session.compliance_failures,
                    session.confirmed_orders,
                    session.failed_attempts,
                    session.validation_failures,
                    session.unsupported_attempts,
                    session.confirmation_failures,
                    int(session.handover_active),
                    session.handover_case_id,
                    session.fulfilment,
                    session.delivery_address,
                    json.dumps(session.menu_context),
                    session.started_at,
                    utc_now(),
                ),
            )

    def record_exchange(
        self,
        session: ConversationSession,
        user_message: str,
        response: AgentResponse,
        channel: str = "text",
    ) -> None:
        self.ensure_session(session)
        user_turn = TranscriptTurn(session.session_id, "customer", user_message, channel=channel)
        ai_turn = TranscriptTurn(session.session_id, "ai", response.content, channel="text")
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO conversation_turns VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (user_turn.id, user_turn.session_id, user_turn.role, user_turn.content, user_turn.channel, user_turn.created_at),
                    (ai_turn.id, ai_turn.session_id, ai_turn.role, ai_turn.content, ai_turn.channel, ai_turn.created_at),
                ],
            )
            connection.execute(
                "INSERT INTO tool_traces VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    session.session_id,
                    user_turn.id,
                    json.dumps([asdict(step) for step in response.tool_trace]),
                    utc_now(),
                ),
            )

    def append_turn(self, turn: TranscriptTurn) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_turns VALUES (?, ?, ?, ?, ?, ?)",
                (
                    turn.id,
                    turn.session_id,
                    turn.role,
                    turn.content,
                    turn.channel,
                    turn.created_at,
                ),
            )

    def replace_latest_ai_response(self, session_id: str, content: str) -> bool:
        """Replace an operational draft with the final model-streamed customer reply."""

        with self._connect() as connection:
            row = connection.execute(
                """SELECT id FROM conversation_turns
                   WHERE session_id = ? AND role = 'ai'
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE conversation_turns SET content = ? WHERE id = ?",
                (content, row["id"]),
            )
            pending = connection.execute(
                "SELECT id, payload FROM handovers WHERE session_id = ? AND status = 'pending'",
                (session_id,),
            ).fetchall()
            for handover in pending:
                payload = json.loads(handover["payload"])
                history = payload.get("conversation_history", [])
                for turn in reversed(history):
                    if turn.get("role") == "ai":
                        turn["content"] = content
                        break
                connection.execute(
                    "UPDATE handovers SET payload = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(payload), utc_now(), handover["id"]),
                )
            return True

    def append_latest_tool_step(self, session_id: str, step: dict[str, Any]) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT id, payload FROM tool_traces
                   WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if row is None:
                return False
            payload = json.loads(row["payload"])
            payload.append(step)
            connection.execute(
                "UPDATE tool_traces SET payload = ? WHERE id = ?",
                (json.dumps(payload), row["id"]),
            )
            return True

    def discard_latest_ai_response(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT id FROM conversation_turns
                   WHERE session_id = ? AND role = 'ai'
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM conversation_turns WHERE id = ?", (row["id"],))
            return True

    def save_order(self, order: OrderRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO orders
                   (id, session_id, status, currency, total, lines_json, fulfilment,
                    delivery_address, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order.id,
                    order.session_id,
                    order.status,
                    order.currency,
                    order.total,
                    json.dumps([asdict(line) for line in order.lines]),
                    order.fulfilment,
                    order.delivery_address,
                    order.created_at,
                ),
            )

    def list_orders(self) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM orders ORDER BY created_at DESC")
        for row in rows:
            row["lines"] = json.loads(row.pop("lines_json"))
        return rows

    def find_order(self, reference: str) -> dict[str, Any] | None:
        """Find one persisted order by its full ID or unambiguous public prefix."""

        normalized = reference.strip().replace("`", "").casefold()
        if len(normalized) < 8 or not all(character in "0123456789abcdef-" for character in normalized):
            return None
        rows = self._rows(
            "SELECT * FROM orders WHERE LOWER(id) LIKE ? ORDER BY created_at DESC LIMIT 2",
            (normalized + "%",),
        )
        if len(rows) != 1:
            return None
        row = rows[0]
        row["lines"] = json.loads(row.pop("lines_json"))
        return row

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM agent_sessions ORDER BY updated_at DESC")

    def list_turns(self, session_id: str) -> list[TranscriptTurn]:
        return [
            TranscriptTurn(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                channel=row["channel"],
                created_at=row["created_at"],
            )
            for row in self._rows(
                "SELECT * FROM conversation_turns WHERE session_id = ? ORDER BY created_at", (session_id,)
            )
        ]

    def list_tool_traces(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id:
            rows = self._rows(
                "SELECT * FROM tool_traces WHERE session_id = ? ORDER BY created_at DESC", (session_id,)
            )
        else:
            rows = self._rows("SELECT * FROM tool_traces ORDER BY created_at DESC")
        for row in rows:
            row["steps"] = json.loads(row.pop("payload"))
        return rows

    def record_signal(self, session_id: str, signal: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO conversation_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    signal.id,
                    session_id,
                    signal.label,
                    signal.confidence,
                    signal.evidence,
                    json.dumps(list(signal.source_turns)),
                    signal.method,
                    signal.created_at,
                ),
            )

    def list_signals(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id:
            rows = self._rows(
                "SELECT * FROM conversation_signals WHERE session_id = ? ORDER BY created_at DESC", (session_id,)
            )
        else:
            rows = self._rows("SELECT * FROM conversation_signals ORDER BY created_at DESC")
        for row in rows:
            row["source_turns"] = json.loads(row.pop("source_turns_json"))
        return rows

    def save_handover(self, handover: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO handovers VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status, payload=excluded.payload,
                   context_carryover_score=excluded.context_carryover_score, updated_at=excluded.updated_at""",
                (
                    handover.id,
                    handover.session_id,
                    handover.status,
                    json.dumps(asdict(handover)),
                    handover.context_carryover_score,
                    handover.created_at,
                    utc_now(),
                ),
            )

    def list_handovers(self) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM handovers ORDER BY created_at DESC")
        for row in rows:
            row["handover"] = json.loads(row.pop("payload"))
        return rows

    def get_handover(self, case_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM handovers WHERE id = ?", (case_id,))
        if not rows:
            return None
        row = rows[0]
        row["handover"] = json.loads(row.pop("payload"))
        return row

    def handover_queue_position(self, case_id: str) -> int | None:
        rows = self._rows(
            "SELECT id FROM handovers WHERE status = 'pending' ORDER BY created_at ASC, id ASC"
        )
        return next((index for index, row in enumerate(rows, start=1) if row["id"] == case_id), None)

    def append_handover_message(self, case_id: str, *, role: str, content: str) -> dict[str, str]:
        clean_role = role.strip().casefold()
        if clean_role not in {"customer", "staff"}:
            raise ValueError("Ticket messages must come from the customer or staff.")
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Ticket message cannot be empty.")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, payload FROM handovers WHERE id = ?",
                (case_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Handover case was not found.")
            if row["status"] != "pending":
                raise ValueError("Resolved handover tickets cannot receive new messages.")
            payload = json.loads(row["payload"])
            message = {
                "id": str(uuid4()),
                "role": clean_role,
                "content": clean_content,
                "created_at": utc_now(),
            }
            payload.setdefault("live_messages", []).append(message)
            connection.execute(
                "UPDATE handovers SET payload = ?, updated_at = ? WHERE id = ?",
                (json.dumps(payload), utc_now(), case_id),
            )
            return message

    def complete_handover(
        self,
        case_id: str,
        *,
        human_response: str,
        facts_carried_forward: Iterable[str] = (),
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM handovers WHERE id = ?",
                (case_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Handover case was not found.")
            payload = json.loads(row["payload"])
            facts = tuple(str(value) for value in facts_carried_forward if str(value).strip())
            cart = payload.get("cart", {})
            cart_facts = {f"{quantity} x {name}" for name, quantity in cart.items()}
            score = 1.0 if not cart_facts else len(cart_facts.intersection(facts)) / len(cart_facts)
            payload["status"] = "completed"
            payload["human_response"] = human_response.strip()
            payload["facts_carried_forward"] = list(facts)
            connection.execute(
                """UPDATE handovers SET status = 'completed', payload = ?,
                   context_carryover_score = ?, updated_at = ? WHERE id = ?""",
                (json.dumps(payload), round(score, 3), utc_now(), case_id),
            )
            return payload

    def save_knowledge_document(
        self,
        *,
        title: str,
        source_name: str,
        mime_type: str,
        text: str,
        chunks: list[Any],
    ) -> tuple[str, bool]:
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self._rows(
            "SELECT id FROM knowledge_documents WHERE source_name = ? AND checksum = ?", (source_name, checksum)
        )
        if existing:
            return existing[0]["id"], False
        document_id = chunks[0].document_id if chunks else str(uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO knowledge_documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, title, source_name, mime_type, text, checksum, utc_now()),
            )
            connection.executemany(
                "INSERT INTO knowledge_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (chunk.id, chunk.document_id, chunk.title, chunk.source_name, chunk.text, chunk.ordinal, chunk.checksum)
                    for chunk in chunks
                ],
            )
        return document_id, True

    def list_knowledge_documents(self, include_text: bool = False) -> list[dict[str, Any]]:
        text_column = ", d.text" if include_text else ""
        return self._rows(
            f"""SELECT d.id, d.title, d.source_name, d.mime_type, d.checksum, d.created_at{text_column},
                       COUNT(c.id) AS chunk_count
                FROM knowledge_documents d LEFT JOIN knowledge_chunks c ON c.document_id = d.id
                GROUP BY d.id ORDER BY d.created_at DESC"""
        )

    def list_knowledge_chunks(self) -> list[Any]:
        from .runtime.rag import KnowledgeChunk

        return [
            KnowledgeChunk(
                id=row["id"], document_id=row["document_id"], title=row["title"],
                source_name=row["source_name"], text=row["text"], ordinal=int(row["ordinal"]),
                checksum=row["checksum"],
            )
            for row in self._rows("SELECT * FROM knowledge_chunks ORDER BY document_id, ordinal")
        ]

    def delete_knowledge_document(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM knowledge_documents WHERE id = ?", (document_id,))
            return cursor.rowcount > 0

    def knowledge_generation(self) -> str:
        rows = self._rows("SELECT id, checksum FROM knowledge_chunks ORDER BY id")
        payload = "|".join(f"{row['id']}:{row['checksum']}" for row in rows)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def save_retrieval_trace(self, result: Any, *, provider_id: str, answer_generated: bool) -> str:
        trace_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO retrieval_traces VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trace_id, result.query, provider_id, int(answer_generated),
                    json.dumps(result.trace.to_dict()), json.dumps([hit.chunk_id for hit in result.hits]), utc_now(),
                ),
            )
        return trace_id

    def list_retrieval_traces(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM retrieval_traces ORDER BY created_at DESC")

    def export_orders_csv(self) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "created_at",
                "id",
                "session_id",
                "status",
                "fulfilment",
                "delivery_address",
                "currency",
                "total",
                "lines",
            ),
        )
        writer.writeheader()
        for order in self.list_orders():
            writer.writerow({**order, "lines": json.dumps(order["lines"], ensure_ascii=True)})
        return output.getvalue()

    def export_orders_json(self) -> str:
        return json.dumps(self.list_orders(), indent=2, ensure_ascii=True)

    def export_snapshot(self) -> dict[str, Any]:
        return {
            "sessions": self.list_sessions(),
            "orders": self.list_orders(),
            "handovers": self.list_handovers(),
            "knowledge_documents": self.list_knowledge_documents(),
            "retrieval_traces": self.list_retrieval_traces(),
        }

    def table_counts(self) -> dict[str, int]:
        tables = (
            "agent_sessions", "conversation_turns", "tool_traces", "conversation_signals", "orders", "handovers",
            "knowledge_documents", "knowledge_chunks", "retrieval_traces",
        )
        with self._connect() as connection:
            return {table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}

    def _rows(self, query: str, parameters: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, tuple(parameters)).fetchall()]
