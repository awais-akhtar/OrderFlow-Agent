"""Gradio customer and operations workspace for OrderFlow-Agent."""

from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Iterator
from uuid import uuid4

import gradio as gr
import uvicorn
from fastapi import FastAPI

from .agent import ConversationalTaskAgent
from .catalog import CatalogItem, JsonCatalogStore
from .evaluation.runner import DEFAULT_CONFIG, run_evaluation
from .knowledge import KnowledgeService
from .models import AgentResponse, MenuAttachment, TranscriptTurn, utc_now
from .multimodal import MenuIntelligence
from .runtime.providers import (
    ProviderRegistry,
    ProviderUnavailable,
    RuntimeSettings,
    discover_local_model_path,
)
from .runtime.streaming import GroundedStreamingResponder, StreamingReplyError
from .storage import SQLiteStorageAdapter


ROOT = Path(__file__).resolve().parent.parent
CATALOG_STORE = JsonCatalogStore()
STORAGE = SQLiteStorageAdapter()
_WORKSPACES: dict[str, "ConversationWorkspace"] = {}
_WORKSPACE_LOCK = RLock()
_LOCAL_PROVIDERS: dict[tuple[str, str, str], object] = {}
_LOCAL_PROVIDER_LOCK = RLock()


@dataclass
class ConversationWorkspace:
    id: str
    storage: SQLiteStorageAdapter
    agent: ConversationalTaskAgent
    session: Any
    settings: RuntimeSettings
    provider: object | None = None
    provider_error: str = ""
    last_response: AgentResponse | None = None
    lock: RLock | None = None

    @classmethod
    def create(cls) -> "ConversationWorkspace":
        settings = RuntimeSettings.from_env()
        agent = ConversationalTaskAgent(catalog_store=CATALOG_STORE, storage=STORAGE)
        workspace = cls(
            id=str(uuid4()),
            storage=STORAGE,
            agent=agent,
            session=agent.open_session(strictness=50),
            settings=settings,
            lock=RLock(),
        )
        workspace.configure(settings)
        return workspace

    def configure(self, settings: RuntimeSettings) -> None:
        previous = self.provider
        self.settings = settings
        self.provider = None
        self.provider_error = ""
        try:
            self.provider = _build_provider(settings)
            if self.provider is None:
                self.provider_error = (
                    "No streaming model is configured. Choose a local KernelLoom model or OpenAI in Settings."
                )
        except ProviderUnavailable as exc:
            self.provider_error = str(exc)
        if previous is not None and previous is not self.provider and previous not in _LOCAL_PROVIDERS.values():
            close = getattr(previous, "close", None)
            if callable(close):
                close()

    def reset(self, strictness: int = 50) -> None:
        self.session = self.agent.open_session(strictness=strictness)
        self.last_response = None


def _build_provider(settings: RuntimeSettings) -> object | None:
    if settings.provider_id != "kernelloom":
        return ProviderRegistry.build(settings)
    key = (
        settings.kernelloom_chat_model_path,
        settings.kernelloom_backend,
        settings.kernelloom_device,
    )
    with _LOCAL_PROVIDER_LOCK:
        if key not in _LOCAL_PROVIDERS:
            _LOCAL_PROVIDERS[key] = ProviderRegistry.build(settings)
        return _LOCAL_PROVIDERS[key]


def _workspace(identifier: str) -> ConversationWorkspace:
    with _WORKSPACE_LOCK:
        if identifier and identifier in _WORKSPACES:
            return _WORKSPACES[identifier]
        workspace = ConversationWorkspace.create()
        _WORKSPACES[workspace.id] = workspace
        return workspace


def _initialize_workspace() -> tuple[str, str, str, str, str, str]:
    workspace = _workspace("")
    return (
        workspace.id,
        _runtime_status(workspace),
        _cart_html(workspace),
        _empty_products(),
        _context_html(workspace),
        _trace_json(workspace),
    )


def _stream_turn(
    message: str,
    history: list[dict[str, Any]] | None,
    workspace_id: str,
) -> Iterator[tuple[str, list[dict[str, Any]], str, str, str, str, str]]:
    workspace = _workspace(workspace_id)
    text = (message or "").strip()
    messages = list(history or [])
    if not text:
        yield "", messages, _runtime_status(workspace), _empty_products(), _cart_html(workspace), _trace_json(workspace), _context_html(workspace)
        return
    prior = [
        ("user" if item.get("role") == "user" else "assistant", str(item.get("content", "")))
        for item in messages[-8:]
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
    ]
    messages.append({"role": "user", "content": text})
    if workspace.provider is None:
        yield (
            "",
            messages,
            _runtime_status(workspace, error=workspace.provider_error),
            _empty_products(),
            _cart_html(workspace),
            _trace_json(workspace),
            _context_html(workspace),
        )
        return
    assert workspace.lock is not None
    with workspace.lock:
        operational = workspace.agent.handle(text, workspace.session)
        workspace.last_response = operational
        messages.append({"role": "assistant", "content": ""})
        product_html = _products_html(operational.menu_attachments)
        started = time.perf_counter()
        generated = ""
        yield (
            "",
            messages,
            _runtime_status(workspace, generating=True),
            product_html,
            _cart_html(workspace),
            _trace_json(workspace, streaming=True),
            _context_html(workspace),
        )
        try:
            responder = GroundedStreamingResponder(workspace.provider, workspace.agent.catalog)
            for token in responder.stream(
                strictness=workspace.session.strictness,
                user_message=text,
                operational_response=operational,
                visible_history=prior,
            ):
                generated += token
                messages[-1] = {"role": "assistant", "content": generated}
                yield (
                    "",
                    messages,
                    _runtime_status(workspace, generating=True),
                    product_html,
                    _cart_html(workspace),
                    _trace_json(workspace, streaming=True),
                    _context_html(workspace),
                )
        except StreamingReplyError as exc:
            if not generated.strip():
                messages.pop()
                workspace.storage.discard_latest_ai_response(workspace.session.session_id)
            else:
                workspace.storage.replace_latest_ai_response(workspace.session.session_id, generated.strip())
            workspace.storage.append_latest_tool_step(
                workspace.session.session_id,
                {
                    "name": "model_stream",
                    "status": "blocked",
                    "detail": str(exc),
                    "created_at": utc_now(),
                },
            )
            yield (
                "",
                messages,
                _runtime_status(workspace, error=str(exc)),
                product_html,
                _cart_html(workspace),
                _trace_json(workspace, stream_error=str(exc)),
                _context_html(workspace),
            )
            return
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        workspace.storage.replace_latest_ai_response(workspace.session.session_id, generated.strip())
        workspace.storage.append_latest_tool_step(
            workspace.session.session_id,
            {
                "name": "model_stream",
                "status": "passed",
                "detail": f"{responder.provider_label} streamed the customer reply in {elapsed_ms} ms.",
                "created_at": utc_now(),
            },
        )
        yield (
            "",
            messages,
            _runtime_status(workspace),
            product_html,
            _cart_html(workspace),
            _trace_json(workspace, latency_ms=elapsed_ms),
            _context_html(workspace),
        )


def _quick_turn(command: str, history: list[dict[str, Any]] | None, workspace_id: str):
    yield from _stream_turn(command, history, workspace_id)


def _command_handler(command: str):
    def handler(history: list[dict[str, Any]] | None, workspace_id: str):
        yield from _quick_turn(command, history, workspace_id)

    return handler


def _new_conversation(workspace_id: str) -> tuple[list[Any], str, str, str, str, str]:
    workspace = _workspace(workspace_id)
    workspace.reset()
    return [], _runtime_status(workspace), _cart_html(workspace), _empty_products(), _context_html(workspace), _trace_json(workspace)


def _runtime_status(
    workspace: ConversationWorkspace,
    *,
    generating: bool = False,
    error: str = "",
) -> str:
    if error:
        return f'<div class="runtime-strip error"><span class="status-dot"></span><strong>Model stopped</strong><span>{html.escape(error)}</span></div>'
    if workspace.provider is None:
        detail = workspace.provider_error or "Configure a streaming model in Settings."
        return f'<div class="runtime-strip warning"><span class="status-dot"></span><strong>Model required</strong><span>{html.escape(detail)}</span></div>'
    label = html.escape(str(getattr(workspace.provider, "label", workspace.settings.provider_id)))
    state = "Streaming response" if generating else "Ready"
    css = " streaming" if generating else ""
    return (
        f'<div class="runtime-strip{css}"><span class="status-dot"></span>'
        f'<strong>{state}</strong><span>{label}</span><span>{workspace.session.agent_mode.value.upper()}</span></div>'
    )


def _cart_html(workspace: ConversationWorkspace) -> str:
    catalog = workspace.agent.catalog
    rows = []
    total = 0
    for name, quantity in workspace.session.order.items():
        item = catalog.by_name.get(name)
        if item is None:
            continue
        line_total = item.price * quantity
        total += line_total
        rows.append(
            '<div class="cart-line">'
            f'<div><strong>{quantity} x {html.escape(name)}</strong><small>PKR {item.price:,} each</small></div>'
            f'<span>PKR {line_total:,}</span></div>'
        )
    body = "".join(rows) or '<div class="empty-state">Your draft is empty.</div>'
    pending = html.escape(workspace.session.pending_action.replace("_", " ").title())
    lock = "Human handover pending" if workspace.session.handover_active else "Ordering active"
    return (
        '<section class="cart-panel"><div class="panel-heading"><div><strong>Current order</strong>'
        f'<small>{html.escape(lock)}</small></div><span>{pending}</span></div>'
        f'<div class="cart-body">{body}</div>'
        f'<div class="cart-total"><span>Total</span><strong>PKR {total:,}</strong></div></section>'
    )


def _products_html(attachments: tuple[MenuAttachment, ...]) -> str:
    if not attachments:
        return _empty_products()
    groups: dict[tuple[str, tuple[str, ...], str], list[MenuAttachment]] = {}
    for item in attachments:
        key = (item.image, item.ingredients, item.description)
        groups.setdefault(key, []).append(item)
    cards = []
    for (image_path, ingredients, description), items in groups.items():
        variants = "".join(
            f'<span><strong>{html.escape(item.title)}</strong> PKR {item.price:,}</span>' for item in items
        )
        image = _image_data_url(image_path)
        image_markup = f'<img src="{image}" alt="{html.escape(items[0].title)}">' if image else '<div class="image-missing">No image</div>'
        ingredient_text = ", ".join(ingredients) or "Ingredients are not listed."
        cards.append(
            '<article class="menu-card">'
            f'{image_markup}<div class="menu-copy"><div class="variant-list">{variants}</div>'
            f'<p>{html.escape(description)}</p><small>Ingredients: {html.escape(ingredient_text)}</small></div></article>'
        )
    return '<section class="response-products"><div class="products-label">Items in this reply</div><div class="product-row">' + "".join(cards) + "</div></section>"


def _empty_products() -> str:
    return '<section class="response-products empty-products"><div class="products-label">Items in this reply</div><span>Product images and ingredients appear here when a menu item is relevant.</span></section>'


def _image_data_url(relative_path: str) -> str:
    if not relative_path:
        return ""
    candidate = (ROOT / relative_path).resolve()
    root = ROOT.resolve()
    if root not in candidate.parents or not candidate.is_file():
        return ""
    mime = mimetypes.guess_type(candidate.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(candidate.read_bytes()).decode('ascii')}"


def _trace_json(
    workspace: ConversationWorkspace,
    *,
    streaming: bool = False,
    stream_error: str = "",
    latency_ms: float | None = None,
) -> str:
    response = workspace.last_response
    payload: dict[str, Any] = {
        "session_id": workspace.session.session_id,
        "mode": workspace.session.agent_mode.value,
        "stream": {
            "provider": getattr(workspace.provider, "provider_id", "none"),
            "state": "blocked" if stream_error else "streaming" if streaming else "ready",
            "latency_ms": latency_ms,
            "error": stream_error or None,
        },
        "tools": [asdict(step) for step in response.tool_trace] if response else [],
    }
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _context_html(workspace: ConversationWorkspace) -> str:
    response = workspace.last_response
    signal = response.conversation_signal if response else None
    if signal is None:
        return '<div class="context-line"><strong>Conversation signal</strong><span>Not enough interaction yet</span></div>'
    return (
        '<div class="context-line"><strong>Conversation signal</strong>'
        f'<span>{html.escape(signal.label.title())} ({signal.confidence:.0%})</span>'
        f'<small>{html.escape(signal.evidence)}</small></div>'
    )


def _order_rows() -> list[list[Any]]:
    rows = []
    for order in STORAGE.list_orders():
        items = ", ".join(f"{line['quantity']} x {line['item']}" for line in order["lines"])
        rows.append([order["id"][:8].upper(), order["status"], items, order["currency"], order["total"], order["created_at"]])
    return rows


def _refresh_orders() -> tuple[list[list[Any]], str]:
    rows = _order_rows()
    return rows, f"{len(rows)} persisted order{'s' if len(rows) != 1 else ''}"


def _export_orders() -> tuple[str, str]:
    output = ROOT / "data" / "exports"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "order-history.csv"
    json_path = output / "order-history.json"
    csv_path.write_text(STORAGE.export_orders_csv(), encoding="utf-8")
    json_path.write_text(STORAGE.export_orders_json(), encoding="utf-8")
    return str(csv_path), str(json_path)


def _handover_rows() -> list[list[Any]]:
    return [
        [
            row["id"][:8].upper(),
            row["status"],
            row["handover"]["decision"]["trigger"],
            row["handover"].get("customer_request", ""),
            row["context_carryover_score"],
            row["created_at"],
        ]
        for row in STORAGE.list_handovers()
    ]


def _refresh_handovers() -> tuple[list[list[Any]], gr.Dropdown]:
    cases = STORAGE.list_handovers()
    choices = [(f"{row['id'][:8].upper()} - {row['status']}", row["id"]) for row in cases]
    return _handover_rows(), gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def _handover_detail(case_id: str | None) -> tuple[str, str]:
    row = next((item for item in STORAGE.list_handovers() if item["id"] == case_id), None)
    if row is None:
        return "No handover selected.", ""
    payload = row["handover"]
    cart_facts = "\n".join(f"{quantity} x {name}" for name, quantity in payload.get("cart", {}).items())
    return payload.get("summary", ""), cart_facts


def _complete_handover(
    case_id: str | None,
    human_response: str,
    facts_text: str,
    workspace_id: str,
    history: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]], list[list[Any]]]:
    if not case_id:
        return "Select a handover case.", list(history or []), _handover_rows()
    if not human_response.strip():
        return "Enter the staff response before completing the handover.", list(history or []), _handover_rows()
    facts = tuple(line.strip() for line in facts_text.splitlines() if line.strip())
    payload = STORAGE.complete_handover(
        case_id,
        human_response=human_response,
        facts_carried_forward=facts,
    )
    workspace = _workspace(workspace_id)
    if workspace.session.handover_case_id == case_id:
        workspace.session.handover_active = False
        workspace.session.handover_case_id = ""
        workspace.session.pending_action = "none"
        workspace.storage.ensure_session(workspace.session)
        workspace.storage.append_turn(
            TranscriptTurn(workspace.session.session_id, "human", human_response.strip())
        )
    messages = list(history or [])
    messages.append({"role": "assistant", "content": human_response.strip(), "metadata": {"title": "Human employee"}})
    score = 1.0
    cart = payload.get("cart", {})
    if cart:
        score = len(set(facts).intersection({f"{q} x {n}" for n, q in cart.items()})) / len(cart)
    return f"Handover completed. Cart context carried forward {score:.0%}.", messages, _handover_rows()


def _menu_recommend(query: str, image_path: str | None) -> tuple[list[list[Any]], str]:
    image_bytes = Path(image_path).read_bytes() if image_path else None
    recommendations = MenuIntelligence(CATALOG_STORE.load(), asset_root=ROOT).recommend(
        query.strip() or "pizza",
        query_image=image_bytes,
        limit=6,
    )
    rows = [
        [item.item.name, item.item.price, item.score, item.text_score, item.image_score, item.reason]
        for item in recommendations
    ]
    attachments = tuple(
        MenuAttachment(
            item.item.sku,
            item.item.name,
            item.item.description,
            item.item.ingredients,
            item.item.image,
            item.item.price,
            CATALOG_STORE.load().currency,
        )
        for item in recommendations[:3]
    )
    return rows, _products_html(attachments)


def _run_scenarios() -> tuple[list[list[Any]], str, str]:
    result = run_evaluation(DEFAULT_CONFIG)
    rows = [
        [
            item.agent_mode,
            item.scenario_id,
            item.passed,
            item.metrics.turns,
            item.metrics.tool_calls,
            item.metrics.handovers,
            item.metrics.validation_failures,
        ]
        for item in result["results"]
    ]
    passed = sum(item.passed for item in result["results"])
    summary = f"{passed}/{len(rows)} mode/scenario runs passed their configured software expectations."
    return rows, summary, result["run_directory"]


def _knowledge_rows() -> list[list[Any]]:
    return [
        [row["title"], row["source_name"], row["mime_type"], row["created_at"]]
        for row in STORAGE.list_knowledge_documents()
    ]


def _ingest_knowledge(file_path: str | None, title: str) -> tuple[str, list[list[Any]]]:
    if not file_path:
        return "Choose a text, Markdown, JSON, or PDF file.", _knowledge_rows()
    path = Path(file_path)
    service = KnowledgeService(STORAGE)
    _, created, chunks = service.ingest(
        path.name,
        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        path.read_bytes(),
        title=title,
    )
    state = "Added" if created else "Already indexed"
    return f"{state}: {path.name} ({chunks} chunks).", _knowledge_rows()


def _query_knowledge(query: str) -> str:
    result = KnowledgeService(STORAGE).answer(query.strip()) if query.strip() else None
    return result[0] if result else "No grounded evidence matched that query."


def _catalog_rows() -> list[list[Any]]:
    return [
        [item.sku, item.name, item.category, item.price, item.active, ", ".join(item.ingredients), item.image]
        for item in CATALOG_STORE.load().items
    ]


def _catalog_choices() -> list[tuple[str, str]]:
    return [(f"{item.name} ({item.sku})", item.sku) for item in CATALOG_STORE.load().items]


def _load_catalog_item(sku: str | None) -> tuple[Any, ...]:
    item = next((row for row in CATALOG_STORE.load().items if row.sku == sku), None)
    if item is None:
        return "", "", "Pizza", 0, "", "", "", "", True
    return (
        item.sku,
        item.name,
        item.category,
        item.price,
        item.description,
        ", ".join(item.ingredients),
        ", ".join(item.tags),
        item.image,
        item.active,
    )


def _save_catalog_item(
    sku: str,
    name: str,
    category: str,
    price: float,
    description: str,
    ingredients: str,
    tags: str,
    image: str,
    active: bool,
) -> tuple[str, list[list[Any]], gr.Dropdown]:
    clean_sku = sku.strip()
    clean_name = name.strip()
    if not clean_sku or not clean_name or not category.strip():
        return "SKU, name, and category are required.", _catalog_rows(), gr.Dropdown(choices=_catalog_choices())
    catalog = CATALOG_STORE.load()
    existing = next((item for item in catalog.items if item.sku == clean_sku), None)
    item = CatalogItem(
        sku=clean_sku,
        name=clean_name,
        category=category.strip(),
        price=max(0, int(price)),
        aliases=existing.aliases if existing else (),
        description=description.strip(),
        tags=tuple(value.strip() for value in tags.split(",") if value.strip()),
        ingredients=tuple(value.strip() for value in ingredients.split(",") if value.strip()),
        image=image.strip(),
        interaction_stats=existing.interaction_stats if existing else {},
        active=bool(active),
        title=existing.title if existing else clean_name,
    )
    CATALOG_STORE.upsert(item)
    return "Catalog item saved.", _catalog_rows(), gr.Dropdown(choices=_catalog_choices(), value=item.sku)


def _save_settings(
    provider_choice: str,
    api_key: str,
    model_id: str,
    local_model_path: str,
    device: str,
    workspace_id: str,
) -> tuple[str, str]:
    workspace = _workspace(workspace_id)
    provider_map = {
        "KernelLoom local model": "kernelloom",
        "OpenAI API": "openai",
        "Hugging Face endpoint": "huggingface",
        "OpenAgent service": "openagent",
    }
    provider_id = provider_map.get(provider_choice, "disabled")
    settings = replace(
        workspace.settings,
        provider_id=provider_id,
        api_key=api_key.strip(),
        response_model=model_id.strip() or ("orderflow-local" if provider_id == "kernelloom" else ""),
        kernelloom_transport="python",
        kernelloom_chat_model_path=local_model_path.strip(),
        kernelloom_backend="openvino",
        kernelloom_device=device.strip().upper() or "CPU",
    )
    workspace.configure(settings)
    return _runtime_status(workspace, error=workspace.provider_error if workspace.provider is None else ""), json.dumps(settings.safe_summary(), indent=2)


def _check_provider(workspace_id: str) -> str:
    workspace = _workspace(workspace_id)
    if workspace.provider is None:
        return workspace.provider_error or "No provider configured."
    status = workspace.provider.check()
    return f"{status.label}: {status.detail}"


CSS = """
:root { color-scheme:light; --ink:#172026; --muted:#66717a; --line:#d9dfe2; --paper:#f4f6f5; --white:#fff; --green:#176b52; --tomato:#b94a3a; --gold:#a16c17; --body-text-color:#172026; --body-text-color-subdued:#66717a; --body-background-fill:#f4f6f5; --background-fill-primary:#fff; --background-fill-secondary:#f7f8f8; --block-background-fill:#fff; --block-label-text-color:#3f4b52; --input-background-fill:#fff; }
body, .gradio-container { background:var(--paper)!important; color:var(--ink)!important; font-family:Inter,Segoe UI,Arial,sans-serif!important; letter-spacing:0!important; }
.gradio-container { max-width:none!important; padding:0!important; }
.app-header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:16px 24px; background:#fff; border-bottom:1px solid var(--line); color:var(--ink)!important; }
.app-header *, .cart-panel *, .response-products *, .context-line * { color:inherit; }
.brand-block strong { display:block; font-size:20px; line-height:1.1; }.brand-block span { color:var(--muted); font-size:12px; }
.header-mark { width:34px; height:34px; border:2px solid var(--tomato); border-radius:50%; display:grid; place-items:center; color:var(--tomato); font-weight:800; }
.app-inner { padding:16px 24px 28px; }
.runtime-strip { display:flex; align-items:center; flex-wrap:wrap; gap:10px; min-height:38px; padding:8px 12px; background:#fff; border:1px solid var(--line); border-left:3px solid var(--green); border-radius:5px; font-size:12px; color:var(--ink)!important; }
.runtime-strip strong { color:var(--ink)!important; }
.runtime-strip span:not(.status-dot) { color:var(--muted); }.runtime-strip.warning { border-left-color:var(--gold); }.runtime-strip.error { border-left-color:var(--tomato); }.runtime-strip.error strong { color:var(--tomato); }
.status-dot { width:8px; height:8px; border-radius:50%; background:var(--green); }.warning .status-dot { background:var(--gold); }.error .status-dot { background:var(--tomato); }
.streaming .status-dot { animation:pulse 1s ease-in-out infinite; } @keyframes pulse { 50% { opacity:.28; transform:scale(.76); } }
.customer-grid { align-items:stretch; gap:14px!important; }.chat-column, .side-column { min-width:0!important; }
.chat-shell, .cart-panel, .response-products, .ops-surface { background:#fff; border:1px solid var(--line); border-radius:6px; overflow:hidden; color:var(--ink)!important; }
#customer-chat { border:0!important; box-shadow:none!important; }.chatbot { background:#fbfcfc!important; }
#customer-chat .bubble-wrap, #customer-chat [role="log"] { background:#fbfcfc!important; color:var(--ink)!important; }
#customer-chat .placeholder-content, #customer-chat .placeholder-content * { color:var(--muted)!important; background:transparent!important; }
#customer-chat .message { border-radius:6px!important; box-shadow:none!important; max-width:78%!important; opacity:1!important; animation:message-in .15s ease-out; } @keyframes message-in { from { opacity:.35; transform:translateY(3px); } }
#customer-chat .message *, #customer-chat .message p, #customer-chat .message span { color:var(--ink)!important; opacity:1!important; }
#customer-chat .message.user { background:#e8f1ee!important; color:var(--ink)!important; } #customer-chat .message.bot { background:#fff!important; border:1px solid #e2e6e8!important; }
.composer-row { padding:10px; border-top:1px solid var(--line); align-items:end!important; }.composer-row button { min-width:88px!important; }
#customer-composer, #customer-composer label, #customer-composer .input-container, #customer-composer textarea { background:#fff!important; color:var(--ink)!important; opacity:1!important; }
#customer-composer { border:1px solid var(--line)!important; box-shadow:none!important; }
#customer-composer textarea::placeholder { color:#778188!important; opacity:1!important; }
.quick-row { padding:0 10px 10px; gap:6px!important; }.quick-row button { font-size:12px!important; min-height:32px!important; }
.response-products { margin-top:10px; padding:12px; }.products-label { font-size:11px; font-weight:750; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
.empty-products { color:var(--muted); font-size:12px; }.product-row { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; }.menu-card { border:1px solid var(--line); border-radius:5px; overflow:hidden; animation:product-in .22s ease-out; } @keyframes product-in { from { opacity:0; transform:translateY(4px); } }
.menu-card img, .image-missing { width:100%; aspect-ratio:16/9; object-fit:cover; background:#eef0ef; }.image-missing { display:grid; place-items:center; color:var(--muted); font-size:11px; }
.menu-copy { padding:8px; }.variant-list { display:flex; flex-direction:column; gap:2px; font-size:12px; }.variant-list span { display:flex; justify-content:space-between; gap:8px; }.menu-copy p { margin:5px 0; color:#3f4b52; font-size:11px; line-height:1.35; }.menu-copy small { display:block; color:var(--muted); font-size:10px; line-height:1.35; }
.cart-panel { min-height:245px; }.panel-heading { padding:12px 14px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:10px; }.panel-heading strong,.panel-heading small { display:block; }.panel-heading small { color:var(--muted); font-size:10px; margin-top:2px; }.panel-heading>span { font-size:10px; color:var(--gold); }
.cart-body { padding:4px 14px; min-height:124px; }.cart-line { display:flex; justify-content:space-between; gap:10px; padding:10px 0; border-bottom:1px solid #eceff0; font-size:12px; }.cart-line small { display:block; color:var(--muted); font-size:10px; margin-top:2px; }.cart-line>span { white-space:nowrap; }.cart-total { display:flex; justify-content:space-between; padding:12px 14px; border-top:1px solid var(--line); }.cart-total strong { font-size:18px; }.empty-state { color:var(--muted); font-size:12px; padding:20px 0; }
.context-line { margin-top:10px; padding:11px 12px; background:#fff; border:1px solid var(--line); border-radius:5px; display:grid; grid-template-columns:1fr auto; gap:4px 10px; font-size:12px; color:var(--ink)!important; }.context-line > span,.context-line small { color:var(--muted)!important; }.context-line small { grid-column:1/-1; line-height:1.35; }
.tabs { border-top:1px solid var(--line)!important; color:var(--ink)!important; }.tab-nav button, [role="tab"] { letter-spacing:0!important; color:#46525a!important; }.ops-surface { padding:14px; }.ops-surface, .ops-surface * { color:var(--ink); }.section-note { color:var(--muted)!important; font-size:12px; margin:0 0 10px; }
button.primary { background:var(--green)!important; border-color:var(--green)!important; } button.secondary { background:#fff!important; color:var(--ink)!important; border:1px solid #bcc5c9!important; }
@media (max-width:900px) { .app-header,.app-inner { padding-left:12px; padding-right:12px; }.customer-grid { flex-direction:column!important; }.product-row { grid-template-columns:1fr; } #customer-chat .message { max-width:92%!important; } }
"""


def build_demo() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.red,
        neutral_hue=gr.themes.colors.gray,
        radius_size=gr.themes.sizes.radius_sm,
    )
    with gr.Blocks(theme=theme, css=CSS, title="OrderFlow-Agent", analytics_enabled=False) as demo:
        workspace_state = gr.State("")
        gr.HTML('<header class="app-header"><div style="display:flex;align-items:center;gap:10px"><div class="header-mark">OF</div><div class="brand-block"><strong>OrderFlow-Agent</strong><span>Pizza ordering with deterministic execution</span></div></div><div class="brand-block" style="text-align:right"><strong style="font-size:13px">Customer service</strong><span>Catalogue prices and confirmation gates stay authoritative</span></div></header>')
        with gr.Column(elem_classes="app-inner"):
            runtime_status = gr.HTML()
            with gr.Tabs(elem_classes="tabs"):
                with gr.Tab("Customer"):
                    with gr.Row(elem_classes="customer-grid"):
                        with gr.Column(scale=7, elem_classes="chat-column"):
                            with gr.Column(elem_classes="chat-shell"):
                                chatbot = gr.Chatbot(
                                    type="messages",
                                    label="Order conversation",
                                    show_label=False,
                                    height=330,
                                    layout="bubble",
                                    elem_id="customer-chat",
                                    elem_classes="chatbot",
                                    placeholder="No messages yet",
                                    show_copy_button=True,
                                )
                                with gr.Row(elem_classes="composer-row"):
                                    message = gr.Textbox(
                                        placeholder="Message OrderFlow...",
                                        show_label=False,
                                        lines=2,
                                        max_lines=5,
                                        scale=8,
                                        elem_id="customer-composer",
                                    )
                                    send = gr.Button("Send", variant="primary", scale=1)
                                with gr.Row(elem_classes="quick-row"):
                                    menu_button = gr.Button("Pizza menu", variant="secondary", size="sm")
                                    bill_button = gr.Button("Show bill", variant="secondary", size="sm")
                                    confirm_button = gr.Button("Confirm order", variant="secondary", size="sm")
                                    human_button = gr.Button("Request human", variant="secondary", size="sm")
                                    new_button = gr.Button("New conversation", variant="secondary", size="sm")
                            products = gr.HTML()
                        with gr.Column(scale=3, elem_classes="side-column"):
                            cart = gr.HTML()
                            context = gr.HTML()
                            gr.Markdown("#### Tool trace")
                            trace = gr.Code(language="json", interactive=False, lines=15, show_label=False)

                with gr.Tab("Orders"):
                    with gr.Column(elem_classes="ops-surface"):
                        gr.Markdown("### Order history")
                        order_count = gr.Markdown("0 persisted orders")
                        orders = gr.Dataframe(
                            headers=["Reference", "Status", "Items", "Currency", "Total", "Created"],
                            datatype=["str", "str", "str", "str", "number", "str"],
                            interactive=False,
                        )
                        with gr.Row():
                            refresh_orders = gr.Button("Refresh", variant="secondary")
                            export_orders = gr.Button("Prepare exports", variant="primary")
                        with gr.Row():
                            order_csv = gr.File(label="CSV export", interactive=False)
                            order_json = gr.File(label="JSON export", interactive=False)

                with gr.Tab("Handovers"):
                    with gr.Column(elem_classes="ops-surface"):
                        gr.Markdown("### Human handover queue")
                        handovers = gr.Dataframe(
                            headers=["Case", "Status", "Trigger", "Customer request", "Carryover", "Created"],
                            interactive=False,
                        )
                        refresh_handovers = gr.Button("Refresh queue", variant="secondary")
                        handover_select = gr.Dropdown(label="Handover case", choices=[])
                        handover_summary = gr.Textbox(label="Structured summary", lines=9, interactive=False)
                        handover_facts = gr.Textbox(label="Cart facts carried forward", lines=3)
                        human_response = gr.Textbox(label="Human response", lines=3)
                        complete_handover = gr.Button("Complete handover", variant="primary")
                        handover_status = gr.Markdown()

                with gr.Tab("Menu intelligence"):
                    with gr.Column(elem_classes="ops-surface"):
                        gr.Markdown("### Text and image menu ranking")
                        menu_query = gr.Textbox(label="Request", value="something like spicy chicken pizza but vegetarian")
                        menu_image = gr.Image(label="Optional reference image", type="filepath", height=180)
                        recommend = gr.Button("Rank menu items", variant="primary")
                        menu_results = gr.Dataframe(
                            headers=["Item", "Price", "Score", "Text", "Image", "Reason"],
                            interactive=False,
                        )
                        menu_products = gr.HTML()

                with gr.Tab("Evaluation"):
                    with gr.Column(elem_classes="ops-surface"):
                        gr.Markdown("### Agent behaviour evaluation")
                        gr.Markdown("The runner replays the configured pizza-order scenarios in CONTROLLED, ASSISTED, and FLEXIBLE modes. Results are software measurements, not claims about people.", elem_classes="section-note")
                        run_scenarios = gr.Button("Run scenario suite", variant="primary")
                        evaluation_summary = gr.Markdown()
                        evaluation_path = gr.Textbox(label="Saved run directory", interactive=False)
                        evaluation_results = gr.Dataframe(
                            headers=["Mode", "Scenario", "Passed", "Turns", "Tools", "Handovers", "Validation failures"],
                            interactive=False,
                        )

                with gr.Tab("Knowledge"):
                    with gr.Column(elem_classes="ops-surface"):
                        gr.Markdown("### Grounded operating knowledge")
                        with gr.Row():
                            knowledge_file = gr.File(label="Document", type="filepath")
                            knowledge_title = gr.Textbox(label="Title")
                        ingest = gr.Button("Add to knowledge index", variant="primary")
                        ingest_status = gr.Markdown()
                        knowledge_docs = gr.Dataframe(headers=["Title", "Source", "Type", "Created"], interactive=False)
                        knowledge_query = gr.Textbox(label="Evidence query")
                        query_knowledge = gr.Button("Retrieve evidence", variant="secondary")
                        knowledge_answer = gr.Markdown()

                with gr.Tab("Catalogue"):
                    with gr.Column(elem_classes="ops-surface"):
                        gr.Markdown("### Catalogue editor")
                        catalog_table = gr.Dataframe(
                            headers=["SKU", "Name", "Category", "Price", "Active", "Ingredients", "Image"],
                            interactive=False,
                        )
                        catalog_select = gr.Dropdown(label="Edit item", choices=_catalog_choices())
                        with gr.Row():
                            catalog_sku = gr.Textbox(label="SKU")
                            catalog_name = gr.Textbox(label="Name")
                            catalog_category = gr.Dropdown(label="Category", choices=["Pizza", "Sides", "Dips", "Drinks", "Desserts", "Deals"], allow_custom_value=True)
                            catalog_price = gr.Number(label="Price", precision=0)
                        catalog_description = gr.Textbox(label="Description")
                        catalog_ingredients = gr.Textbox(label="Ingredients, comma separated")
                        catalog_tags = gr.Textbox(label="Tags, comma separated")
                        catalog_image = gr.Textbox(label="Project-relative image path")
                        catalog_active = gr.Checkbox(label="Active", value=True)
                        save_catalog = gr.Button("Save catalogue item", variant="primary")
                        catalog_status = gr.Markdown()

                with gr.Tab("Settings"):
                    with gr.Column(elem_classes="ops-surface"):
                        gr.Markdown("### Model runtime")
                        provider_choice = gr.Dropdown(
                            label="Provider",
                            choices=["KernelLoom local model", "OpenAI API", "Hugging Face endpoint", "OpenAgent service"],
                            value="KernelLoom local model" if discover_local_model_path() else "OpenAI API",
                        )
                        api_key = gr.Textbox(label="API key or endpoint token", type="password")
                        model_id = gr.Textbox(label="Model ID", value="orderflow-local" if discover_local_model_path() else "")
                        local_model_path = gr.Textbox(label="Local model path", value=discover_local_model_path())
                        device = gr.Dropdown(label="Local device", choices=["CPU", "GPU", "NPU", "AUTO"], value="CPU")
                        with gr.Row():
                            save_settings = gr.Button("Apply runtime", variant="primary")
                            check_provider = gr.Button("Check runtime", variant="secondary")
                        provider_check = gr.Markdown()
                        safe_settings = gr.Code(language="json", interactive=False, label="Non-secret runtime summary")

        turn_outputs = [message, chatbot, runtime_status, products, cart, trace, context]
        message.submit(_stream_turn, [message, chatbot, workspace_state], turn_outputs)
        send.click(_stream_turn, [message, chatbot, workspace_state], turn_outputs)
        for button, command in (
            (menu_button, "Show me the pizza menu"),
            (bill_button, "Show my current bill"),
            (confirm_button, "Confirm order"),
            (human_button, "I would like to speak with restaurant staff"),
        ):
            button.click(
                _command_handler(command),
                [chatbot, workspace_state],
                turn_outputs,
            )
        new_button.click(
            _new_conversation,
            [workspace_state],
            [chatbot, runtime_status, cart, products, context, trace],
        )
        refresh_orders.click(_refresh_orders, outputs=[orders, order_count])
        export_orders.click(_export_orders, outputs=[order_csv, order_json])
        refresh_handovers.click(_refresh_handovers, outputs=[handovers, handover_select])
        handover_select.change(_handover_detail, [handover_select], [handover_summary, handover_facts])
        complete_handover.click(
            _complete_handover,
            [handover_select, human_response, handover_facts, workspace_state, chatbot],
            [handover_status, chatbot, handovers],
        )
        recommend.click(_menu_recommend, [menu_query, menu_image], [menu_results, menu_products])
        run_scenarios.click(_run_scenarios, outputs=[evaluation_results, evaluation_summary, evaluation_path])
        ingest.click(_ingest_knowledge, [knowledge_file, knowledge_title], [ingest_status, knowledge_docs])
        query_knowledge.click(_query_knowledge, [knowledge_query], [knowledge_answer])
        catalog_select.change(
            _load_catalog_item,
            [catalog_select],
            [catalog_sku, catalog_name, catalog_category, catalog_price, catalog_description, catalog_ingredients, catalog_tags, catalog_image, catalog_active],
        )
        save_catalog.click(
            _save_catalog_item,
            [catalog_sku, catalog_name, catalog_category, catalog_price, catalog_description, catalog_ingredients, catalog_tags, catalog_image, catalog_active],
            [catalog_status, catalog_table, catalog_select],
        )
        save_settings.click(
            _save_settings,
            [provider_choice, api_key, model_id, local_model_path, device, workspace_state],
            [runtime_status, safe_settings],
        )
        check_provider.click(_check_provider, [workspace_state], [provider_check])
        demo.load(
            _initialize_workspace,
            outputs=[workspace_state, runtime_status, cart, products, context, trace],
        ).then(_refresh_orders, outputs=[orders, order_count]).then(
            _refresh_handovers, outputs=[handovers, handover_select]
        ).then(lambda: _catalog_rows(), outputs=[catalog_table]).then(
            lambda: _knowledge_rows(), outputs=[knowledge_docs]
        )
    return demo


api = FastAPI(title="OrderFlow-Agent", version="1.1.0")


@api.get("/api/health")
def api_health() -> dict[str, Any]:
    settings = RuntimeSettings.from_env()
    return {
        "status": "ok",
        "application": "OrderFlow-Agent",
        "provider": settings.provider_id,
        "local_model_configured": bool(settings.kernelloom_chat_model_path),
    }


@api.get("/api/orders")
def api_orders() -> dict[str, Any]:
    orders = STORAGE.list_orders()
    return {"count": len(orders), "orders": orders}


@api.get("/api/handovers")
def api_handovers() -> dict[str, Any]:
    handovers = STORAGE.list_handovers()
    return {"count": len(handovers), "handovers": handovers}


@api.get("/api/sessions/{session_id}/context")
def api_session_context(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "signals": STORAGE.list_signals(session_id),
        "turns": [asdict(turn) for turn in STORAGE.list_turns(session_id)],
        "tool_traces": STORAGE.list_tool_traces(session_id),
    }


demo = build_demo()
app = gr.mount_gradio_app(
    api,
    demo,
    path="/",
    show_api=False,
    allowed_paths=[str(ROOT / "data" / "menu_images"), str(ROOT / "data" / "exports")],
    max_file_size="15mb",
)


def run_app() -> None:
    host = os.getenv("ORDERFLOW_HOST", "127.0.0.1")
    port = int(os.getenv("ORDERFLOW_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["app", "build_demo", "run_app"]
