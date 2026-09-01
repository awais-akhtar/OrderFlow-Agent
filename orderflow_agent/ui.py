"""Professional NiceGUI workspace for OrderFlow-Agent."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from nicegui import app, events, run, ui
from PIL import Image, UnidentifiedImageError

from .agent import ConversationalTaskAgent
from .catalog import CatalogItem, JsonCatalogStore
from .context.models import ConversationSignal
from .evaluation.runner import run_evaluation
from .handover.models import HandoverCase, HandoverDecision
from .knowledge import KnowledgeService, RAGError
from .models import AgentResponse, ConversationSession, ToolStep, TranscriptTurn
from .multimodal import (
    LightweightMenuEncoder,
    MenuIntelligence,
    MenuInterestModel,
    ProviderMenuEncoder,
)
from .policy import PromptPolicyCompiler
from .runtime.analysis import condition_summary
from .runtime.providers import (
    ProviderRegistry,
    ProviderUnavailable,
    RuntimeSettings,
    kernelloom_package_version,
)
from .storage import SQLiteStorageAdapter


ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_PATH = ROOT / "data" / "knowledge_base.md"
MENU_IMAGE_DIR = ROOT / "data" / "menu_images"

if MENU_IMAGE_DIR.exists():
    app.add_static_files("/menu-images", str(MENU_IMAGE_DIR))


def _menu_image_url(image_path: str) -> str | None:
    if not image_path:
        return None
    candidate = (ROOT / image_path).resolve()
    try:
        relative = candidate.relative_to(MENU_IMAGE_DIR.resolve())
    except ValueError:
        return None
    return f"/menu-images/{relative.as_posix()}" if candidate.is_file() else None


def _stream_chunks(content: str, words_per_chunk: int = 4) -> tuple[str, ...]:
    """Split validated copy into display chunks without changing its text."""

    if not content:
        return ()
    tokens = re.findall(r"\S+\s*", content)
    if not tokens:
        return (content,)
    width = max(1, int(words_per_chunk))
    return tuple("".join(tokens[index : index + width]) for index in range(0, len(tokens), width))


@dataclass
class WorkspaceState:
    storage: SQLiteStorageAdapter = field(default_factory=SQLiteStorageAdapter)
    catalog_store: JsonCatalogStore = field(default_factory=JsonCatalogStore)
    settings: RuntimeSettings = field(default_factory=RuntimeSettings.from_env)
    provider: object | None = None
    knowledge: KnowledgeService | None = None
    agent: ConversationalTaskAgent | None = None
    session: ConversationSession | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    trace_history: list[tuple[str, tuple[ToolStep, ...]]] = field(default_factory=list)
    menu_query_image: bytes | None = None
    menu_query_image_name: str = ""
    menu_recommendations: list[Any] = field(default_factory=list)
    interest_model: MenuInterestModel | None = None
    interest_report: Any | None = None
    evaluation_result: dict[str, Any] | None = None
    tts_enabled: bool = False
    provider_error: str = ""
    response_in_progress: bool = False

    def initialize(self) -> None:
        self.provider = self._build_provider(self.settings)
        self.knowledge = KnowledgeService(self.storage, self.provider)
        self.knowledge.seed(KNOWLEDGE_PATH)
        self.agent = ConversationalTaskAgent(
            catalog_store=self.catalog_store,
            storage=self.storage,
            provider=self.provider,
            knowledge_responder=self.knowledge.answer,
        )
        self.session = self.agent.open_session(strictness=50)
        welcome = self.agent.welcome(self.session)
        self.add_response(welcome)

    def reconfigure(self, settings: RuntimeSettings) -> None:
        provider = ProviderRegistry.build(settings)
        assert self.session is not None
        knowledge = KnowledgeService(self.storage, provider)
        agent = ConversationalTaskAgent(
            catalog_store=self.catalog_store,
            storage=self.storage,
            provider=provider,
            knowledge_responder=knowledge.answer,
        )
        previous = self.provider
        self.settings = settings
        self.provider = provider
        self.knowledge = knowledge
        self.agent = agent
        self.provider_error = ""
        self._close_provider(previous)

    def new_session(self, strictness: int) -> None:
        assert self.agent is not None
        self.session = self.agent.open_session(strictness=strictness)
        self.messages.clear()
        self.trace_history.clear()
        self.add_response(self.agent.welcome(self.session))

    def menu_intelligence(self) -> MenuIntelligence:
        capabilities = getattr(self.provider, "capabilities", None)
        backend = (
            ProviderMenuEncoder(self.provider)
            if self.provider is not None and getattr(capabilities, "embeddings", False)
            else LightweightMenuEncoder()
        )
        return MenuIntelligence(self.catalog_store.load(), backend=backend, asset_root=ROOT)

    def add_user(self, content: str, channel: str = "text") -> None:
        self.messages.append(
            {"role": "customer", "content": content, "channel": channel, "stamp": datetime.now().strftime("%H:%M")}
        )

    def add_response(self, response: AgentResponse, *, progressive: bool = False) -> dict[str, Any]:
        message = {
            "role": "ai",
            "content": "" if progressive else response.content,
            "channel": "text",
            "stamp": datetime.now().strftime("%H:%M"),
            "streaming": progressive,
            "attachments": [asdict(attachment) for attachment in response.menu_attachments],
        }
        self.messages.append(message)
        self.trace_history.append((response.content.splitlines()[0][:72], response.tool_trace))
        return message

    def close(self) -> None:
        self._close_provider(self.provider)
        self.provider = None

    def _build_provider(self, settings: RuntimeSettings) -> object | None:
        try:
            return ProviderRegistry.build(settings)
        except ProviderUnavailable as exc:
            self.provider_error = str(exc)
            return None

    @staticmethod
    def _close_provider(provider: object | None) -> None:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


def _provider_label(state: WorkspaceState) -> str:
    if state.provider is None:
        return "Deterministic runtime"
    return str(getattr(state.provider, "label", state.settings.provider_id))


def _status_color(status: str) -> str:
    return {"passed": "positive", "blocked": "negative", "fallback": "warning", "info": "grey-7"}.get(status, "grey-7")


@app.get("/api/orders")
def api_orders() -> dict[str, Any]:
    orders = SQLiteStorageAdapter().list_orders()
    return {"count": len(orders), "orders": orders}


@app.get("/api/handovers")
def api_handovers() -> dict[str, Any]:
    handovers = SQLiteStorageAdapter().list_handovers()
    return {"count": len(handovers), "handovers": handovers}


@app.get("/api/sessions/{session_id}/context")
def api_session_context(session_id: str) -> dict[str, Any]:
    storage = SQLiteStorageAdapter()
    return {
        "session_id": session_id,
        "signals": storage.list_signals(session_id),
        "turns": [vars(turn) for turn in storage.list_turns(session_id)],
        "tool_traces": storage.list_tool_traces(session_id),
    }


def _page_heading(title: str, detail: str) -> None:
    with ui.column().classes("page-heading gap-1"):
        ui.label(title).classes("page-title")
        ui.label(detail).classes("page-detail")


@ui.page("/")
def workspace() -> None:
    state = WorkspaceState()
    state.initialize()
    ui.context.client.on_disconnect(state.close)
    assert state.agent is not None and state.session is not None and state.knowledge is not None

    ui.page_title("OrderFlow-Agent")
    ui.colors(primary="#176b52", secondary="#2e5b88", accent="#d16b3f", positive="#176b52", negative="#b63f46")
    ui.add_css(
        """
        :root { --ink:#182126; --muted:#66727a; --line:#d9dee1; --paper:#f7f8f8; --panel:#ffffff; }
        body { background:var(--paper); color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; letter-spacing:0; }
        .q-page { min-height:calc(100vh - 112px)!important; }
        .app-header { background:#fff!important; color:var(--ink)!important; border-bottom:1px solid var(--line); }
        .brand-name { font-size:19px; font-weight:750; line-height:1.1; }
        .brand-line { font-size:12px; color:var(--muted); }
        .top-tabs { background:#fff; border-bottom:1px solid var(--line); color:#48545b; overflow-x:auto; }
        .top-tabs .q-tab { min-height:48px; padding:0 14px; }
        .workspace-panels { background:var(--paper)!important; }
        .workspace-panels > .q-panel > .q-tab-panel { padding:24px; }
        .page-heading { margin-bottom:18px; }
        .page-title { font-size:23px; line-height:1.2; font-weight:750; }
        .page-detail { font-size:14px; color:var(--muted); max-width:850px; }
        .surface { background:var(--panel); border:1px solid var(--line); border-radius:6px; }
        .surface-header { padding:14px 16px; border-bottom:1px solid var(--line); }
        .surface-body { padding:16px; }
        .section-title { font-size:15px; font-weight:700; }
        .section-detail { font-size:12px; color:var(--muted); }
        .live-layout { display:grid; grid-template-columns:minmax(0,1.75fr) minmax(300px,.85fr); gap:16px; width:100%; }
        .chat-scroll { height:510px; background:#fbfcfc; }
        .chat-scroll .q-scrollarea__content { width:100%!important; min-width:0!important; max-width:100%!important; }
        .chat-scroll-content { width:100%; min-width:0; max-width:100%; padding:12px 16px; box-sizing:border-box; overflow-x:hidden; }
        .composer { border-top:1px solid var(--line); padding:12px; }
        .draft-lines { min-height:160px; }
        .order-total { font-size:24px; font-weight:780; color:#1f2d34; }
        .status-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); width:100%; background:#fff; border:1px solid var(--line); border-radius:6px; margin-bottom:16px; }
        .status-item { min-width:0; padding:10px 14px; border-right:1px solid var(--line); }
        .status-item:last-child { border-right:0; }
        .status-value { font-size:13px; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .status-label { color:var(--muted); font-size:10px; text-transform:uppercase; }
        .quick-menu-row { padding:9px 0; border-top:1px solid #e7eaec; }
        .quick-menu-row:first-of-type { border-top:0; }
        .quick-thumb { width:52px; height:52px; border-radius:5px; object-fit:cover; flex:0 0 auto; }
        .runtime-band { padding:13px 15px; background:#edf6f2; border:1px solid #bad5ca; border-radius:6px; }
        .chat-message-body { min-width:0; max-width:100%; }
        .chat-products { display:flex; gap:10px; max-width:620px; padding-top:10px; margin-top:8px; border-top:1px solid #d7dddd; overflow-x:auto; }
        .chat-product { width:154px; min-width:154px; margin:0; padding:0 0 4px; }
        .chat-product-image { width:154px; height:96px; border-radius:5px; object-fit:cover; background:#e8eceb; }
        .chat-product-title { margin-top:7px; font-size:12px; line-height:1.25; font-weight:700; }
        .chat-product-price { font-size:11px; line-height:1.3; color:#35584d; font-weight:700; }
        .chat-product-ingredients { margin-top:3px; font-size:10px; line-height:1.35; color:var(--muted); white-space:normal; }
        .stream-indicator { height:18px; padding-top:2px; }
        .q-message-text--received { background:#eef2f1!important; color:var(--ink)!important; }
        .q-message-text--sent { background:#e6eef5!important; color:var(--ink)!important; }
        .q-message-text--received:before { border-right-color:#eef2f1!important; }
        .q-message-text--sent:before { border-left-color:#e6eef5!important; }
        .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; width:100%; }
        .metric { padding:15px; background:#fff; border:1px solid var(--line); border-radius:6px; }
        .metric-value { font-size:25px; font-weight:780; }
        .metric-label { color:var(--muted); font-size:12px; }
        .two-col { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; width:100%; }
        .three-col { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; width:100%; }
        .trace-drawer { background:#f5f7f7; border-left:1px solid var(--line); }
        .trace-item { border-left:3px solid #a9b2b6; padding:8px 10px; background:#fff; }
        .trace-item.passed { border-left-color:#176b52; }
        .trace-item.blocked { border-left-color:#b63f46; }
        .trace-item.fallback { border-left-color:#c27b2c; }
        .empty-state { color:var(--muted); padding:20px 4px; }
        .compact-table .q-table th { color:#59666d; font-size:11px; text-transform:uppercase; font-weight:700; }
        .compact-table .q-table td { font-size:13px; }
        .q-card { border-radius:6px!important; box-shadow:none!important; border:1px solid var(--line); }
        .q-btn { letter-spacing:0; }
        .q-field__native, .q-field__input { letter-spacing:0; }
        @media (max-width: 1000px) {
          .live-layout,.two-col { grid-template-columns:1fr; }
          .metric-grid { grid-template-columns:repeat(2,minmax(130px,1fr)); }
          .workspace-panels > .q-panel > .q-tab-panel { padding:16px; }
        }
        @media (max-width: 600px) {
          .brand-line { display:none; }
          .provider-chip { display:none!important; }
          .top-tabs .q-tab { padding:0 9px; min-width:72px; }
          .top-tabs .q-tab__label { font-size:11px; }
          .chat-scroll { height:430px; }
          .metric-grid,.three-col { grid-template-columns:1fr; }
          .status-strip { grid-template-columns:repeat(2,minmax(0,1fr)); }
          .status-item:nth-child(2) { border-right:0; }
          .status-item:nth-child(-n+2) { border-bottom:1px solid var(--line); }
          .workspace-panels > .q-panel > .q-tab-panel { padding:12px; }
          .chat-products { max-width:calc(100vw - 98px); }
          .chat-product,.chat-product-image { width:136px; min-width:136px; }
          .chat-product-image { height:86px; }
        }
        """
    )

    @ui.refreshable
    def trace_view() -> None:
        with ui.column().classes("w-full gap-3 p-4"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Tool trace").classes("section-title")
                ui.badge(f"{len(state.trace_history)} turns", color="grey-7")
            ui.label("Latest deterministic path first").classes("section-detail")
            if not state.trace_history:
                ui.label("No tools have run in this session.").classes("empty-state")
                return
            for title, steps in reversed(state.trace_history[-5:]):
                with ui.expansion(title, icon="account_tree", value=title == state.trace_history[-1][0]).classes("w-full"):
                    if not steps:
                        ui.label("No state-changing tool was needed.").classes("section-detail")
                    for index, step in enumerate(steps, start=1):
                        with ui.column().classes(f"trace-item {step.status} w-full gap-1"):
                            with ui.row().classes("w-full items-center justify-between"):
                                ui.label(f"{index}. {step.name}").classes("text-sm font-semibold")
                                ui.badge(step.status, color=_status_color(step.status))
                            if step.detail:
                                ui.label(step.detail).classes("section-detail")

    trace_drawer = (
        ui.right_drawer(value=False)
        .classes("trace-drawer")
        .props("width=360 breakpoint=1180 show-if-above")
    )
    with trace_drawer:
        trace_view()

    with ui.header(elevated=False).classes("app-header h-16 px-4"):
        with ui.row().classes("w-full items-center no-wrap"):
            with ui.column().classes("gap-0 mr-auto"):
                ui.label("OrderFlow-Agent").classes("brand-name")
                ui.label("Conversational Task Agent with Deterministic Tool Execution").classes("brand-line")
            provider_badge = ui.badge(_provider_label(state), color="secondary").props("outline").classes("provider-chip")
            mode_badge = ui.badge("ASSISTED 50", color="primary")
            ui.button(icon="add_comment", on_click=lambda: reset_workspace_session()).props(
                'flat round dense aria-label="New conversation"'
            ).tooltip("New conversation")
            ui.button(icon="account_tree", on_click=trace_drawer.toggle).props(
                'flat round dense aria-label="Toggle tool trace"'
            ).tooltip("Toggle tool trace")

    with ui.tabs().classes("top-tabs w-full").props("align=left mobile-arrows outside-arrows") as tabs:
        live_tab = ui.tab("live", label="Live agent", icon="forum")
        orders_tab = ui.tab("orders", label="Orders", icon="receipt_long")
        behaviour_tab = ui.tab("behaviour", label="Behaviour", icon="monitoring")
        handover_tab = ui.tab("handover", label="Handovers", icon="support_agent")
        menu_ai_tab = ui.tab("menu-ai", label="Menu intelligence", icon="recommend")
        evaluation_tab = ui.tab("evaluation", label="Evaluation", icon="fact_check")
        knowledge_tab = ui.tab("knowledge", label="Knowledge", icon="library_books")
        catalog_tab = ui.tab("catalog", label="Catalog", icon="restaurant_menu")
        settings_tab = ui.tab("settings", label="Settings", icon="settings")

    with ui.dialog() as handover_dialog, ui.card().classes("w-[560px] max-w-[94vw]"):
        ui.label("Prepare staff handover").classes("text-lg font-bold")
        ui.label("Use the customer's own description so the brief does not invent an emotional state.").classes("section-detail")
        handover_state = ui.select(
            ["Frustrated", "Worried", "Disappointed", "Confused", "Other"],
            label="How are you feeling?",
        ).classes("w-full")
        handover_issue = ui.textarea(label="What should the operator resolve?").classes("w-full")

        async def submit_handover() -> None:
            if state.response_in_progress:
                ui.notify("Wait for the current reply to finish.", type="info")
                return
            if not handover_state.value or not str(handover_issue.value or "").strip():
                ui.notify("Enter both the customer state and unresolved issue.", type="warning")
                return
            assert state.agent is not None and state.session is not None
            state.response_in_progress = True
            message_input.disable()
            try:
                _, response = await run.io_bound(
                    partial(
                        state.agent.create_handover,
                        state.session,
                        customer_state=str(handover_state.value),
                        unresolved_issue=str(handover_issue.value),
                    )
                )
                handover_dialog.close()
                await reveal_response(response)
                draft_view.refresh()
                trace_view.refresh()
                handovers_view.refresh()
            finally:
                state.response_in_progress = False
                message_input.enable()
                message_input.run_method("focus")

        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=handover_dialog.close).props("flat")
            ui.button("Create handover", icon="support_agent", on_click=submit_handover)

    chat_scroll: Any = None

    @ui.refreshable
    def chat_view() -> None:
        nonlocal chat_scroll
        chat_scroll = ui.scroll_area().classes("chat-scroll w-full")
        with chat_scroll:
            with ui.column().classes("chat-scroll-content gap-3"):
                for message in state.messages:
                    sent = message["role"] == "customer"
                    speaker = (
                        "You"
                        if sent
                        else "Human employee"
                        if message["role"] == "human"
                        else "OrderFlow"
                    )
                    with ui.chat_message(
                        name=speaker,
                        stamp=message["stamp"],
                        sent=sent,
                    ).classes("w-full"):
                        with ui.column().classes("chat-message-body w-full gap-0"):
                            if sent:
                                ui.label(message["content"]).classes("whitespace-pre-wrap")
                            else:
                                ui.markdown(message["content"]).classes("max-w-full")
                                if message.get("streaming"):
                                    with ui.row().classes("stream-indicator items-center"):
                                        ui.spinner("dots", size="sm", color="primary")
                                elif message.get("attachments"):
                                    with ui.element("div").classes("chat-products"):
                                        for attachment in message["attachments"]:
                                            with ui.element("figure").classes("chat-product"):
                                                image_url = _menu_image_url(str(attachment.get("image", "")))
                                                if image_url:
                                                    ui.image(image_url).classes("chat-product-image")
                                                else:
                                                    with ui.element("div").classes(
                                                        "chat-product-image flex items-center justify-center"
                                                    ):
                                                        ui.icon("restaurant_menu", color="grey-6", size="md")
                                                ui.label(str(attachment["title"])).classes("chat-product-title")
                                                ui.label(
                                                    f"{attachment['currency']} {int(attachment['price']):,}"
                                                ).classes("chat-product-price")
                                                ingredients = attachment.get("ingredients") or []
                                                detail = (
                                                    "Ingredients: " + ", ".join(ingredients)
                                                    if ingredients
                                                    else str(attachment.get("description", ""))
                                                )
                                                if detail:
                                                    ui.label(detail).classes("chat-product-ingredients")

    @ui.refreshable
    def draft_view() -> None:
        assert state.session is not None
        catalog = state.catalog_store.load()
        with ui.column().classes("w-full gap-3"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Current draft").classes("section-title")
                ui.badge(state.session.pending_action.replace("_", " "), color="warning" if state.session.pending_action != "none" else "grey-7")
            with ui.column().classes("draft-lines w-full gap-2"):
                if not state.session.order:
                    ui.label("No items in the draft.").classes("empty-state")
                for name, quantity in state.session.order.items():
                    item = catalog.by_name.get(name)
                    with ui.row().classes("w-full items-start justify-between no-wrap"):
                        with ui.column().classes("gap-0"):
                            ui.label(name).classes("text-sm font-semibold")
                            ui.label(f"Quantity {quantity}").classes("section-detail")
                        ui.label(f"{catalog.currency} {(item.price * quantity if item else 0):,}").classes("text-sm font-semibold")
            ui.separator()
            total = sum(catalog.by_name[name].price * quantity for name, quantity in state.session.order.items() if name in catalog.by_name)
            with ui.row().classes("w-full items-end justify-between"):
                ui.label("Draft total").classes("section-detail")
                ui.label(f"{catalog.currency} {total:,}").classes("order-total")
            with ui.row().classes("w-full"):
                bill_button = ui.button("Bill", icon="receipt", on_click=lambda: send_command("show my bill")).props("outline").classes("flex-1")
                confirm_button = ui.button("Confirm", icon="check_circle", on_click=lambda: send_command("confirm order")).classes("flex-1")
            handover_button = ui.button(
                "Handover pending" if state.session.handover_active else "Human handover",
                icon="support_agent",
                on_click=handover_dialog.open,
            ).props("flat color=secondary").classes("w-full")
            if state.session.handover_active:
                bill_button.disable()
                confirm_button.disable()
                handover_button.disable()

    @ui.refreshable
    def session_status_view() -> None:
        assert state.session is not None
        catalog = state.catalog_store.load()
        cart_items = sum(state.session.order.values())
        values = (
            ("Session", state.session.session_id[:8].upper()),
            ("Cart", f"{cart_items} item{'s' if cart_items != 1 else ''}"),
            ("Gate", state.session.pending_action.replace("_", " ").title()),
            ("Catalog", f"{len(catalog.active_items)} active"),
        )
        with ui.element("div").classes("status-strip"):
            for label, value in values:
                with ui.column().classes("status-item gap-0"):
                    ui.label(label).classes("status-label")
                    ui.label(value).classes("status-value").props(f'title="{value}"')

    @ui.refreshable
    def quick_menu_view() -> None:
        catalog = state.catalog_store.load()
        available_pizzas = [
            item for item in catalog.active_items if item.category.casefold() == "pizza"
        ]
        pizzas = []
        seen_images: set[str] = set()
        for item in available_pizzas:
            image_url = _menu_image_url(item.image)
            if image_url and image_url not in seen_images:
                pizzas.append(item)
                seen_images.add(image_url)
            if len(pizzas) == 3:
                break
        for item in available_pizzas:
            if len(pizzas) == 3:
                break
            if item not in pizzas:
                pizzas.append(item)
        with ui.column().classes("w-full gap-0"):
            with ui.row().classes("w-full items-center justify-between mb-1"):
                ui.label("Quick menu").classes("section-title")
                ui.label(f"{len(catalog.active_items)} available").classes("section-detail")
            for item in pizzas:
                with ui.row().classes("quick-menu-row w-full items-center no-wrap"):
                    image_url = _menu_image_url(item.image)
                    if image_url:
                        ui.image(image_url).classes("quick-thumb")
                    else:
                        with ui.element("div").classes("quick-thumb bg-grey-2 flex items-center justify-center"):
                            ui.icon("local_pizza", color="grey-6")
                    with ui.column().classes("gap-0 flex-1 min-w-0"):
                        ui.label(item.title or item.name).classes("text-sm font-semibold truncate")
                        ui.label(f"{catalog.currency} {item.price:,}").classes("section-detail")
                    ui.button(
                        icon="add_circle",
                        on_click=lambda name=item.name: send_command(f"add one {name}"),
                    ).props(
                        f'flat round dense color=primary aria-label="Add {item.name}" title="Add {item.name}"'
                    )

    message_input: Any = None
    record_button: Any = None
    stop_button: Any = None

    async def speak_response(response: AgentResponse) -> None:
        if not state.tts_enabled or state.provider is None:
            return
        if not getattr(getattr(state.provider, "capabilities", None), "speech", False):
            return
        try:
            audio = await run.io_bound(state.provider.synthesize, response.content[:4000])
            source = "data:audio/mpeg;base64," + base64.b64encode(audio).decode("ascii")
            ui.audio(source).props("autoplay").classes("hidden")
        except Exception as exc:
            ui.notify(f"Speech output failed: {type(exc).__name__}", type="warning")

    async def process_message(content: str, channel: str = "text") -> None:
        text = content.strip()
        if not text:
            return
        if state.response_in_progress:
            ui.notify("Wait for the current reply to finish.", type="info")
            return
        assert state.agent is not None and state.session is not None
        state.response_in_progress = True
        message_input.disable()
        try:
            state.add_user(text, channel)
            chat_view.refresh()
            response = await run.io_bound(state.agent.handle, text, state.session, channel)
            await reveal_response(response)
            draft_view.refresh()
            session_status_view.refresh()
            trace_view.refresh()
            behaviour_view.refresh()
            orders_view.refresh()
            handovers_view.refresh()
            mode_badge.set_text(f"{state.session.agent_mode.value.upper()} {state.session.strictness}")
            await speak_response(response)
        finally:
            state.response_in_progress = False
            message_input.enable()
            message_input.run_method("focus")

    async def reveal_response(response: AgentResponse) -> None:
        message = state.add_response(response, progressive=True)
        chat_view.refresh()
        chat_scroll.scroll_to(percent=1)
        delay_ms = max(0, min(100, int(os.getenv("ORDERFLOW_STREAM_DELAY_MS", "18"))))
        chunks = _stream_chunks(response.content)
        if not chunks:
            message["content"] = response.content
        for chunk in chunks:
            message["content"] += chunk
            chat_view.refresh()
            chat_scroll.scroll_to(percent=1)
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)
        message["streaming"] = False
        chat_view.refresh()
        await asyncio.sleep(0.03)
        chat_scroll.scroll_to(percent=1)

    async def send_message() -> None:
        content = str(message_input.value or "")
        message_input.set_value("")
        await process_message(content)

    def send_command(command: str) -> None:
        message_input.set_value(command)
        ui.timer(0.01, send_message, once=True)

    async def start_recording() -> None:
        if state.provider is None or not getattr(getattr(state.provider, "capabilities", None), "transcription", False):
            ui.notify("Select a provider with transcription support in Settings.", type="warning")
            return
        result = await ui.run_javascript(
            """
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
              return {ok:false, error:'Microphone capture is not available in this browser.'};
            }
            try {
              const stream = await navigator.mediaDevices.getUserMedia({audio:true});
              const recorder = new MediaRecorder(stream);
              window.orderflowVoice = {recorder, stream, chunks:[]};
              recorder.ondataavailable = event => { if (event.data.size) window.orderflowVoice.chunks.push(event.data); };
              recorder.start();
              return {ok:true, mime:recorder.mimeType};
            } catch (error) {
              return {ok:false, error:String(error)};
            }
            """,
            timeout=15,
        )
        if not result.get("ok"):
            ui.notify(result.get("error", "Microphone could not start."), type="negative")
            return
        record_button.visible = False
        stop_button.visible = True
        ui.notify("Recording started", type="positive")

    async def stop_recording() -> None:
        result = await ui.run_javascript(
            """
            return await new Promise(resolve => {
              const voice = window.orderflowVoice;
              if (!voice || voice.recorder.state === 'inactive') {
                resolve({ok:false, error:'No recording is active.'});
                return;
              }
              voice.recorder.onstop = () => {
                const blob = new Blob(voice.chunks, {type:voice.recorder.mimeType || 'audio/webm'});
                const reader = new FileReader();
                reader.onloadend = () => {
                  voice.stream.getTracks().forEach(track => track.stop());
                  resolve({ok:true, data:reader.result, mime:blob.type});
                  delete window.orderflowVoice;
                };
                reader.readAsDataURL(blob);
              };
              voice.recorder.stop();
            });
            """,
            timeout=30,
        )
        record_button.visible = True
        stop_button.visible = False
        if not result.get("ok"):
            ui.notify(result.get("error", "Recording could not be read."), type="negative")
            return
        try:
            encoded = str(result["data"]).split(",", 1)[1]
            audio = base64.b64decode(encoded)
            assert state.provider is not None
            transcript = await run.io_bound(state.provider.transcribe, audio, "voice-turn.webm")
        except Exception as exc:
            ui.notify(f"Transcription failed: {type(exc).__name__}", type="negative")
            return
        await process_message(transcript, "voice")

    def switch_policy(event: events.ValueChangeEventArguments) -> None:
        strictness = int(event.value)
        state.new_session(strictness)
        mode_badge.set_text(f"{state.session.agent_mode.value.upper()} {strictness}")
        chat_view.refresh()
        draft_view.refresh()
        session_status_view.refresh()
        trace_view.refresh()
        ui.notify(f"Started a new {state.session.agent_mode.value.upper()} session.", type="positive")

    def reset_workspace_session() -> None:
        assert state.session is not None
        state.new_session(state.session.strictness)
        chat_view.refresh()
        draft_view.refresh()
        session_status_view.refresh()
        trace_view.refresh()
        ui.notify("New ordering conversation started.", type="positive")

    @ui.refreshable
    def orders_view() -> None:
        orders = state.storage.list_orders()
        with ui.column().classes("w-full gap-4"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(f"{len(orders)} persisted order{'s' if len(orders) != 1 else ''}").classes("section-title")
                with ui.row().classes("gap-2"):
                    ui.button(
                        icon="download",
                        on_click=lambda: ui.download.content(state.storage.export_orders_csv(), "orderflow-orders.csv", "text/csv"),
                    ).props('outline aria-label="Export orders as CSV" title="Export CSV"')
                    ui.button(
                        icon="data_object",
                        on_click=lambda: ui.download.content(state.storage.export_orders_json(), "orderflow-orders.json", "application/json"),
                    ).props('outline aria-label="Export orders as JSON" title="Export JSON"')
            if not orders:
                ui.label("Confirmed orders will appear here after the explicit confirmation gate succeeds.").classes("empty-state")
                return
            rows = [
                {
                    "id": order["id"][:8].upper(),
                    "created": order["created_at"][:19].replace("T", " "),
                    "items": sum(int(line["quantity"]) for line in order["lines"]),
                    "total": f"{order['currency']} {order['total']:,}",
                    "status": order["status"],
                }
                for order in orders
            ]
            ui.table(
                rows=rows,
                columns=[
                    {"name": "id", "label": "Reference", "field": "id", "align": "left"},
                    {"name": "created", "label": "Confirmed", "field": "created", "sortable": True},
                    {"name": "items", "label": "Items", "field": "items", "sortable": True},
                    {"name": "total", "label": "Total", "field": "total"},
                    {"name": "status", "label": "Status", "field": "status"},
                ],
                row_key="id",
                pagination=10,
            ).classes("compact-table w-full")

    @ui.refreshable
    def behaviour_view() -> None:
        rows = state.storage.list_sessions()
        summary = condition_summary(rows)
        totals = {
            "sessions": len(rows),
            "orders": sum(int(row["confirmed_orders"]) for row in rows),
            "repairs": sum(int(row["repair_requests"]) for row in rows),
            "failures": sum(int(row["compliance_failures"]) for row in rows),
        }
        with ui.column().classes("w-full gap-4"):
            with ui.element("div").classes("metric-grid"):
                for label, value in (
                    ("Sessions", totals["sessions"]),
                    ("Verified task successes", totals["orders"]),
                    ("Repair requests", totals["repairs"]),
                    ("Compliance failures", totals["failures"]),
                ):
                    with ui.element("section").classes("metric"):
                        ui.label(str(value)).classes("metric-value")
                        ui.label(label).classes("metric-label")
            if summary:
                formatted = [
                    {
                        "mode": row["agent_mode"].upper(),
                        "sessions": row["sessions"],
                        "success": f"{row['verified_task_success_rate'] * 100:.0f}%",
                        "repairs": row["repair_requests"],
                        "repair_rate": "n/a" if row["repair_success_rate"] is None else f"{row['repair_success_rate'] * 100:.0f}%",
                        "failures": row["compliance_failures"],
                    }
                    for row in summary
                ]
                ui.table(rows=formatted, pagination=0).classes("compact-table w-full")
            else:
                ui.label("Behavioural measures appear after live sessions run.").classes("empty-state")

    @ui.refreshable
    def handovers_view() -> None:
        handovers = state.storage.list_handovers()
        with ui.column().classes("w-full gap-3"):
            if not handovers:
                ui.label("No handovers are waiting for an operator.").classes("empty-state")
                return
            for row in handovers:
                payload = row["handover"]
                with ui.element("section").classes("surface w-full"):
                    with ui.row().classes("surface-header w-full items-center justify-between"):
                        with ui.column().classes("gap-0"):
                            ui.label(payload["decision"]["trigger"].replace("_", " ").title()).classes("section-title")
                            ui.label(payload["issue"]).classes("section-detail")
                        ui.badge(row["status"], color="positive" if row["status"] == "completed" else "warning")
                    with ui.column().classes("surface-body w-full gap-2"):
                        ui.label(
                            f"Conversation signal: {payload['signal']['label']} "
                            f"({payload['signal']['confidence']:.2f})"
                        ).classes("text-sm font-semibold")
                        ui.label(payload["signal"]["evidence"]).classes("section-detail")
                        ui.code(payload["summary"]).classes("w-full")
                        if row["status"] == "completed":
                            context_note = (
                                f"Cart context carried forward {row['context_carryover_score'] * 100:.0f}%"
                                if payload["cart"]
                                else "No active cart context was present at handover."
                            )
                            ui.label(context_note).classes("text-sm font-semibold")
                        else:
                            ui.button(
                                "Open operator handover",
                                icon="support_agent",
                                on_click=lambda payload=payload: open_operator_handover(payload),
                            ).props("outline color=secondary")

    with ui.dialog() as operator_dialog, ui.card().classes("w-[680px] max-w-[95vw]"):
        operator_title = ui.label("Complete handover").classes("text-lg font-bold")
        operator_context = ui.label().classes("section-detail")
        operator_response = ui.textarea(label="Human response").classes("w-full")
        operator_facts = ui.select([], label="Cart facts carried forward", multiple=True).classes("w-full")
        active_handover: dict[str, Any] = {}

        def complete_handover() -> None:
            if not str(operator_response.value or "").strip():
                ui.notify("Enter the human response before completing the handover.", type="warning")
                return
            payload = active_handover
            record = HandoverCase(
                id=payload["id"],
                session_id=payload["session_id"],
                decision=HandoverDecision(**payload["decision"]),
                signal=ConversationSignal(**payload["signal"]),
                customer_request=payload["customer_request"],
                issue=payload["issue"],
                cart=dict(payload["cart"]),
                conversation_history=tuple(payload["conversation_history"]),
                tool_history=tuple(payload["tool_history"]),
                actions_attempted=tuple(payload["actions_attempted"]),
                outstanding_problem=payload["outstanding_problem"],
                relevant_customer_context=payload["relevant_customer_context"],
                suggested_next_action=payload["suggested_next_action"],
                summary=payload["summary"],
                human_response=str(operator_response.value),
                facts_carried_forward=tuple(operator_facts.value or []),
                status="completed",
                created_at=payload["created_at"],
            )
            state.storage.save_handover(record)
            if (
                state.session is not None
                and state.session.handover_active
                and state.session.handover_case_id == record.id
            ):
                state.session.handover_active = False
                state.session.handover_case_id = ""
                state.session.pending_action = "none"
                state.storage.ensure_session(state.session)
                human_turn = TranscriptTurn(
                    state.session.session_id,
                    "human",
                    record.human_response,
                )
                state.storage.append_turn(human_turn)
                state.messages.append(
                    {
                        "role": "human",
                        "content": record.human_response,
                        "channel": "text",
                        "stamp": datetime.now().strftime("%H:%M"),
                    }
                )
            operator_dialog.close()
            handovers_view.refresh()
            chat_view.refresh()
            draft_view.refresh()
            session_status_view.refresh()
            ui.notify("Operator handover saved.", type="positive")

        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=operator_dialog.close).props("flat")
            ui.button("Complete handover", icon="done_all", on_click=complete_handover)

    def open_operator_handover(payload: dict[str, Any]) -> None:
        active_handover.clear()
        active_handover.update(payload)
        operator_title.set_text(f"Handover {payload['id'][:8].upper()}")
        operator_context.set_text(payload["relevant_customer_context"] + " " + payload["outstanding_problem"])
        operator_response.set_value("")
        cart_facts = [f"{quantity} x {name}" for name, quantity in payload["cart"].items()]
        operator_facts.set_options(cart_facts or ["No items in cart"])
        operator_facts.set_value([])
        operator_dialog.open()

    menu_query: Any = None

    async def capture_menu_query_image(event: events.UploadEventArguments) -> None:
        state.menu_query_image = await event.file.read()
        state.menu_query_image_name = event.file.name
        ui.notify(f"Reference image loaded: {event.file.name}", type="positive")

    async def recommend_menu_items() -> None:
        query = str(menu_query.value or "").strip()
        if not query:
            ui.notify("Describe the pizza or item you want.", type="warning")
            return
        try:
            state.menu_recommendations = await run.io_bound(
                partial(
                    state.menu_intelligence().recommend,
                    query,
                    query_image=state.menu_query_image,
                    limit=6,
                )
            )
        except Exception as exc:
            ui.notify(f"Menu ranking failed: {type(exc).__name__}", type="negative")
            return
        menu_results_view.refresh()

    def fit_synthetic_interest_model() -> None:
        payload = json.loads((ROOT / "data" / "synthetic_menu_interactions.json").read_text(encoding="utf-8"))
        model = MenuInterestModel(state.menu_intelligence())
        try:
            state.interest_report = model.fit(payload["labels"], dataset_label=payload["dataset_label"])
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
            return
        state.interest_model = model
        menu_results_view.refresh()
        ui.notify("Synthetic demo interest model fitted.", type="positive")

    @ui.refreshable
    def menu_results_view() -> None:
        with ui.column().classes("w-full gap-3"):
            if state.interest_report is not None:
                report = state.interest_report
                ui.label(
                    f"Demo interest model: {report.sample_count} synthetic labels, "
                    f"{report.folds}-fold MAE {report.mae:.4f}"
                ).classes("section-detail")
            if not state.menu_recommendations:
                ui.label("No menu ranking has been requested.").classes("empty-state")
                return
            catalog = state.catalog_store.load()
            for recommendation in state.menu_recommendations:
                item = recommendation.item
                image_url = _menu_image_url(item.image)
                with ui.element("section").classes("surface w-full"):
                    with ui.row().classes("surface-body w-full items-start no-wrap"):
                        if image_url:
                            ui.image(image_url).classes("w-28 h-24 object-cover rounded")
                        else:
                            with ui.element("div").classes("w-28 h-24 bg-grey-2 flex items-center justify-center rounded"):
                                ui.icon("local_pizza", size="36px", color="grey-6")
                        with ui.column().classes("gap-1 flex-1 min-w-0"):
                            with ui.row().classes("w-full items-start justify-between no-wrap"):
                                ui.label(item.title or item.name).classes("section-title")
                                ui.label(f"{catalog.currency} {item.price:,}").classes("font-bold")
                            ui.label(item.description).classes("section-detail")
                            ui.label(recommendation.reason).classes("section-detail")
                            with ui.row().classes("gap-2"):
                                ui.badge(f"similarity {recommendation.score:.3f}", color="secondary").props("outline")
                                if state.interest_model is not None:
                                    score = state.interest_model.predict(item)
                                    ui.badge(f"synthetic interest {score:.3f}", color="warning").props("outline")

    async def run_scenario_evaluation() -> None:
        try:
            state.evaluation_result = await run.io_bound(
                partial(
                    run_evaluation,
                    ROOT / "evaluation" / "configs" / "default.json",
                    output_root=ROOT / "evaluation" / "results",
                )
            )
        except Exception as exc:
            ui.notify(f"Evaluation failed: {type(exc).__name__}", type="negative")
            return
        evaluation_view.refresh()
        ui.notify("Scenario evaluation completed.", type="positive")

    @ui.refreshable
    def evaluation_view() -> None:
        with ui.column().classes("w-full gap-3"):
            if state.evaluation_result is None:
                ui.label("No scenario run has been started from this workspace.").classes("empty-state")
                return
            results = state.evaluation_result["results"]
            passed = sum(result.passed for result in results)
            with ui.element("div").classes("metric-grid"):
                for label, value in (
                    ("Mode/scenario runs", len(results)),
                    ("Expectation checks passed", passed),
                    ("Handovers", sum(result.metrics.handovers for result in results)),
                    ("Validation failures", sum(result.metrics.validation_failures for result in results)),
                ):
                    with ui.element("section").classes("metric"):
                        ui.label(str(value)).classes("metric-value")
                        ui.label(label).classes("metric-label")
            ui.label(state.evaluation_result["run_directory"]).classes("section-detail")

    async def ingest_document(event: events.UploadEventArguments) -> None:
        data = await event.file.read()
        try:
            _, created, chunks = await run.io_bound(
                partial(
                    state.knowledge.ingest,
                    event.file.name,
                    event.file.content_type,
                    data,
                    title=Path(event.file.name).stem,
                )
            )
        except RAGError as exc:
            ui.notify(str(exc), type="negative")
            return
        ui.notify(f"{'Indexed' if created else 'Already indexed'}: {chunks} chunk(s).", type="positive")
        knowledge_documents_view.refresh()

    knowledge_query: Any = None
    knowledge_answer: Any = None

    async def run_knowledge_query() -> None:
        query = str(knowledge_query.value or "").strip()
        if not query:
            return
        try:
            result = await run.io_bound(state.knowledge.answer, query)
        except Exception as exc:
            ui.notify(f"Retrieval failed: {type(exc).__name__}", type="negative")
            return
        knowledge_answer.clear()
        with knowledge_answer:
            if result is None:
                ui.label("No stored source supported an answer.").classes("empty-state")
            else:
                answer, steps = result
                ui.markdown(answer)
                for step in steps:
                    ui.badge(step.detail, color="secondary").props("outline")

    @ui.refreshable
    def knowledge_documents_view() -> None:
        documents = state.storage.list_knowledge_documents()
        if not documents:
            ui.label("No documents indexed.").classes("empty-state")
            return
        rows = [
            {"source": row["source_name"], "title": row["title"], "chunks": row["chunk_count"], "added": row["created_at"][:10]}
            for row in documents
        ]
        ui.table(rows=rows, pagination=8).classes("compact-table w-full")

    @ui.refreshable
    def catalog_view() -> None:
        catalog = state.catalog_store.load()
        rows = [
            {
                "sku": item.sku,
                "name": item.name,
                "category": item.category,
                "price": f"{catalog.currency} {item.price:,}",
                "active": "Yes" if item.active else "No",
            }
            for item in catalog.items
        ]
        ui.table(rows=rows, row_key="sku", pagination=12).classes("compact-table w-full")

    editor_item: Any = None
    editor_sku: Any = None
    editor_name: Any = None
    editor_title: Any = None
    editor_category: Any = None
    editor_price: Any = None
    editor_aliases: Any = None
    editor_ingredients: Any = None
    editor_image: Any = None
    editor_stats: Any = None
    editor_description: Any = None
    editor_active: Any = None

    def load_catalog_item(event: events.ValueChangeEventArguments) -> None:
        sku = str(event.value or "")
        item = next((row for row in state.catalog_store.load().items if row.sku == sku), None)
        if item is None:
            editor_sku.set_value("")
            editor_name.set_value("")
            editor_title.set_value("")
            editor_category.set_value("")
            editor_price.set_value(None)
            editor_aliases.set_value("")
            editor_ingredients.set_value("")
            editor_image.set_value("")
            editor_stats.set_value("{}")
            editor_description.set_value("")
            editor_active.set_value(True)
            return
        editor_sku.set_value(item.sku)
        editor_name.set_value(item.name)
        editor_title.set_value(item.title or item.name)
        editor_category.set_value(item.category)
        editor_price.set_value(item.price)
        editor_aliases.set_value(", ".join(item.aliases))
        editor_ingredients.set_value(", ".join(item.ingredients))
        editor_image.set_value(item.image)
        editor_stats.set_value(json.dumps(item.interaction_stats, ensure_ascii=True))
        editor_description.set_value(item.description)
        editor_active.set_value(item.active)

    def save_catalog_item() -> None:
        try:
            item = CatalogItem(
                sku=str(editor_sku.value or "").strip(),
                name=str(editor_name.value or "").strip(),
                category=str(editor_category.value or "").strip(),
                price=int(editor_price.value),
                aliases=tuple(value.strip() for value in str(editor_aliases.value or "").split(",") if value.strip()),
                description=str(editor_description.value or "").strip(),
                title=str(editor_title.value or editor_name.value or "").strip(),
                ingredients=tuple(
                    value.strip() for value in str(editor_ingredients.value or "").split(",") if value.strip()
                ),
                image=str(editor_image.value or "").strip(),
                interaction_stats={
                    str(key): float(value)
                    for key, value in json.loads(str(editor_stats.value or "{}")).items()
                },
                active=bool(editor_active.value),
            )
            state.catalog_store.upsert(item)
        except (TypeError, ValueError, OSError) as exc:
            ui.notify(str(exc), type="negative")
            return
        editor_item.set_options({row.sku: row.name for row in state.catalog_store.load().items})
        editor_item.set_value(item.sku)
        catalog_view.refresh()
        draft_view.refresh()
        quick_menu_view.refresh()
        session_status_view.refresh()
        ui.notify("Catalog item saved.", type="positive")

    async def upload_catalog_image(event: events.UploadEventArguments) -> None:
        data = await event.file.read()
        suffix = Path(event.file.name).suffix.casefold()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            ui.notify("Use a PNG, JPEG, or WebP image.", type="negative")
            return
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            ui.notify("The uploaded file is not a readable image.", type="negative")
            return
        stem = re.sub(r"[^a-z0-9]+", "-", Path(event.file.name).stem.casefold()).strip("-")
        filename = f"{stem or 'menu-item'}{suffix}"
        MENU_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        (MENU_IMAGE_DIR / filename).write_bytes(data)
        editor_image.set_value(f"data/menu_images/{filename}")
        ui.notify(f"Image ready: {filename}", type="positive")

    with ui.tab_panels(tabs, value=live_tab).classes("workspace-panels w-full"):
        with ui.tab_panel(live_tab):
            with ui.row().classes("w-full items-end justify-between gap-4 mb-4"):
                _page_heading("Live ordering agent", "The conversation is flexible. Catalog state, prices, confirmation, and persistence remain tool-owned.")
                ui.toggle(
                    {"80": "Controlled", "50": "Assisted", "20": "Flexible"},
                    value="50",
                    on_change=switch_policy,
                ).props("no-caps unelevated toggle-color=primary")
            session_status_view()
            with ui.element("div").classes("live-layout"):
                with ui.element("section").classes("surface overflow-hidden"):
                    chat_view()
                    with ui.column().classes("composer w-full gap-2"):
                        with ui.row().classes("w-full items-center no-wrap"):
                            message_input = ui.input(placeholder="Message OrderFlow...").props("borderless dense").classes("flex-1")
                            message_input.on("keydown.enter", send_message)
                            record_button = ui.button(icon="mic", on_click=start_recording).props(
                                'flat round aria-label="Start voice turn"'
                            ).tooltip("Start voice turn")
                            stop_button = ui.button(icon="stop_circle", on_click=stop_recording, color="negative").props(
                                'flat round aria-label="Stop and transcribe"'
                            ).tooltip("Stop and transcribe")
                            stop_button.visible = False
                            ui.button(icon="send", on_click=send_message).props(
                                'round unelevated aria-label="Send message"'
                            ).tooltip("Send")
                        with ui.row().classes("w-full gap-2"):
                            ui.button("Menu", icon="restaurant_menu", on_click=lambda: send_command("show the menu")).props("flat dense")
                            ui.button("Add example", icon="add", on_click=lambda: send_command("add one medium tandoori chicken pizza and one ranch dip")).props("flat dense")
                            ui.switch("Voice reply", value=False, on_change=lambda e: setattr(state, "tts_enabled", bool(e.value))).props("dense color=secondary")
                with ui.element("section").classes("surface"):
                    with ui.column().classes("surface-body w-full gap-4"):
                        draft_view()
                        ui.separator()
                        quick_menu_view()

        with ui.tab_panel(orders_tab):
            _page_heading("Order history", "Only orders that crossed the explicit confirmation gate are included.")
            orders_view()

        with ui.tab_panel(behaviour_tab):
            _page_heading("Prompt policy and behaviour", "Compare verified task completion, conversational repair, and outward compliance failures across operating modes.")
            behaviour_view()
            with ui.expansion("Current compiled policy", icon="policy").classes("surface w-full mt-4"):
                ui.code(PromptPolicyCompiler().compile(state.session.strictness).instructions).classes("w-full")

        with ui.tab_panel(handover_tab):
            _page_heading("Human handovers", "Operators receive the customer-stated context, current task facts, and unresolved issue in one brief.")
            handovers_view()

        with ui.tab_panel(menu_ai_tab):
            _page_heading(
                "Multimodal menu intelligence",
                "Rank related pizzas from descriptions and an optional reference image. Missing images use the text-only fallback.",
            )
            with ui.element("div").classes("two-col"):
                with ui.element("section").classes("surface"):
                    with ui.column().classes("surface-body w-full gap-3"):
                        ui.label("Find a related menu item").classes("section-title")
                        menu_query = ui.textarea(
                            label="Request",
                            value="I want something similar to a spicy chicken pizza but vegetarian",
                        ).classes("w-full")
                        ui.upload(
                            on_upload=capture_menu_query_image,
                            auto_upload=True,
                            max_file_size=8_000_000,
                        ).props("accept=image/* flat bordered").classes("w-full")
                        with ui.row().classes("w-full justify-end"):
                            ui.button("Rank menu", icon="recommend", on_click=recommend_menu_items)
                with ui.element("section").classes("surface"):
                    with ui.column().classes("surface-body w-full gap-3"):
                        ui.label("Synthetic interest demonstration").classes("section-title")
                        ui.label(
                            "The optional score uses only the labelled synthetic dataset in "
                            "data/synthetic_menu_interactions.json. It is not customer behaviour."
                        ).classes("section-detail")
                        ui.button(
                            "Fit synthetic demo model",
                            icon="model_training",
                            on_click=fit_synthetic_interest_model,
                        ).props("outline color=secondary")
            with ui.element("section").classes("surface mt-4"):
                with ui.column().classes("surface-body w-full"):
                    menu_results_view()

        with ui.tab_panel(evaluation_tab):
            _page_heading(
                "Agent behaviour evaluation",
                "Replay the same pizza-order scenarios across CONTROLLED, ASSISTED, and FLEXIBLE modes.",
            )
            with ui.row().classes("w-full items-center justify-between mb-4"):
                ui.label(
                    "Outputs include configuration, scenario set, seed, software metadata, JSON, CSV, and a readable summary."
                ).classes("section-detail")
                ui.button("Run full scenario matrix", icon="play_arrow", on_click=run_scenario_evaluation)
            evaluation_view()

        with ui.tab_panel(knowledge_tab):
            _page_heading("Operational knowledge", "Dual retrieval fuses lexical and vector rankings while retaining source, rank, and timing evidence.")
            with ui.element("div").classes("two-col"):
                with ui.element("section").classes("surface"):
                    with ui.column().classes("surface-body w-full gap-3"):
                        ui.label("Ask stored sources").classes("section-title")
                        knowledge_query = ui.input(label="Question").classes("w-full")
                        ui.button("Retrieve", icon="search", on_click=run_knowledge_query).classes("self-end")
                        knowledge_answer = ui.column().classes("w-full gap-2")
                with ui.element("section").classes("surface"):
                    with ui.column().classes("surface-body w-full gap-3"):
                        ui.label("Add source").classes("section-title")
                        ui.upload(on_upload=ingest_document, auto_upload=True, max_file_size=12_000_000).props("accept=.txt,.md,.csv,.json,.pdf flat bordered").classes("w-full")
                        knowledge_documents_view()

        with ui.tab_panel(catalog_tab):
            _page_heading("Catalog administration", "Operators can update the JSON catalog without changing Python code. Validation runs before every save.")
            with ui.element("div").classes("two-col"):
                with ui.element("section").classes("surface overflow-hidden"):
                    catalog_view()
                with ui.element("section").classes("surface"):
                    with ui.column().classes("surface-body w-full gap-3"):
                        ui.label("Item editor").classes("section-title")
                        editor_item = ui.select(
                            {item.sku: item.name for item in state.catalog_store.load().items},
                            label="Existing item",
                            clearable=True,
                            on_change=load_catalog_item,
                        ).classes("w-full")
                        editor_sku = ui.input(label="SKU").classes("w-full")
                        editor_name = ui.input(label="Name").classes("w-full")
                        editor_title = ui.input(label="Display title").classes("w-full")
                        editor_category = ui.input(label="Category").classes("w-full")
                        editor_price = ui.number(label="Price", min=0, precision=0).classes("w-full")
                        editor_aliases = ui.input(label="Aliases, comma separated").classes("w-full")
                        editor_ingredients = ui.input(label="Ingredients, comma separated").classes("w-full")
                        editor_image = ui.input(label="Optional image path").classes("w-full")
                        ui.upload(
                            on_upload=upload_catalog_image,
                            auto_upload=True,
                            max_file_size=8_000_000,
                        ).props("accept=image/png,image/jpeg,image/webp flat bordered").classes("w-full")
                        editor_stats = ui.input(label="Optional interaction statistics JSON", value="{}").classes("w-full")
                        editor_description = ui.textarea(label="Description").classes("w-full")
                        editor_active = ui.switch("Active", value=True)
                        ui.button("Save item", icon="save", on_click=save_catalog_item).classes("self-end")

        with ui.tab_panel(settings_tab):
            _page_heading("Runtime settings", "Credentials stay in this browser workspace process and are not written to SQLite, exports, or traces.")
            package_version = kernelloom_package_version()
            with ui.row().classes("runtime-band w-full items-center no-wrap mb-4"):
                ui.icon("verified_user", color="primary", size="sm")
                with ui.column().classes("gap-0 min-w-0"):
                    ui.label("Deterministic transaction boundary active").classes("text-sm font-semibold")
                    ui.label(
                        "The selected model can phrase approved replies. Catalog validation, prices, confirmation, and persistence remain server-owned."
                    ).classes("section-detail")
            with ui.element("div").classes("two-col"):
                with ui.element("section").classes("surface"):
                    with ui.column().classes("surface-body w-full gap-3"):
                        ui.label("Model connection").classes("section-title")
                        provider_select = ui.select(
                            ProviderRegistry.labels,
                            value=state.settings.provider_id,
                            label="Model provider",
                        ).classes("w-full")
                        provider_fields = ui.column().classes("w-full gap-3")
                        provider_fields.bind_visibility_from(
                            provider_select,
                            "value",
                            backward=lambda value: value != "disabled",
                        )
                        with provider_fields:
                            api_key = ui.input(
                                "API key or local token",
                                password=True,
                                password_toggle_button=True,
                                value=state.settings.api_key,
                            ).classes("w-full")
                            base_url = ui.input("Base URL", value=state.settings.base_url).classes("w-full")
                            response_model = ui.input(
                                "Chat or response model ID",
                                value=state.settings.response_model,
                            ).classes("w-full")
                            embedding_model = ui.input(
                                "Embedding model ID",
                                value=state.settings.embedding_model,
                            ).classes("w-full")

                        openai_fields = ui.column().classes("w-full gap-3")
                        openai_fields.bind_visibility_from(
                            provider_select,
                            "value",
                            backward=lambda value: value == "openai",
                        )
                        with openai_fields:
                            ui.label("OpenAI voice models").classes("text-sm font-semibold")
                            transcription_model = ui.input(
                                "Transcription model",
                                value=state.settings.transcription_model,
                            ).classes("w-full")
                            speech_model = ui.input(
                                "Speech model",
                                value=state.settings.speech_model,
                            ).classes("w-full")
                            voice = ui.input("Voice", value=state.settings.voice).classes("w-full")

                        kernelloom_fields = ui.column().classes("w-full gap-3")
                        kernelloom_fields.bind_visibility_from(
                            provider_select,
                            "value",
                            backward=lambda value: value == "kernelloom",
                        )
                        with kernelloom_fields:
                            ui.separator()
                            with ui.row().classes("w-full items-center justify-between"):
                                ui.label("KernelLoom execution").classes("text-sm font-semibold")
                                ui.badge(
                                    f"PyPI {package_version}" if package_version else "package missing",
                                    color="positive" if package_version else "negative",
                                ).props("outline")
                            kernelloom_transport = ui.toggle(
                                {"http": "Local server", "python": "Python package"},
                                value=state.settings.kernelloom_transport,
                            ).props("no-caps unelevated toggle-color=secondary").classes("w-full")
                            python_fields = ui.column().classes("w-full gap-3")
                            python_fields.bind_visibility_from(
                                kernelloom_transport,
                                "value",
                                backward=lambda value: value == "python",
                            )
                            with python_fields:
                                kernelloom_chat_path = ui.input(
                                    "Local chat model path",
                                    value=state.settings.kernelloom_chat_model_path,
                                ).classes("w-full")
                                kernelloom_embedding_path = ui.input(
                                    "Optional local embedding model path",
                                    value=state.settings.kernelloom_embedding_model_path,
                                ).classes("w-full")
                                with ui.row().classes("w-full gap-3"):
                                    kernelloom_backend = ui.select(
                                        ["auto", "llama-cpp", "openvino"],
                                        value=state.settings.kernelloom_backend,
                                        label="Backend",
                                    ).classes("flex-1")
                                    kernelloom_device = ui.select(
                                        ["CPU", "GPU", "NPU", "AUTO"],
                                        value=state.settings.kernelloom_device,
                                        label="Device",
                                    ).classes("flex-1")
                                with ui.row().classes("w-full gap-3"):
                                    kernelloom_cpu_profile = ui.select(
                                        ["latency", "throughput", "efficient", "auto"],
                                        value=state.settings.kernelloom_cpu_profile,
                                        label="CPU profile",
                                    ).classes("flex-1")
                                    kernelloom_reserve_cores = ui.number(
                                        "Reserved cores",
                                        value=state.settings.kernelloom_reserve_cores,
                                        min=0,
                                        precision=0,
                                    ).classes("flex-1")

                        openagent_fields = ui.column().classes("w-full gap-3")
                        openagent_fields.bind_visibility_from(
                            provider_select,
                            "value",
                            backward=lambda value: value == "openagent",
                        )
                        with openagent_fields:
                            allow_external = ui.switch(
                                "Allow OpenAgent external providers",
                                value=state.settings.allow_external,
                            )
                        settings_status = ui.label(
                            state.provider_error or "No runtime changes are pending."
                        ).classes("section-detail")

                        def save_settings() -> None:
                            settings = RuntimeSettings(
                                provider_id=str(provider_select.value),
                                api_key=str(api_key.value or "").strip(),
                                base_url=str(base_url.value or "").strip(),
                                response_model=str(response_model.value or "").strip(),
                                embedding_model=str(embedding_model.value or "").strip(),
                                transcription_model=str(transcription_model.value or "").strip(),
                                speech_model=str(speech_model.value or "").strip(),
                                voice=str(voice.value or "").strip(),
                                openagent_provider=state.settings.openagent_provider,
                                openagent_project="orderflow-agent",
                                allow_external=bool(allow_external.value),
                                kernelloom_transport=str(kernelloom_transport.value or "http"),
                                kernelloom_chat_model_path=str(kernelloom_chat_path.value or "").strip(),
                                kernelloom_embedding_model_path=str(
                                    kernelloom_embedding_path.value or ""
                                ).strip(),
                                kernelloom_backend=str(kernelloom_backend.value or "auto"),
                                kernelloom_device=str(kernelloom_device.value or "CPU"),
                                kernelloom_cpu_profile=str(
                                    kernelloom_cpu_profile.value or "latency"
                                ),
                                kernelloom_reserve_cores=max(
                                    0, int(kernelloom_reserve_cores.value or 0)
                                ),
                            )
                            try:
                                state.reconfigure(settings)
                            except ProviderUnavailable as exc:
                                settings_status.set_text(str(exc))
                                ui.notify(str(exc), type="negative")
                                return
                            provider_badge.set_text(_provider_label(state))
                            refresh_runtime_summary()
                            settings_status.set_text("Runtime settings applied to this workspace.")
                            ui.notify("Runtime updated.", type="positive")

                        async def test_provider() -> None:
                            if state.provider is None:
                                settings_status.set_text("No model provider is active. Deterministic ordering remains available.")
                                return
                            status = await run.io_bound(state.provider.check)
                            settings_status.set_text(status.detail)
                            ui.notify(status.label, type="positive" if status.ok else "negative")

                        with ui.row().classes("w-full justify-end"):
                            ui.button("Apply", icon="save", on_click=save_settings)
                            ui.button("Test", icon="network_check", on_click=test_provider).props("outline")
                with ui.element("section").classes("surface"):
                    with ui.column().classes("surface-body w-full gap-3"):
                        ui.label("Active runtime").classes("section-title")
                        runtime_name = ui.label().classes("text-lg font-bold")
                        runtime_detail = ui.label().classes("section-detail")
                        runtime_capabilities = ui.row().classes("w-full gap-2")

                        def refresh_runtime_summary() -> None:
                            runtime_name.set_text(_provider_label(state))
                            if state.provider is None:
                                runtime_detail.set_text(
                                    "Ordering remains available with deterministic interpretation and tools."
                                )
                                capabilities = ("catalog", "confirmation", "persistence")
                            else:
                                provider_capabilities = getattr(state.provider, "capabilities", None)
                                enabled = [
                                    label
                                    for attribute, label in (
                                        ("text", "text"),
                                        ("embeddings", "embeddings"),
                                        ("transcription", "transcription"),
                                        ("speech", "speech"),
                                    )
                                    if getattr(provider_capabilities, attribute, False)
                                ]
                                runtime_detail.set_text(
                                    "Model access is scoped to response wording and optional retrieval features."
                                )
                                capabilities = tuple(enabled)
                            runtime_capabilities.clear()
                            with runtime_capabilities:
                                for capability in capabilities:
                                    ui.badge(capability, color="secondary").props("outline")

                        refresh_runtime_summary()
                        ui.separator()
                        ui.label("Available adapters").classes("section-title")
                        for name, detail in (
                            ("OpenAI API", "Generation, embeddings, transcription, and speech."),
                            (
                                "KernelLoom",
                                "Local generation and embeddings through the PyPI Python API or loopback server.",
                            ),
                            ("OpenAgent", "Local routed chat and its own LatticeRAG service."),
                            ("Hugging Face", "Hosted OpenAI-compatible chat endpoint."),
                            ("Disabled", "Full deterministic ordering with no model calls."),
                        ):
                            with ui.row().classes("w-full items-start no-wrap"):
                                ui.icon("check_circle", color="primary").classes("mt-1")
                                with ui.column().classes("gap-0"):
                                    ui.label(name).classes("text-sm font-semibold")
                                    ui.label(detail).classes("section-detail")


def run_app() -> None:
    ui.run(
        host=os.getenv("ORDERFLOW_HOST", "127.0.0.1"),
        port=int(os.getenv("ORDERFLOW_PORT", "8080")),
        title="OrderFlow-Agent",
        reload=os.getenv("ORDERFLOW_RELOAD", "0") == "1",
        show=False,
    )
