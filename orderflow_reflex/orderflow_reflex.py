"""Reflex customer ordering surface and staff operations workspace."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, Iterator

import reflex as rx

from orderflow_agent.agent import ConversationalTaskAgent
from orderflow_agent.catalog import CatalogItem, JsonCatalogStore
from orderflow_agent.evaluation.runner import DEFAULT_CONFIG, run_evaluation
from orderflow_agent.knowledge import KnowledgeService
from orderflow_agent.models import AgentResponse, utc_now
from orderflow_agent.multimodal import MenuIntelligence
from orderflow_agent.runtime.customer_service import (
    check_runtime,
    configure_runtime,
    current_settings,
    streaming_provider,
)
from orderflow_agent.runtime.streaming import GroundedStreamingResponder, StreamingReplyError
from orderflow_agent.storage import SQLiteStorageAdapter
from orderflow_agent.tools import generate_bill


ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_UPLOAD_ID = "orderflow-knowledge-upload"
STORAGE = SQLiteStorageAdapter()
CATALOG_STORE = JsonCatalogStore()
KnowledgeService(STORAGE).seed(ROOT / "data" / "knowledge_base.md")


def _asset_path(path: str) -> str:
    return "/menu/" + Path(path).name if path else ""


def _product_dict(item: Any) -> dict[str, Any]:
    return {
        "sku": item.sku,
        "title": item.title,
        "description": item.description,
        "ingredients": ", ".join(item.ingredients) or "Ingredients not listed",
        "image": _asset_path(item.image),
        "price": f"{item.currency} {item.price:,}",
    }


def _client_is_connected(client_token: str) -> bool:
    reflex_app = globals().get("app")
    namespace = getattr(reflex_app, "event_namespace", None)
    manager = getattr(namespace, "_token_manager", None)
    if manager is None:
        return True
    return bool(client_token and client_token in manager.token_to_socket)


class CustomerState(rx.State):
    """Per-browser ordering state; model and storage adapters stay server-side."""

    messages: list[dict[str, str]] = []
    products: list[dict[str, Any]] = []
    cart_lines: list[dict[str, str]] = []
    cart_total: str = "PKR 0"
    fulfilment: str = "Not selected"
    stage: str = "Choose a pizza"
    prompt: str = ""
    busy: bool = False
    error: str = ""
    handover_pending: bool = False
    handover_ticket_id: str = ""
    handover_ticket: str = ""
    handover_status: str = ""
    handover_queue_position: int = 0
    handover_seen_messages: list[str] = []
    confirmed_reference: str = ""
    confirmed_total: str = ""
    confirmed_fulfilment: str = ""
    confirmed_delivery_address: str = ""
    _agent: Any = None
    _session: Any = None

    def _ensure_session(self) -> None:
        if self._agent is None:
            self._agent = ConversationalTaskAgent(catalog_store=CATALOG_STORE, storage=STORAGE)
        if self._session is None:
            self._session = self._agent.open_session(mode="assisted")

    def _sync_cart(self) -> None:
        self._ensure_session()
        bill = generate_bill(self._session.order, self._agent.catalog)
        self.cart_lines = [
            {
                "name": line.item,
                "quantity": str(line.quantity),
                "total": f"{bill.currency} {line.total:,}",
            }
            for line in bill.lines
        ]
        self.cart_total = f"{bill.currency} {bill.grand_total:,}"
        if self._session.fulfilment == "delivery":
            self.fulfilment = "Delivery" + (
                f" to {self._session.delivery_address}" if self._session.delivery_address else ""
            )
        elif self._session.fulfilment == "pickup":
            self.fulfilment = "Pickup"
        else:
            self.fulfilment = "Not selected"
        if self.confirmed_reference:
            self.stage = "Order confirmed"
        elif not self._session.order:
            self.stage = "Choose a pizza"
        elif self._session.fulfilment == "undecided":
            self.stage = "Choose delivery or pickup"
        elif self._session.fulfilment == "delivery" and not self._session.delivery_address:
            self.stage = "Add delivery address"
        elif self._session.pending_action == "confirm_order":
            self.stage = "Confirm order"
        else:
            self.stage = "Review order"
        self.handover_pending = bool(self._session.handover_active)

    def _sync_products(self, response: AgentResponse | None = None) -> None:
        if response and response.menu_attachments:
            self.products = [_product_dict(item) for item in response.menu_attachments]
            return
        self._ensure_session()
        pizzas = [item for item in self._agent.catalog.active_items if item.category.casefold() == "pizza"]
        self.products = [
            {
                "sku": item.sku,
                "title": item.name,
                "description": item.description,
                "ingredients": ", ".join(item.ingredients) or "Ingredients not listed",
                "image": _asset_path(item.image),
                "price": f"{self._agent.catalog.currency} {item.price:,}",
            }
            for item in pizzas
        ]

    def _sync_handover(self) -> None:
        case_id = self.handover_ticket_id or (
            self._session.handover_case_id if self._session is not None else ""
        )
        if not case_id:
            return
        row = STORAGE.get_handover(case_id)
        if row is None:
            return
        payload = row["handover"]
        self.handover_ticket_id = case_id
        self.handover_ticket = case_id[:8].upper()
        self.handover_status = str(row["status"])
        self.handover_queue_position = STORAGE.handover_queue_position(case_id) or 0
        self.handover_pending = row["status"] == "pending"
        seen = set(self.handover_seen_messages)
        updated_messages = list(self.messages)
        for message in payload.get("live_messages", []):
            message_id = str(message.get("id", ""))
            if not message_id or message_id in seen:
                continue
            seen.add(message_id)
            if message.get("role") == "staff":
                updated_messages.append({"role": "staff", "content": str(message.get("content", ""))})
        self.messages = updated_messages
        self.handover_seen_messages = list(seen)
        if self.handover_pending:
            self.stage = "Waiting for restaurant staff"
            return
        if self._session is not None and self._session.handover_active:
            self._session.handover_active = False
            self._session.handover_case_id = ""
            self._session.pending_action = "none"
            STORAGE.ensure_session(self._session)
        self.stage = "Support ticket resolved"

    @rx.event
    def initialize(self) -> None:
        self._ensure_session()
        self._sync_cart()
        self._sync_products()

    @rx.event
    def new_order(self) -> None:
        self._ensure_session()
        self._session = self._agent.open_session(mode="assisted")
        self.messages = []
        self.prompt = ""
        self.error = ""
        self.confirmed_reference = ""
        self.confirmed_total = ""
        self.confirmed_fulfilment = ""
        self.confirmed_delivery_address = ""
        self.handover_ticket_id = ""
        self.handover_ticket = ""
        self.handover_status = ""
        self.handover_queue_position = 0
        self.handover_seen_messages = []
        self._sync_cart()
        self._sync_products()

    def _stream_turn(self, text: str) -> Iterator[None]:
        self._ensure_session()
        clean = text.strip()
        if not clean or self.busy:
            return
        prior = tuple((row["role"], row["content"]) for row in self.messages[-8:])
        self.messages = [*self.messages, {"role": "user", "content": clean}]
        self.prompt = ""
        self.error = ""
        if self.handover_pending and self.handover_ticket_id:
            try:
                message = STORAGE.append_handover_message(
                    self.handover_ticket_id,
                    role="customer",
                    content=clean,
                )
                self.handover_seen_messages = [*self.handover_seen_messages, message["id"]]
                self._sync_handover()
            except ValueError as exc:
                self.error = str(exc)
            yield
            return
        self.busy = True
        yield
        operational = self._agent.handle(clean, self._session)
        if operational.confirmed_order_id:
            stored = next(
                (row for row in STORAGE.list_orders() if row["id"] == operational.confirmed_order_id),
                None,
            )
            if stored:
                self.confirmed_reference = stored["id"][:8].upper()
                self.confirmed_total = f"{stored['currency']} {stored['total']:,}"
                self.confirmed_fulfilment = str(stored.get("fulfilment", "pickup")).title()
                self.confirmed_delivery_address = str(stored.get("delivery_address", ""))
        self._sync_products(operational)
        self._sync_cart()
        if operational.handover_case_id:
            self.handover_ticket_id = operational.handover_case_id
        self._sync_handover()
        self.messages = [*self.messages, {"role": "assistant", "content": ""}]
        yield
        generated = ""
        started = time.perf_counter()
        try:
            responder = GroundedStreamingResponder(streaming_provider(), self._agent.catalog)
            for fragment in responder.stream(
                strictness=self._session.strictness,
                user_message=clean,
                operational_response=operational,
                visible_history=prior,
            ):
                generated += fragment
                updated = [*self.messages]
                updated[-1] = {"role": "assistant", "content": generated}
                self.messages = updated
                yield
        except Exception as exc:
            if operational.handover_requested:
                generated = operational.content.strip()
                updated = [*self.messages]
                updated[-1] = {"role": "assistant", "content": generated}
                self.messages = updated
                STORAGE.replace_latest_ai_response(self._session.session_id, generated)
            elif not generated.strip():
                self.messages = self.messages[:-1]
                STORAGE.discard_latest_ai_response(self._session.session_id)
            else:
                STORAGE.replace_latest_ai_response(self._session.session_id, generated.strip())
            detail = str(exc) if isinstance(exc, StreamingReplyError) else "The configured model could not answer."
            STORAGE.append_latest_tool_step(
                self._session.session_id,
                {"name": "model_stream", "status": "blocked", "detail": detail, "created_at": utc_now()},
            )
            self.error = (
                ""
                if operational.handover_requested
                else "The ordering assistant is temporarily unavailable. Please try again or ask a staff member."
            )
            self.busy = False
            yield
            return
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        STORAGE.replace_latest_ai_response(self._session.session_id, generated.strip())
        STORAGE.append_latest_tool_step(
            self._session.session_id,
            {
                "name": "model_stream",
                "status": "passed",
                "detail": f"Customer reply streamed in {elapsed_ms} ms.",
                "created_at": utc_now(),
            },
        )
        self.busy = False
        self._sync_cart()
        self._sync_handover()
        yield

    @rx.event
    def send(self, form_data: dict[str, Any]) -> Iterator[None]:
        yield from self._stream_turn(str(form_data.get("message", self.prompt)))

    @rx.event
    def quick_message(self, message: str) -> Iterator[None]:
        yield from self._stream_turn(message)

    @rx.event
    def add_product(self, sku: str) -> Iterator[None]:
        self._ensure_session()
        item = next((row for row in self._agent.catalog.active_items if row.sku == sku), None)
        if item is None:
            self.error = "That menu item is no longer available."
            yield
            return
        yield from self._stream_turn(f"Add one {item.name}")

    @rx.event(background=True)
    async def monitor_handover(self) -> None:
        async with self:
            client_token = self.router.session.client_token
        while True:
            await asyncio.sleep(1.5)
            if not _client_is_connected(client_token):
                return
            async with self:
                self._sync_handover()


class StaffState(rx.State):
    """Operational views kept away from the customer ordering page."""

    section: str = "overview"
    section_title: str = "Overview"
    order_count: int = 0
    handover_count: int = 0
    session_count: int = 0
    orders: list[dict[str, str]] = []
    handovers: list[dict[str, str]] = []
    traces: list[dict[str, str]] = []
    catalog_rows: list[dict[str, str]] = []
    handover_options: list[str] = []
    catalog_options: list[str] = []
    knowledge_documents: list[dict[str, str]] = []
    evaluation_rows: list[dict[str, str]] = []
    evaluation_summary: str = ""
    evaluation_path: str = ""
    busy: bool = False
    notice: str = ""
    selected_handover: str = ""
    ticket_reply: str = ""
    ticket_number: str = ""
    ticket_status: str = ""
    ticket_queue: str = ""
    ticket_trigger: str = ""
    ticket_request: str = ""
    ticket_issue: str = ""
    ticket_context: str = ""
    ticket_cart: str = ""
    ticket_fulfilment: str = ""
    ticket_created: str = ""
    ticket_messages: list[dict[str, str]] = []
    menu_query: str = "something spicy and vegetarian"
    recommendations: list[dict[str, str]] = []
    knowledge_query: str = ""
    knowledge_answer: str = ""
    provider_id: str = "kernelloom"
    api_key: str = ""
    response_model: str = ""
    base_url: str = ""
    local_model_path: str = ""
    device: str = "CPU"
    runtime_result: str = ""
    catalog_sku: str = ""
    catalog_name: str = ""
    catalog_category: str = "Pizza"
    catalog_price: str = "0"
    catalog_description: str = ""
    catalog_aliases: str = ""
    catalog_ingredients: str = ""
    catalog_tags: str = ""
    catalog_image: str = ""
    catalog_active: bool = True

    def _refresh_data(self) -> None:
        raw_orders = STORAGE.list_orders()
        raw_handovers = STORAGE.list_handovers()
        raw_traces = STORAGE.list_tool_traces()
        self.order_count = len(raw_orders)
        self.handover_count = sum(row["status"] == "pending" for row in raw_handovers)
        self.session_count = len(STORAGE.list_sessions())
        self.orders = [
            {
                "id": row["id"][:8].upper(),
                "status": row["status"],
                "fulfilment": row.get("fulfilment", "pickup").title(),
                "total": f"{row['currency']} {row['total']:,}",
                "items": ", ".join(f"{line['quantity']} x {line['item']}" for line in row["lines"]),
                "created": row["created_at"][:19].replace("T", " "),
            }
            for row in raw_orders
        ]
        pending = sorted(
            (row for row in raw_handovers if row["status"] == "pending"),
            key=lambda row: (row["created_at"], row["id"]),
        )
        queue = {row["id"]: index for index, row in enumerate(pending, start=1)}
        completed = [row for row in raw_handovers if row["status"] != "pending"]
        ordered_handovers = [*pending, *completed]
        self.handovers = [
            {
                "id": row["id"],
                "short_id": row["id"][:8].upper(),
                "status": row["status"],
                "queue": str(queue.get(row["id"], "")),
                "trigger": row["handover"].get("decision", {}).get("trigger", "unknown").replace("_", " ").title(),
                "request": row["handover"].get("customer_request", ""),
                "summary": row["handover"].get("summary", ""),
                "created": row["created_at"][:19].replace("T", " "),
            }
            for row in ordered_handovers
        ]
        self.handover_options = [row["id"] for row in raw_handovers if row["status"] == "pending"]
        self.traces = [
            {
                "session": row["session_id"][:8].upper(),
                "steps": " | ".join(
                    f"{step.get('name', 'tool')}: {step.get('status', 'info')}" for step in row["steps"]
                ),
                "created": row["created_at"][:19].replace("T", " "),
            }
            for row in raw_traces[:80]
        ]
        catalog = CATALOG_STORE.load()
        self.catalog_rows = [
            {
                "sku": item.sku,
                "name": item.name,
                "category": item.category,
                "price": f"{catalog.currency} {item.price:,}",
                "active": "Yes" if item.active else "No",
            }
            for item in catalog.items
        ]
        self.catalog_options = [item.sku for item in catalog.items]
        self.knowledge_documents = [
            {
                "title": row["title"],
                "source": row["source_name"],
                "chunks": str(row["chunk_count"]),
                "created": row["created_at"][:19].replace("T", " "),
            }
            for row in STORAGE.list_knowledge_documents()
        ]
        if self.selected_handover and not STORAGE.get_handover(self.selected_handover):
            self.selected_handover = ""
        if not self.selected_handover and ordered_handovers:
            self.selected_handover = ordered_handovers[0]["id"]
        self._sync_selected_ticket()

    def _sync_selected_ticket(self) -> None:
        if not self.selected_handover:
            self.ticket_number = ""
            self.ticket_messages = []
            return
        row = STORAGE.get_handover(self.selected_handover)
        if row is None:
            return
        payload = row["handover"]
        self.ticket_number = row["id"][:8].upper()
        self.ticket_status = str(row["status"])
        position = STORAGE.handover_queue_position(row["id"])
        self.ticket_queue = f"Queue {position}" if position else "Resolved"
        self.ticket_trigger = str(payload.get("decision", {}).get("trigger", "unknown")).replace("_", " ").title()
        self.ticket_request = str(payload.get("customer_request", ""))
        self.ticket_issue = str(payload.get("outstanding_problem", payload.get("issue", "")))
        self.ticket_context = str(payload.get("relevant_customer_context", ""))
        cart = payload.get("cart", {})
        self.ticket_cart = ", ".join(f"{quantity} x {name}" for name, quantity in cart.items()) or "No items"
        fulfilment = str(payload.get("fulfilment", "undecided")).title()
        address = str(payload.get("delivery_address", ""))
        self.ticket_fulfilment = fulfilment + (f" · {address}" if address else "")
        self.ticket_created = str(row["created_at"][:19]).replace("T", " ")
        history = [
            {
                "role": {"customer": "customer", "ai": "agent", "human": "staff"}.get(
                    str(message.get("role", "")), "agent"
                ),
                "content": str(message.get("content", "")),
                "time": str(message.get("created_at", ""))[11:16],
            }
            for message in payload.get("conversation_history", [])
            if str(message.get("content", "")).strip()
        ]
        live = [
            {
                "role": str(message.get("role", "agent")),
                "content": str(message.get("content", "")),
                "time": str(message.get("created_at", ""))[11:16],
            }
            for message in payload.get("live_messages", [])
        ]
        self.ticket_messages = [*history, *live]

    @rx.event
    def initialize(self) -> None:
        settings = current_settings()
        self.provider_id = settings.provider_id
        self.response_model = settings.response_model
        self.base_url = settings.base_url
        self.local_model_path = settings.kernelloom_chat_model_path
        self.device = settings.kernelloom_device
        self._refresh_data()

    @rx.event
    def navigate(self, section: str) -> None:
        self.section = section
        self.section_title = dict((key, label) for key, _, label in NAV_ITEMS).get(section, "Overview")
        self.notice = ""
        self._refresh_data()

    @rx.event
    def refresh(self) -> None:
        self._refresh_data()
        self.notice = "Workspace refreshed."

    @rx.event
    def download_orders_csv(self):
        return rx.download(data=STORAGE.export_orders_csv(), filename="orderflow-orders.csv")

    @rx.event
    def download_orders_json(self):
        return rx.download(data=STORAGE.export_orders_json(), filename="orderflow-orders.json")

    @rx.event
    def select_handover(self, case_id: str) -> None:
        self.selected_handover = case_id
        self.ticket_reply = ""
        self._sync_selected_ticket()

    @rx.event
    def send_ticket_reply(self, form_data: dict[str, Any]) -> None:
        if not self.selected_handover:
            self.notice = "Select a ticket first."
            return
        reply = str(form_data.get("reply", self.ticket_reply)).strip()
        if not reply:
            self.notice = "Write a reply before sending."
            return
        try:
            STORAGE.append_handover_message(self.selected_handover, role="staff", content=reply)
        except ValueError as exc:
            self.notice = str(exc)
            return
        self.ticket_reply = ""
        self.notice = "Reply sent."
        self._refresh_data()

    @rx.event
    def resolve_handover(self) -> None:
        if not self.selected_handover:
            self.notice = "Select a ticket first."
            return
        row = STORAGE.get_handover(self.selected_handover)
        if row is None:
            self.notice = "Ticket was not found."
            return
        staff_messages = [
            message
            for message in row["handover"].get("live_messages", [])
            if message.get("role") == "staff" and str(message.get("content", "")).strip()
        ]
        if not staff_messages:
            self.notice = "Send a customer reply before resolving the ticket."
            return
        cart = row["handover"].get("cart", {})
        facts = tuple(f"{quantity} x {name}" for name, quantity in cart.items())
        STORAGE.complete_handover(
            self.selected_handover,
            human_response=str(staff_messages[-1]["content"]),
            facts_carried_forward=facts,
        )
        self.notice = f"Ticket {self.ticket_number} resolved."
        self._refresh_data()

    @rx.event(background=True)
    async def monitor_workspace(self) -> None:
        async with self:
            client_token = self.router.session.client_token
        while True:
            await asyncio.sleep(1.5)
            if not _client_is_connected(client_token):
                return
            async with self:
                self._refresh_data()

    @rx.event
    def rank_menu(self) -> None:
        rows = MenuIntelligence(CATALOG_STORE.load(), asset_root=ROOT).recommend(self.menu_query, limit=6)
        self.recommendations = [
            {
                "name": row.item.name,
                "price": f"{CATALOG_STORE.load().currency} {row.item.price:,}",
                "score": f"{row.score:.3f}",
                "reason": row.reason,
            }
            for row in rows
        ]

    @rx.event
    def query_knowledge(self) -> None:
        result = KnowledgeService(STORAGE).answer(self.knowledge_query.strip())
        self.knowledge_answer = result[0] if result else "No stored evidence matched this query."

    @rx.event
    async def upload_knowledge(self, files: list[rx.UploadFile]) -> None:
        if not files:
            self.notice = "Choose a document first."
            return
        upload = files[0]
        data = await upload.read()
        filename = str(getattr(upload, "filename", "") or getattr(upload, "name", "document.txt"))
        mime_type = str(getattr(upload, "content_type", "") or "application/octet-stream")
        _, created, chunk_count = KnowledgeService(STORAGE).ingest(filename, mime_type, data)
        self.notice = (
            f"Added {filename} with {chunk_count} chunks."
            if created
            else f"{filename} already exists in the knowledge store."
        )
        self._refresh_data()

    @rx.event
    def run_scenarios(self) -> Iterator[None]:
        self.busy = True
        self.notice = ""
        yield
        result = run_evaluation(DEFAULT_CONFIG, output_root=ROOT / "evaluation" / "results")
        self.evaluation_path = str(result["run_directory"])
        rows = result["results"]
        self.evaluation_rows = [
            {
                "mode": row.agent_mode,
                "scenario": row.scenario_id,
                "passed": "Pass" if row.passed else "Fail",
                "turns": str(row.metrics.turns),
                "tools": str(row.metrics.tool_calls),
                "handovers": str(row.metrics.handovers),
            }
            for row in rows
        ]
        passed = sum(row.passed for row in rows)
        self.evaluation_summary = f"{passed} of {len(rows)} scenario and mode checks passed."
        self.busy = False
        yield

    @rx.event
    def apply_runtime(self) -> None:
        try:
            configure_runtime(
                provider_id=self.provider_id,
                api_key=self.api_key,
                response_model=self.response_model,
                local_model_path=self.local_model_path,
                device=self.device,
                base_url=self.base_url,
            )
            self.api_key = ""
            self.runtime_result = "Runtime settings applied for this server process."
        except Exception as exc:
            self.runtime_result = str(exc)

    @rx.event
    def test_runtime(self) -> Iterator[None]:
        self.busy = True
        self.runtime_result = "Checking model runtime..."
        yield
        available, detail = check_runtime()
        self.runtime_result = ("Available: " if available else "Unavailable: ") + detail
        self.busy = False
        yield

    @rx.event
    def load_catalog_item(self, sku: str) -> None:
        item = next((row for row in CATALOG_STORE.load().items if row.sku == sku), None)
        if item is None:
            return
        self.catalog_sku = item.sku
        self.catalog_name = item.name
        self.catalog_category = item.category
        self.catalog_price = str(item.price)
        self.catalog_description = item.description
        self.catalog_aliases = ", ".join(item.aliases)
        self.catalog_ingredients = ", ".join(item.ingredients)
        self.catalog_tags = ", ".join(item.tags)
        self.catalog_image = item.image
        self.catalog_active = item.active

    @rx.event
    def save_catalog_item(self) -> None:
        try:
            current = next((row for row in CATALOG_STORE.load().items if row.sku == self.catalog_sku), None)
            item = CatalogItem(
                sku=self.catalog_sku.strip(),
                name=self.catalog_name.strip(),
                title=self.catalog_name.strip(),
                category=self.catalog_category.strip(),
                price=int(self.catalog_price),
                aliases=tuple(value.strip() for value in self.catalog_aliases.split(",") if value.strip()),
                description=self.catalog_description.strip(),
                tags=tuple(value.strip() for value in self.catalog_tags.split(",") if value.strip()),
                ingredients=tuple(value.strip() for value in self.catalog_ingredients.split(",") if value.strip()),
                image=self.catalog_image.strip(),
                interaction_stats=current.interaction_stats if current else {},
                active=self.catalog_active,
            )
            CATALOG_STORE.upsert(item)
            self.notice = f"Saved {item.name}."
            self._refresh_data()
        except Exception as exc:
            self.notice = str(exc)


def brand_mark() -> rx.Component:
    return rx.hstack(
        rx.box(rx.icon("pizza", size=22), class_name="brand-icon"),
        rx.vstack(
            rx.heading("OrderFlow", size="5", class_name="brand-title"),
            rx.text("Pizza ordering", class_name="brand-subtitle"),
            spacing="0",
            align="start",
        ),
        spacing="3",
        align="center",
    )


def customer_header() -> rx.Component:
    return rx.hstack(
        brand_mark(),
        rx.spacer(),
        rx.hstack(
            rx.box(class_name="status-dot"),
            rx.text("Taking orders"),
            class_name="header-service-status",
            align="center",
            spacing="2",
        ),
        rx.tooltip(
            rx.icon_button(
                rx.icon("rotate-ccw", size=18),
                on_click=CustomerState.new_order,
                variant="soft",
                class_name="header-action",
                aria_label="Start a new order",
            ),
            content="Start a new order",
        ),
        width="100%",
        class_name="customer-header",
        align="center",
    )


def chat_message(message: dict[str, str]) -> rx.Component:
    is_user = message["role"] == "user"
    is_staff = message["role"] == "staff"
    return rx.hstack(
        rx.box(
            rx.icon("sparkles", size=16),
            class_name="assistant-avatar",
            display=rx.cond(is_user | is_staff, "none", "grid"),
        ),
        rx.box(
            rx.icon("headset", size=16),
            class_name="staff-avatar",
            display=rx.cond(is_staff, "grid", "none"),
        ),
        rx.box(
            rx.text(message["content"]),
            class_name="message message-user",
            display=rx.cond(is_user, "block", "none"),
        ),
        rx.box(
            rx.text(message["content"], white_space="pre-wrap"),
            class_name="message message-assistant",
            display=rx.cond(is_user | is_staff, "none", "block"),
        ),
        rx.box(
            rx.text(message["content"], white_space="pre-wrap"),
            class_name="message message-staff",
            display=rx.cond(is_staff, "block", "none"),
        ),
        width="100%",
        align="start",
        justify=rx.cond(is_user, "end", "start"),
    )


def quick_actions() -> rx.Component:
    return rx.hstack(
        rx.button(
            rx.icon("menu", size=15),
            "Menu",
            on_click=lambda: CustomerState.quick_message("Show me the pizza menu"),
            variant="soft",
            size="2",
        ),
        rx.button(
            rx.icon("receipt-text", size=15),
            "Bill",
            on_click=lambda: CustomerState.quick_message("Show my current bill"),
            variant="soft",
            size="2",
        ),
        rx.button(
            rx.icon("flame", size=15),
            "Something spicy",
            on_click=lambda: CustomerState.quick_message("Recommend a spicy pizza from the menu"),
            variant="soft",
            size="2",
        ),
        wrap="wrap",
        spacing="2",
        class_name="quick-actions",
    )


def customer_ticket() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.box(rx.icon("headset", size=17), class_name="ticket-icon"),
            rx.vstack(
                rx.text("Support ticket", class_name="ticket-kicker"),
                rx.text("#", CustomerState.handover_ticket, class_name="ticket-reference"),
                spacing="0",
                align="start",
            ),
            rx.spacer(),
            rx.vstack(
                rx.badge(
                    rx.cond(CustomerState.handover_status == "pending", "Waiting", "Resolved"),
                    color_scheme=rx.cond(CustomerState.handover_status == "pending", "orange", "green"),
                    variant="soft",
                ),
                rx.text(
                    "Queue ",
                    CustomerState.handover_queue_position,
                    class_name="queue-position",
                    display=rx.cond(CustomerState.handover_status == "pending", "block", "none"),
                ),
                spacing="1",
                align="end",
            ),
            width="100%",
            align="center",
        ),
        class_name="customer-ticket",
        display=rx.cond(CustomerState.handover_ticket != "", "block", "none"),
    )


def typing_indicator() -> rx.Component:
    return rx.hstack(
        rx.box(rx.icon("sparkles", size=16), class_name="assistant-avatar"),
        rx.hstack(rx.box(), rx.box(), rx.box(), class_name="typing-dots", spacing="1"),
        width="100%",
        align="center",
        display=rx.cond(CustomerState.busy, "flex", "none"),
    )


def chat_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.text("Live ordering agent", class_name="eyebrow"),
                    rx.heading(CustomerState.stage, size="6"),
                    spacing="1",
                    align="start",
                ),
                rx.spacer(),
                rx.hstack(
                    rx.box(class_name=rx.cond(CustomerState.busy, "status-dot status-dot-busy", "status-dot")),
                    rx.text(rx.cond(CustomerState.busy, "Writing", "Online")),
                    class_name="chat-service-status",
                    align="center",
                    spacing="2",
                ),
                width="100%",
                align="center",
                class_name="chat-panel-header",
            ),
            customer_ticket(),
            rx.box(
                rx.vstack(
                    rx.box(
                        rx.image(src="/menu/cheese.png", alt="Cheese pizza", class_name="empty-chat-image"),
                        rx.box(rx.icon("message-circle", size=18), class_name="empty-chat-icon"),
                        class_name="empty-chat-visual",
                    ),
                    rx.heading("What can I make for you?", size="5"),
                    rx.text("Ask for the menu or tell me the pizza, size, and quantity you want."),
                    align="center",
                    class_name="empty-chat",
                    display=rx.cond(CustomerState.messages.length() == 0, "flex", "none"),
                ),
                rx.vstack(
                    rx.foreach(CustomerState.messages, chat_message),
                    typing_indicator(),
                    width="100%",
                    spacing="4",
                    display=rx.cond(CustomerState.messages.length() == 0, "none", "flex"),
                ),
                class_name="chat-scroll",
                width="100%",
            ),
            rx.callout(
                CustomerState.error,
                icon="triangle-alert",
                color_scheme="red",
                size="1",
                display=rx.cond(CustomerState.error != "", "flex", "none"),
            ),
            quick_actions(),
            rx.form(
                rx.hstack(
                    rx.input(
                        name="message",
                        max_length=1200,
                        value=CustomerState.prompt,
                        on_change=CustomerState.set_prompt,
                        placeholder=rx.cond(
                            CustomerState.handover_pending,
                            "Message restaurant staff...",
                            "Type your pizza order...",
                        ),
                        disabled=CustomerState.busy,
                        class_name="chat-input",
                    ),
                    rx.tooltip(
                        rx.icon_button(
                            rx.icon("send", size=18),
                            type="submit",
                            disabled=CustomerState.busy,
                            class_name="send-button",
                            aria_label="Send message",
                        ),
                        content="Send message",
                    ),
                    width="100%",
                ),
                on_submit=CustomerState.send,
                reset_on_submit=True,
                width="100%",
                class_name="chat-composer",
            ),
            width="100%",
            height="100%",
            spacing="4",
            align="stretch",
        ),
        class_name="chat-panel",
    )


def product_card(product: dict[str, Any]) -> rx.Component:
    return rx.box(
        rx.box(
            rx.image(src=product["image"], alt=product["title"], class_name="product-image"),
            class_name="product-media",
        ),
        rx.vstack(
            rx.hstack(
                rx.text(product["title"], class_name="product-title"),
                rx.spacer(),
                rx.text(product["price"], class_name="product-price"),
                rx.tooltip(
                    rx.icon_button(
                        rx.icon("shopping-cart", size=15),
                        on_click=lambda: CustomerState.add_product(product["sku"]),
                        disabled=CustomerState.busy | CustomerState.handover_pending,
                        variant="soft",
                        size="1",
                        class_name="product-add",
                        aria_label="Add to order",
                    ),
                    content="Add to order",
                ),
                width="100%",
                align="start",
            ),
            rx.text(product["description"], class_name="product-description"),
            rx.text("Ingredients: ", product["ingredients"], class_name="ingredients"),
            spacing="1",
            align="start",
            width="100%",
        ),
        class_name="product-card",
    )


def cart_line(line: dict[str, str]) -> rx.Component:
    return rx.hstack(
        rx.box(line["quantity"], class_name="quantity"),
        rx.text(line["name"], class_name="cart-name"),
        rx.spacer(),
        rx.text(line["total"], class_name="cart-price"),
        width="100%",
        align="center",
    )


def order_sidebar() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.hstack(
                rx.icon("shopping-cart", size=18),
                rx.heading("Your order", size="4"),
                rx.spacer(),
                rx.badge(CustomerState.cart_lines.length(), variant="soft", class_name="cart-count"),
                spacing="2",
                width="100%",
                align="center",
                class_name="cart-heading",
            ),
            rx.box(
                rx.hstack(rx.icon("circle-check", size=17), rx.text("Order confirmed", class_name="receipt-title"), spacing="2"),
                rx.hstack(rx.text("Reference", class_name="muted"), rx.spacer(), rx.text(CustomerState.confirmed_reference, class_name="receipt-value"), width="100%"),
                rx.hstack(rx.text("Total", class_name="muted"), rx.spacer(), rx.text(CustomerState.confirmed_total, class_name="receipt-value"), width="100%"),
                rx.hstack(rx.text("Fulfilment", class_name="muted"), rx.spacer(), rx.text(CustomerState.confirmed_fulfilment, class_name="receipt-value"), width="100%"),
                rx.hstack(
                    rx.text("Address", class_name="muted"),
                    rx.spacer(),
                    rx.text(CustomerState.confirmed_delivery_address, class_name="receipt-value receipt-address"),
                    width="100%",
                    display=rx.cond(CustomerState.confirmed_delivery_address != "", "flex", "none"),
                ),
                class_name="confirmation-receipt",
                display=rx.cond(CustomerState.confirmed_reference != "", "block", "none"),
            ),
            rx.text(
                "Your cart is empty.",
                class_name="muted",
                display=rx.cond(CustomerState.cart_lines.length() == 0, "block", "none"),
            ),
            rx.vstack(
                rx.foreach(CustomerState.cart_lines, cart_line),
                width="100%",
                spacing="3",
                display=rx.cond(CustomerState.cart_lines.length() == 0, "none", "flex"),
            ),
            rx.separator(),
            rx.hstack(rx.text("Total"), rx.spacer(), rx.text(CustomerState.cart_total, class_name="cart-total"), width="100%"),
            rx.hstack(rx.icon("map-pin", size=15), rx.text(CustomerState.fulfilment, class_name="muted"), spacing="2"),
            class_name="cart-box",
        ),
        rx.hstack(
            rx.heading("Pizza menu", size="4"),
            rx.spacer(),
            rx.text(str(CATALOG_STORE.load().currency), class_name="currency-label"),
            width="100%",
            align="center",
        ),
        rx.box(
            rx.vstack(rx.foreach(CustomerState.products, product_card), width="100%", spacing="3"),
            class_name="menu-scroll",
            width="100%",
        ),
        class_name="order-sidebar",
        width="100%",
        align="stretch",
    )


def customer_page() -> rx.Component:
    return rx.box(
        customer_header(),
        rx.grid(
            chat_panel(),
            order_sidebar(),
            columns=rx.breakpoints(initial="1", lg="minmax(0, 1.45fr) minmax(330px, .72fr)"),
            gap="20px",
            class_name="customer-grid",
        ),
        class_name="customer-shell",
    )


NAV_ITEMS = (
    ("overview", "layout-dashboard", "Overview"),
    ("orders", "receipt-text", "Orders"),
    ("handovers", "users", "Handovers"),
    ("catalog", "notebook-tabs", "Catalogue"),
    ("intelligence", "scan-search", "Menu intelligence"),
    ("evaluation", "flask-conical", "Evaluation"),
    ("knowledge", "book-open", "Knowledge"),
    ("traces", "route", "Tool traces"),
    ("settings", "settings", "Settings"),
)


def nav_button(item: tuple[str, str, str]) -> rx.Component:
    key, icon, label = item
    return rx.button(
        rx.icon(icon, size=17),
        label,
        variant="ghost",
        class_name=rx.cond(
            StaffState.section == key,
            "staff-nav-button staff-nav-active",
            "staff-nav-button",
        ),
        on_click=lambda: StaffState.navigate(key),
        width="100%",
    )


def staff_sidebar() -> rx.Component:
    return rx.vstack(
        rx.box(brand_mark(), class_name="staff-brand"),
        rx.separator(),
        rx.text("Operations", class_name="nav-section-label"),
        rx.vstack(
            *[nav_button(item) for item in NAV_ITEMS],
            width="100%",
            spacing="1",
            class_name="staff-nav-list",
        ),
        rx.spacer(),
        class_name="staff-sidebar",
        align="stretch",
    )


def metric(label: str, value: Any, icon: str, accent: str) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.box(rx.icon(icon, size=19), class_name=f"metric-icon {accent}"),
            rx.spacer(),
            rx.text(label, class_name="metric-label"),
            width="100%",
            align="center",
        ),
        rx.heading(value, size="7"),
        class_name="metric",
    )


def table_header(*labels: str) -> rx.Component:
    return rx.table.header(rx.table.row(*[rx.table.column_header_cell(label) for label in labels]))


def order_row(row: dict[str, str]) -> rx.Component:
    return rx.table.row(
        rx.table.cell(row["id"]),
        rx.table.cell(row["items"]),
        rx.table.cell(row["fulfilment"]),
        rx.table.cell(row["total"]),
        rx.table.cell(row["created"]),
    )


def overview_view() -> rx.Component:
    return rx.vstack(
        rx.grid(
            metric("Confirmed orders", StaffState.order_count, "receipt-text", "green"),
            metric("Pending handovers", StaffState.handover_count, "users", "coral"),
            metric("Conversation sessions", StaffState.session_count, "messages-square", "blue"),
            columns=rx.breakpoints(initial="1", md="3"),
            gap="14px",
            width="100%",
        ),
        rx.box(
            rx.heading("Recent orders", size="5"),
            rx.table.root(
                table_header("Order", "Items", "Fulfilment", "Total", "Created"),
                rx.table.body(rx.foreach(StaffState.orders[:8], order_row)),
                width="100%",
                variant="surface",
            ),
            class_name="staff-section",
        ),
        width="100%",
        spacing="4",
        align="stretch",
    )


def orders_view() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.heading("Order history", size="5"),
            rx.spacer(),
            rx.button(rx.icon("download", size=16), "CSV", on_click=StaffState.download_orders_csv, variant="soft"),
            rx.button(rx.icon("braces", size=16), "JSON", on_click=StaffState.download_orders_json, variant="soft"),
            width="100%",
            wrap="wrap",
        ),
        rx.table.root(
            table_header("Order", "Items", "Fulfilment", "Total", "Created"),
            rx.table.body(rx.foreach(StaffState.orders, order_row)),
            width="100%",
            variant="surface",
        ),
        class_name="staff-section",
    )


def handover_ticket_row(row: dict[str, str]) -> rx.Component:
    return rx.button(
        rx.vstack(
            rx.hstack(
                rx.text("#", row["short_id"], class_name="ticket-list-id"),
                rx.spacer(),
                rx.badge(
                    rx.cond(row["status"] == "pending", "Queue " + row["queue"], "Resolved"),
                    color_scheme=rx.cond(row["status"] == "pending", "orange", "green"),
                    variant="soft",
                ),
                width="100%",
                align="center",
            ),
            rx.text(row["request"], class_name="ticket-list-request"),
            rx.hstack(
                rx.text(row["trigger"], class_name="ticket-list-meta"),
                rx.spacer(),
                rx.text(row["created"], class_name="ticket-list-meta"),
                width="100%",
            ),
            width="100%",
            spacing="2",
            align="start",
        ),
        on_click=lambda: StaffState.select_handover(row["id"]),
        class_name=rx.cond(
            StaffState.selected_handover == row["id"],
            "ticket-list-item ticket-list-item-active",
            "ticket-list-item",
        ),
        variant="ghost",
        width="100%",
    )


def ticket_conversation_message(message: dict[str, str]) -> rx.Component:
    is_customer = message["role"] == "customer"
    is_staff = message["role"] == "staff"
    return rx.box(
        rx.hstack(
            rx.text(
                rx.cond(is_customer, "Customer", rx.cond(is_staff, "Staff", "OrderFlow")),
                class_name="ticket-chat-role",
            ),
            rx.spacer(),
            rx.text(message["time"], class_name="ticket-chat-time"),
            width="100%",
        ),
        rx.text(message["content"], white_space="pre-wrap", class_name="ticket-chat-content"),
        class_name=rx.cond(
            is_customer,
            "ticket-chat-message ticket-chat-customer",
            rx.cond(is_staff, "ticket-chat-message ticket-chat-staff", "ticket-chat-message ticket-chat-agent"),
        ),
    )


def handovers_view() -> rx.Component:
    return rx.grid(
        rx.box(
            rx.hstack(
                rx.heading("Tickets", size="5"),
                rx.spacer(),
                rx.badge(StaffState.handover_count, " open", color_scheme="orange", variant="soft"),
                width="100%",
                align="center",
            ),
            rx.vstack(
                rx.foreach(StaffState.handovers, handover_ticket_row),
                width="100%",
                spacing="2",
                align="stretch",
            ),
            class_name="ticket-inbox",
        ),
        rx.box(
            rx.hstack(
                rx.vstack(
                    rx.text("Ticket #", StaffState.ticket_number, class_name="ticket-detail-id"),
                    rx.heading(StaffState.ticket_request, size="5"),
                    spacing="1",
                    align="start",
                ),
                rx.spacer(),
                rx.badge(
                    StaffState.ticket_queue,
                    color_scheme=rx.cond(StaffState.ticket_status == "pending", "orange", "green"),
                    variant="soft",
                ),
                width="100%",
                align="start",
                class_name="ticket-detail-header",
            ),
            rx.grid(
                rx.box(rx.text("Issue", class_name="metadata-label"), rx.text(StaffState.ticket_issue, class_name="metadata-value")),
                rx.box(rx.text("Cart", class_name="metadata-label"), rx.text(StaffState.ticket_cart, class_name="metadata-value")),
                rx.box(rx.text("Fulfilment", class_name="metadata-label"), rx.text(StaffState.ticket_fulfilment, class_name="metadata-value")),
                rx.box(rx.text("Context", class_name="metadata-label"), rx.text(StaffState.ticket_context, class_name="metadata-value")),
                columns=rx.breakpoints(initial="1", md="2"),
                gap="12px",
                class_name="ticket-metadata",
            ),
            rx.box(
                rx.vstack(
                    rx.foreach(StaffState.ticket_messages, ticket_conversation_message),
                    width="100%",
                    spacing="3",
                    align="stretch",
                ),
                class_name="ticket-chat-scroll",
            ),
            rx.form(
                rx.hstack(
                    rx.input(
                        name="reply",
                        value=StaffState.ticket_reply,
                        on_change=StaffState.set_ticket_reply,
                        placeholder="Reply to customer...",
                        width="100%",
                        disabled=StaffState.ticket_status != "pending",
                    ),
                    rx.tooltip(
                        rx.icon_button(
                            rx.icon("send", size=17),
                            type="submit",
                            disabled=StaffState.ticket_status != "pending",
                            aria_label="Send staff reply",
                        ),
                        content="Send reply",
                    ),
                    rx.button(
                        rx.icon("circle-check", size=16),
                        "Resolve ticket",
                        type="button",
                        on_click=StaffState.resolve_handover,
                        variant="soft",
                        disabled=StaffState.ticket_status != "pending",
                    ),
                    width="100%",
                    align="center",
                ),
                on_submit=StaffState.send_ticket_reply,
                reset_on_submit=True,
                class_name="ticket-composer",
            ),
            class_name="ticket-workspace",
            display=rx.cond(StaffState.ticket_number != "", "flex", "none"),
        ),
        rx.box(
            rx.icon("inbox", size=26),
            rx.text("No ticket selected"),
            class_name="ticket-empty",
            display=rx.cond(StaffState.ticket_number == "", "grid", "none"),
        ),
        columns=rx.breakpoints(initial="1", lg="320px minmax(0, 1fr)"),
        gap="18px",
        width="100%",
    )


def catalog_row(row: dict[str, str]) -> rx.Component:
    return rx.table.row(
        rx.table.cell(row["sku"]), rx.table.cell(row["name"]), rx.table.cell(row["category"]),
        rx.table.cell(row["price"]), rx.table.cell(row["active"]),
    )


def catalog_view() -> rx.Component:
    return rx.grid(
        rx.box(
            rx.heading("Catalogue", size="5"),
            rx.table.root(
                table_header("SKU", "Name", "Category", "Price", "Active"),
                rx.table.body(rx.foreach(StaffState.catalog_rows, catalog_row)),
                width="100%",
                variant="surface",
            ),
            class_name="staff-section",
        ),
        rx.box(
            rx.heading("Edit item", size="5"),
            rx.select(
                StaffState.catalog_options,
                placeholder="Choose catalogue item",
                on_change=StaffState.load_catalog_item,
                width="100%",
            ),
            rx.input(value=StaffState.catalog_sku, on_change=StaffState.set_catalog_sku, placeholder="SKU"),
            rx.input(value=StaffState.catalog_name, on_change=StaffState.set_catalog_name, placeholder="Name"),
            rx.hstack(
                rx.input(value=StaffState.catalog_category, on_change=StaffState.set_catalog_category, placeholder="Category"),
                rx.input(value=StaffState.catalog_price, on_change=StaffState.set_catalog_price, placeholder="Price", type="number"),
                width="100%",
            ),
            rx.text_area(value=StaffState.catalog_description, on_change=StaffState.set_catalog_description, placeholder="Description"),
            rx.input(value=StaffState.catalog_ingredients, on_change=StaffState.set_catalog_ingredients, placeholder="Ingredients, comma separated"),
            rx.input(value=StaffState.catalog_aliases, on_change=StaffState.set_catalog_aliases, placeholder="Aliases, comma separated"),
            rx.input(value=StaffState.catalog_tags, on_change=StaffState.set_catalog_tags, placeholder="Tags, comma separated"),
            rx.input(value=StaffState.catalog_image, on_change=StaffState.set_catalog_image, placeholder="Image path"),
            rx.checkbox("Active", checked=StaffState.catalog_active, on_change=StaffState.set_catalog_active),
            rx.button(rx.icon("save", size=16), "Save item", on_click=StaffState.save_catalog_item),
            class_name="staff-section compact-form",
        ),
        columns=rx.breakpoints(initial="1", xl="minmax(0, 1.25fr) minmax(320px, .75fr)"),
        gap="16px",
        width="100%",
    )


def recommendation_row(row: dict[str, str]) -> rx.Component:
    return rx.table.row(rx.table.cell(row["name"]), rx.table.cell(row["price"]), rx.table.cell(row["score"]), rx.table.cell(row["reason"]))


def intelligence_view() -> rx.Component:
    return rx.box(
        rx.heading("Menu intelligence", size="5"),
        rx.hstack(
            rx.input(value=StaffState.menu_query, on_change=StaffState.set_menu_query, placeholder="Describe a pizza preference", width="100%"),
            rx.icon_button(
                rx.icon("search", size=17),
                on_click=StaffState.rank_menu,
                aria_label="Rank menu",
            ),
            width="100%",
        ),
        rx.table.root(
            table_header("Item", "Price", "Score", "Reason"),
            rx.table.body(rx.foreach(StaffState.recommendations, recommendation_row)),
            width="100%",
            variant="surface",
        ),
        class_name="staff-section",
    )


def evaluation_row(row: dict[str, str]) -> rx.Component:
    return rx.table.row(
        rx.table.cell(row["mode"]), rx.table.cell(row["scenario"]), rx.table.cell(row["passed"]),
        rx.table.cell(row["turns"]), rx.table.cell(row["tools"]), rx.table.cell(row["handovers"]),
    )


def evaluation_view() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.heading("Agent evaluation", size="5"),
            rx.spacer(),
            rx.button(rx.icon("play", size=16), "Run scenarios", on_click=StaffState.run_scenarios, loading=StaffState.busy),
            width="100%",
        ),
        rx.callout(
            StaffState.evaluation_summary,
            icon="circle-check",
            color_scheme="green",
            display=rx.cond(StaffState.evaluation_summary != "", "flex", "none"),
        ),
        rx.text(StaffState.evaluation_path, class_name="path-text"),
        rx.table.root(
            table_header("Mode", "Scenario", "Result", "Turns", "Tools", "Handovers"),
            rx.table.body(rx.foreach(StaffState.evaluation_rows, evaluation_row)),
            width="100%",
            variant="surface",
        ),
        class_name="staff-section",
    )


def knowledge_view() -> rx.Component:
    return rx.box(
        rx.heading("Operational knowledge", size="5"),
        rx.upload(
            rx.vstack(
                rx.icon("cloud-upload", size=22),
                rx.text("Drop one text, Markdown, CSV, JSON, or text-based PDF here"),
                rx.foreach(rx.selected_files(KNOWLEDGE_UPLOAD_ID), lambda filename: rx.text(filename, class_name="path-text")),
                align="center",
                spacing="2",
            ),
            id=KNOWLEDGE_UPLOAD_ID,
            accept={
                "text/plain": [".txt"],
                "text/markdown": [".md"],
                "text/csv": [".csv"],
                "application/json": [".json"],
                "application/pdf": [".pdf"],
            },
            max_files=1,
            max_size=12 * 1024 * 1024,
            class_name="knowledge-drop",
        ),
        rx.hstack(
            rx.button(
                rx.icon("upload", size=16),
                "Add document",
                on_click=StaffState.upload_knowledge(rx.upload_files(upload_id=KNOWLEDGE_UPLOAD_ID)),
            ),
            rx.button(
                rx.icon("x", size=16),
                "Clear",
                on_click=rx.clear_selected_files(KNOWLEDGE_UPLOAD_ID),
                variant="soft",
            ),
        ),
        rx.table.root(
            table_header("Title", "Source", "Chunks", "Created"),
            rx.table.body(
                rx.foreach(
                    StaffState.knowledge_documents,
                    lambda row: rx.table.row(
                        rx.table.cell(row["title"]),
                        rx.table.cell(row["source"]),
                        rx.table.cell(row["chunks"]),
                        rx.table.cell(row["created"]),
                    ),
                )
            ),
            width="100%",
            variant="surface",
        ),
        rx.hstack(
            rx.input(value=StaffState.knowledge_query, on_change=StaffState.set_knowledge_query, placeholder="Ask an operating-policy question", width="100%"),
            rx.icon_button(
                rx.icon("search", size=17),
                on_click=StaffState.query_knowledge,
                aria_label="Search knowledge",
            ),
            width="100%",
        ),
        rx.text(StaffState.knowledge_answer, white_space="pre-wrap"),
        class_name="staff-section",
    )


def trace_row(row: dict[str, str]) -> rx.Component:
    return rx.table.row(rx.table.cell(row["session"]), rx.table.cell(row["steps"]), rx.table.cell(row["created"]))


def traces_view() -> rx.Component:
    return rx.box(
        rx.heading("Deterministic tool traces", size="5"),
        rx.table.root(
            table_header("Session", "Steps", "Created"),
            rx.table.body(rx.foreach(StaffState.traces, trace_row)),
            width="100%",
            variant="surface",
        ),
        class_name="staff-section",
    )


def settings_view() -> rx.Component:
    return rx.box(
        rx.heading("Model runtime", size="5"),
        rx.select(
            ["kernelloom", "openai", "huggingface", "openagent", "disabled"],
            value=StaffState.provider_id,
            on_change=StaffState.set_provider_id,
            width="100%",
        ),
        rx.input(value=StaffState.response_model, on_change=StaffState.set_response_model, placeholder="Model ID"),
        rx.input(value=StaffState.local_model_path, on_change=StaffState.set_local_model_path, placeholder="Local model path"),
        rx.input(value=StaffState.base_url, on_change=StaffState.set_base_url, placeholder="Provider base URL"),
        rx.input(value=StaffState.api_key, on_change=StaffState.set_api_key, placeholder="API key", type="password"),
        rx.select(["CPU", "GPU", "NPU", "AUTO"], value=StaffState.device, on_change=StaffState.set_device, width="100%"),
        rx.hstack(
            rx.button(rx.icon("save", size=16), "Apply", on_click=StaffState.apply_runtime),
            rx.button(rx.icon("activity", size=16), "Check runtime", on_click=StaffState.test_runtime, variant="soft", loading=StaffState.busy),
        ),
        rx.callout(
            StaffState.runtime_result,
            icon="info",
            size="1",
            display=rx.cond(StaffState.runtime_result != "", "flex", "none"),
        ),
        class_name="staff-section compact-form settings-form",
    )


def staff_content() -> rx.Component:
    return rx.box(
        rx.match(
            StaffState.section,
            ("overview", overview_view()),
            ("orders", orders_view()),
            ("handovers", handovers_view()),
            ("catalog", catalog_view()),
            ("intelligence", intelligence_view()),
            ("evaluation", evaluation_view()),
            ("knowledge", knowledge_view()),
            ("traces", traces_view()),
            ("settings", settings_view()),
            overview_view(),
        ),
        width="100%",
        class_name="staff-view",
    )


def staff_page() -> rx.Component:
    return rx.grid(
        staff_sidebar(),
        rx.box(
            rx.hstack(
                rx.vstack(rx.text("Restaurant workspace", class_name="eyebrow"), rx.heading(StaffState.section_title, size="7"), spacing="1", align="start"),
                rx.spacer(),
                rx.hstack(
                    rx.link(
                        rx.button(
                            rx.icon("store", size=16),
                            "Customer app",
                            variant="soft",
                            class_name="staff-customer-link",
                        ),
                        href="/",
                    ),
                    rx.tooltip(
                        rx.icon_button(
                            rx.icon("refresh-cw", size=18),
                            on_click=StaffState.refresh,
                            variant="soft",
                            aria_label="Refresh workspace",
                        ),
                        content="Refresh workspace",
                    ),
                    spacing="2",
                    class_name="topbar-actions",
                ),
                width="100%",
                align="center",
                class_name="staff-topbar",
            ),
            rx.callout(
                StaffState.notice,
                icon="info",
                size="1",
                display=rx.cond(StaffState.notice != "", "flex", "none"),
            ),
            staff_content(),
            class_name="staff-main",
        ),
        columns=rx.breakpoints(initial="1", lg="240px minmax(0, 1fr)"),
        class_name="staff-shell",
    )


app = rx.App(
    theme=rx.theme(appearance="light", accent_color="tomato", radius="small"),
    stylesheets=["/styles.css"],
)
app.add_page(
    customer_page,
    route="/",
    title="OrderFlow | Pizza ordering",
    description="Live pizza ordering with deterministic menu, pricing, and confirmation tools.",
    on_load=[CustomerState.initialize, CustomerState.monitor_handover],
)
app.add_page(
    staff_page,
    route="/staff",
    title="OrderFlow | Staff workspace",
    description="OrderFlow operational controls and evaluation workspace.",
    on_load=[StaffState.initialize, StaffState.monitor_workspace],
)
