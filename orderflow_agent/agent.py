"""Conversation orchestration for the operational OrderFlow task agent."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable, Iterable
from urllib.parse import unquote
from uuid import uuid4

from .catalog import Catalog, CatalogStore, JsonCatalogStore
from .context import ConversationContextAnalyzer, ConversationSignal
from .handover import HandoverCase, HandoverDecision, HandoverService
from .models import (
    AgentResponse,
    ConversationSession,
    MenuAttachment,
    OrderRecord,
    ToolStep,
)
from .modes import AgentMode, coerce_mode
from .policy import GuardedResponseComposer
from .storage import SQLiteStorageAdapter, StorageAdapter
from .tools import (
    add_items,
    answer_menu_question,
    asks_about_item_details,
    contains_any,
    extract_order_from_text,
    find_menu_matches,
    format_bill,
    format_order,
    generate_bill,
    invalid_order_quantity_reason,
    missing_order_details_hint,
    normalize_text,
    remove_items,
    specific_menu_query_terms,
    validate_order,
)

AFFIRMATIVE = {"yes", "yep", "yeah"}
NEGATIVE = {"no", "nope", "not", "wait", "edit", "change"}
CANCEL_WORDS = {"cancel", "clear", "abandon", "discard"}
CONFIRM_WORDS = {"confirm", "checkout", "final", "done", "place"}
REMOVE_WORDS = {"remove", "delete", "drop"}
MENU_WORDS = {"menu", "price", "prices", "cost", "available", "have", "show", "list", "deal", "deals"}
ORDER_WORDS = {"add", "order", "want", "need", "get", "take", "make", "buy"}
DELIVERY_WORDS = {"deliver", "delivery"}
PICKUP_WORDS = {"collect", "collection", "pickup", "pick-up"}
REPAIR_CUES = (
    "that is not what i meant",
    "i meant",
    "not that",
    "you misunderstood",
    "let me explain",
    "no,",
)
AFFIRMATIVE_FILLERS = {"please", "it", "the", "order", "now", "go", "ahead", "do", "so", "thanks", "thank", "you"}
NEGATIVE_FILLERS = AFFIRMATIVE_FILLERS | {"keep", "editing", "draft", "leave"}
PIZZA_SERVICE_WORDS = {
    "address", "available", "base", "bill", "cart", "cheese", "collect", "collection", "cost",
    "delivery", "dip", "drink", "employee", "ingredient", "ingredients", "menu", "order", "pepperoni",
    "pickup", "pizza", "price", "refund", "restaurant", "sauce", "size", "staff", "topping", "toppings",
    "total",
}
OFF_TOPIC_WORDS = {
    "algebra", "biology", "cake", "calculate", "calculus", "capital", "chemistry", "code", "coding", "discovered",
    "equation", "essay", "compose", "creative", "function", "geography", "history", "homework", "javascript",
    "math", "mathematics", "penicillin", "physics", "poem",
    "exploit", "hacking", "malware", "password", "payload", "politics", "president",
    "python", "recipe", "reverse", "science", "shell", "sonnet", "story", "translate", "translation", "weather",
}
COURTESY_WORDS = {"bye", "goodbye", "great", "ok", "okay", "please", "thanks", "thank", "you"}
NON_ORDERING_PIZZA_WORDS = {
    "dance", "embassy", "marry", "passport", "poetry", "school", "sing", "vote", "wedding",
}
ORDER_HISTORY_CUES = (
    "last order", "previous order", "old order", "past order", "order history", "order number",
    "order reference", "order ref", "bill from", "bill of", "yesterday", "there is an order",
    "there is order", "my confirmed order", "reorder", "order again", "same order",
)
REORDER_CUES = ("reorder", "order again", "same order", "order from it", "order pizza from it")
ORDER_REFERENCE_PATTERN = re.compile(
    r"\b(?:order\s*(?:number|no|reference|ref)?|reference|ref)\s*[:#-]?\s*([a-f0-9]{8}(?:-[a-f0-9-]{9,})?)\b"
    r"|\b([a-f0-9]{8})\b",
    re.IGNORECASE,
)
MAX_CUSTOMER_MESSAGE_LENGTH = 1200
_SECURITY_LEET_TRANSLATION = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
_SECURITY_CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "а": "a", "е": "e", "і": "i", "ј": "j", "к": "k", "м": "m", "о": "o", "р": "p",
        "с": "c", "т": "t", "у": "y", "х": "x", "ι": "i", "ο": "o",
    }
)
_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|override|forget|bypass)\b.{0,60}"
        r"\b(?:constraint|constraints|directive|directives|guideline|guidelines|instruction|instructions|"
        r"rule|rules|prompt|policy|policies|guardrail|guardrails|system|developer)\b"
    ),
    re.compile(
        r"\b(?:abandon|discard|set aside)\b.{0,45}"
        r"\b(?:constraints?|directives?|guidelines?|instructions?|policies|policy|rules?)\b"
    ),
    re.compile(
        r"\b(?:do not|dont|never|stop)\s+(?:following|follow|obeying|obey)\b.{0,45}"
        r"\b(?:constraints?|directives?|guidelines?|instructions?|policies|policy|rules?)\b"
    ),
    re.compile(
        r"\b(?:quote|recite|reveal|show|print|repeat|expose|leak)\b.{0,45}"
        r"\b(?:system|developer|hidden|internal)\b.{0,30}"
        r"\b(?:directive|directives|guideline|guidelines|policy|policies|prompt|instructions|message|rules)\b"
    ),
    re.compile(r"\b(?:jailbreak|do anything now|dan mode|unrestricted mode|developer mode)\b"),
    re.compile(r"\bact as\b.{0,35}\b(?:dan|system|developer|unrestricted|unfiltered)\b"),
    re.compile(
        r"\b(?:dump|expose|leak|list|output|print|reveal|show)\b.{0,45}"
        r"\b(?:api key|secret|credentials|environment variables?|env vars?|token)\b"
    ),
    re.compile(
        r"\b(?:decode|decrypt)\b.{0,45}\b(?:base6a|base64|rot13|encoded)\b.{0,55}"
        r"\b(?:follow|execute|obey|run)\b"
    ),
    re.compile(
        r"\b(?:system|developer|tool)\s*(?:message|prompt|output)?\s+.{0,70}"
        r"\b(?:answer|disable|ignore|new rule|override|policy|prompt|rules?)\b"
    ),
    re.compile(
        r"\b(?:from now on|new goal|new objective|new role|pretend)\b.{0,55}"
        r"\b(?:agent|assistant|bot|instructions?|policy|prompt|rules?|system)\b"
    ),
    re.compile(r"\byou are\b.{0,25}\b(?:no longer|now an?|now the)\b.{0,35}\b(?:agent|assistant|bot)\b"),
    re.compile(
        r"\b(?:begin|end|prefix|start|suffix)\b.{0,35}\b(?:answer|message|reply|response)\b"
        r".{0,25}\b(?:using|with)\b"
    ),
    re.compile(
        r"\b(?:answer|output|print|reply|respond|say|write)\b.{0,30}"
        r"\b(?:exactly|instead|only|verbatim)\b"
    ),
    re.compile(r"\brole\s+(?:assistant|developer|system|tool)\b"),
    re.compile(r"\b(?:assistant|developer|system|tool)\s+(?:message|output|result|role)\b"),
    re.compile(
        r"\b(?:assistant|bot|model)\b.{0,25}\b(?:must|should)\b.{0,25}"
        r"\b(?:answer|output|reply|respond|say|write)\b"
    ),
    re.compile(r"\b(?:become|simulate)\b.{0,35}\b(?:unfiltered|unrestricted)\b.{0,25}\b(?:assistant|bot|model)\b"),
    re.compile(
        r"\b(?:rules?|instructions?|directives?|policies|policy|constraints?|requirements?)\b"
        r".{0,45}\b(?:no longer apply|are optional|as optional|do not apply|dont apply|merely suggestions?)\b"
    ),
    re.compile(
        r"\b(?:follow|obey|execute)\b.{0,30}\b(?:my|these|new)\b.{0,25}"
        r"\b(?:commands?|instructions?|directions?)\b.{0,35}\b(?:instead|over|rather than)\b"
    ),
    re.compile(
        r"\b(?:temporary|one time|one-time|special)\b.{0,25}\b(?:exception|exemption)\b"
        r".{0,45}\b(?:rules?|instructions?|policy|constraints?)\b"
    ),
    re.compile(
        r"\b(?:ignore|disregard|override|forget|bypass)\b.{0,45}\b(?:prior|previous|original|operating)\b"
        r".{0,25}\b(?:requirements?|mandate|controls?)\b"
    ),
    re.compile(
        r"\bbase(?:64|6a)\b.{0,100}\b(?:execute|follow|obey|run)\b"
        r"|\b(?:execute|follow|obey|run)\b.{0,100}\bbase(?:64|6a)\b"
    ),
    re.compile(r"\bignora\b.{0,40}\binstrucciones\b|\bignorez\b.{0,40}\binstructions\b"),
    re.compile(
        r"\b(?:always|henceforth)\b.{0,25}\b(?:call|include|mention|refer|reply|respond|say|write)\b"
        r"|\brefer\s+to\s+me\s+as\b"
    ),
    re.compile(
        r"\b(?:supersede|suspend|invalidate)\b.{0,45}\b(?:instructions?|rules?|policy|constraints?)\b"
        r"|\bprioriti[sz]e\b.{0,45}\b(?:above|over)\b.{0,25}\b(?:system|developer|policy|rules?)\b"
    ),
    re.compile(
        r"\b(?:deactivate|disable|remove|switch off|turn off)\b.{0,35}"
        r"\b(?:checks?|constraints?|filters?|guards?|guardrails?|policies|policy|rules?|safety)\b"
    ),
    re.compile(
        r"\b(?:remember|memorize|store)\b.{0,35}\b(?:permanently|from now on|for future)\b.{0,70}"
        r"\b(?:override|replace|supersede|ignore)\b.{0,35}\b(?:catalog|menu|policy|price|prices|rules?)\b"
        r"|\b(?:remember|memorize|store)\b.{0,55}\b(?:customer|user)\b.{0,30}"
        r"\b(?:prices?|instructions?)\b.{0,30}\b(?:override|replace|supersede)\b"
    ),
    re.compile(
        r"\bignoriere\b.{0,45}\b(?:anweisungen|regeln)\b"
        r"|\bignora\b.{0,45}\b(?:istruzioni|regras)\b"
        r"|\bignore\b.{0,45}\b(?:instrucoes|instruções)\b"
    ),
    re.compile(
        r"\b(?:guardrails?|policies|policy|rules?|safety)\b.{0,18}"
        r"\b(?:are|is|status)?\s*(?:disabled|inactive|off|suspended)\b"
    ),
    re.compile(
        r"\bdo\b.{0,12}\bopposite\b.{0,35}\b(?:prior|previous|every|all)\b.{0,18}"
        r"\b(?:instructions?|rules?|policies|policy|constraints?)\b"
    ),
    re.compile(
        r"\b(?:enter|enable|switch to|use)\b.{0,20}\b(?:admin|debug|maintenance|test)\s+mode\b"
        r".{0,40}\b(?:no|without)\b.{0,12}\b(?:constraints?|filters?|guards?|restrictions?|rules?)\b"
    ),
)


def security_normalize(text: str) -> str:
    """Normalize common low-effort obfuscation without changing ordinary ordering text."""

    compatible = unicodedata.normalize("NFKC", text)
    for _ in range(3):
        decoded = unquote(html.unescape(compatible))
        if decoded == compatible:
            break
        compatible = decoded
    compatible = re.sub(r"<!--.*?-->", "", compatible, flags=re.DOTALL)
    compatible = re.sub(r"</?[^>\r\n]{1,80}>", "", compatible)
    compatible = "".join(character for character in compatible if unicodedata.category(character) != "Cf")
    normalized = normalize_text(
        compatible.casefold().translate(_SECURITY_CONFUSABLE_TRANSLATION).translate(_SECURITY_LEET_TRANSLATION)
    )

    def collapse_spaced_letters(match: re.Match[str]) -> str:
        return match.group(0).replace(" ", "")

    return re.sub(r"(?<![a-z])(?:[a-z]\s+){3,}[a-z](?![a-z])", collapse_spaced_letters, normalized)


def is_prompt_injection(text: str) -> bool:
    """Detect explicit attempts to replace or expose the agent's governing instructions."""

    if re.search(
        r"<\s*script\b|javascript\s*:|\bonerror\s*=|^\s*(?:assistant|developer|system|tool)\s*:"
        r"|(?:^|\s)[<\[]/?(?:assistant|developer|system|tool)(?:[_\s-](?:call|message|instruction|output))?[>\]]"
        r"|<!--(?:(?!-->).)*(?:developer|instruction|rules?|system|tool)(?:(?!-->).)*-->",
        text,
        re.IGNORECASE | re.MULTILINE,
    ):
        return True
    normalized = security_normalize(text)
    return any(pattern.search(normalized) for pattern in _PROMPT_INJECTION_PATTERNS)


def is_split_prompt_injection(parts: Iterable[str]) -> bool:
    """Detect an override phrase deliberately divided across recent customer turns."""

    fragments = tuple(part.strip() for part in parts if part.strip())[-5:]
    if len(fragments) < 2 or sum(map(len, fragments)) > 360:
        return False
    if any(is_prompt_injection(fragment) for fragment in fragments):
        return False
    return is_prompt_injection(" ".join(fragments))


def _confirmation_choice(text: str) -> str | None:
    tokens = set(normalize_text(text).split())
    affirmative = tokens.intersection(AFFIRMATIVE)
    negative = tokens.intersection(NEGATIVE)
    if affirmative and not negative and tokens <= AFFIRMATIVE | AFFIRMATIVE_FILLERS:
        return "yes"
    if negative and not affirmative and tokens <= NEGATIVE | NEGATIVE_FILLERS:
        return "no"
    return None


class ConversationalTaskAgent:
    """Natural conversation around catalog-owned and confirmation-gated tools."""

    def __init__(
        self,
        *,
        catalog_store: CatalogStore | None = None,
        storage: StorageAdapter | None = None,
        provider: object | None = None,
        knowledge_responder: Callable[[str], tuple[str, tuple[ToolStep, ...]] | None] | None = None,
    ) -> None:
        self.catalog_store = catalog_store or JsonCatalogStore()
        self.storage = storage or SQLiteStorageAdapter()
        self.composer = GuardedResponseComposer(provider)
        self.knowledge_responder = knowledge_responder
        self.context_analyzer = ConversationContextAnalyzer()
        self.handover_service = HandoverService(self.storage)

    @property
    def catalog(self) -> Catalog:
        return self.catalog_store.load()

    def open_session(
        self,
        strictness: int = 50,
        session_id: str | None = None,
        mode: AgentMode | str | None = None,
    ) -> ConversationSession:
        selected = coerce_mode(mode) if mode is not None else coerce_mode(strictness)
        session = ConversationSession(session_id=session_id or str(uuid4()), strictness=selected.strictness)
        self.storage.ensure_session(session)
        return session

    def welcome(self, session: ConversationSession) -> AgentResponse:
        return AgentResponse(
            "Hi there. What can I get started for you?",
            (
                ToolStep("open_session", "passed", f"Session uses {session.agent_mode.value.upper()} agent mode."),
                ToolStep("load_catalog", "passed", f"{len(self.catalog.active_items)} active items loaded."),
            ),
            menu_attachments=self._featured_menu_attachments(),
        )

    def handle(self, message: str, session: ConversationSession, channel: str = "text") -> AgentResponse:
        text = message.strip()
        session.turn_count += 1
        is_repair = any(cue in normalize_text(text) for cue in REPAIR_CUES)
        if is_repair:
            session.repair_requests += 1
            session.failed_attempts += 1
        if not text:
            return self._finalize(
                text,
                session,
                AgentResponse("Tell me what you would like to order, or ask to see the menu."),
                channel,
            )

        signal = self._conversation_signal(text, session)
        if session.handover_active:
            response = AgentResponse(
                "Your request is already in the staff queue. The conversation and cart are saved, and ordering "
                "actions will stay paused until the restaurant team responds.",
                (
                    ToolStep(
                        "handover_lock",
                        "blocked",
                        f"Automated action blocked while handover {session.handover_case_id[:8]} is pending.",
                    ),
                ),
                handover_requested=True,
                conversation_signal=signal,
                handover_case_id=session.handover_case_id or None,
            )
            session.last_user_message = text
            return self._finalize(text, session, response, channel)
        decision = self.handover_service.decide(text, session, signal)
        if decision.should_handover:
            session.last_user_message = text
            return self._finalize(text, session, self._handover_response(signal, decision), channel)

        if len(text) > MAX_CUSTOMER_MESSAGE_LENGTH:
            response = AgentResponse(
                "That message is too long to process safely. Please send the pizza request in a shorter message.",
                (
                    ToolStep(
                        "input_length_guard",
                        "blocked",
                        f"Customer input exceeded {MAX_CUSTOMER_MESSAGE_LENGTH} characters.",
                    ),
                ),
            )
            return self._finalize(
                text[:MAX_CUSTOMER_MESSAGE_LENGTH] + " [truncated]",
                session,
                response,
                channel,
            )

        recent_customer_turns = tuple(
            turn.content
            for turn in self.storage.list_turns(session.session_id)
            if turn.role == "customer"
        )[-4:]
        split_injection = is_split_prompt_injection((*recent_customer_turns, text))
        if is_prompt_injection(text) or split_injection:
            collecting_address = session.pending_action == "collect_delivery_address"
            response = AgentResponse(
                (
                    "That text was not saved as the delivery address. Please send only the street, building, and area details."
                    if collecting_address
                    else "I can help with the pizza menu or your order, but I cannot change or disclose the rules that protect ordering and customer data."
                ),
                (
                    ToolStep(
                        "prompt_injection_guard",
                        "blocked",
                        (
                            "A cross-turn instruction-manipulation sequence was blocked before tool execution."
                            if split_injection
                            else "Instruction-manipulation content was blocked before intent extraction or tool execution."
                        ),
                    ),
                ),
            )
            response = self._apply_language_layer("pizza-domain security redirect", session, response)
            return self._finalize(text, session, response, channel)

        intent_conflict = self._intent_conflict(text, session)
        if intent_conflict:
            if session.pending_action in {"confirm_order", "confirm_cancel"}:
                session.confirmation_failures += 1
                session.failed_attempts += 1
            response = AgentResponse(
                intent_conflict,
                (
                    ToolStep(
                        "validate_intent",
                        "blocked",
                        "Conflicting consequential instructions were rejected without changing the draft.",
                    ),
                ),
            )
        elif self._is_explicitly_off_topic(text):
            response = self._domain_guard_response()
        else:
            fulfilment_response = self._handle_fulfilment_turn(text, session)
            if fulfilment_response is not None:
                response = fulfilment_response
            else:
                pending = self._handle_pending(text, session)
                if pending is not None:
                    response = pending
                else:
                    intent = self._classify(text)
                    if intent == "cancel":
                        response = self._start_cancel(session)
                    elif intent == "confirm":
                        response = self._start_confirmation(session)
                    elif intent == "bill":
                        response = self._show_bill(session)
                    elif intent == "order_lookup":
                        response = self._lookup_order(text, session)
                    elif intent == "remove":
                        response = self._remove_from_order(text, session)
                    elif intent == "order":
                        response = self._add_to_order(text, session)
                    elif intent == "greeting":
                        response = self.welcome(session)
                    else:
                        response = self._answer_enquiry(text, session)

        response = self._apply_language_layer(text, session, response)

        if is_repair and response.content:
            session.successful_repairs += 1
            response = AgentResponse(
                "Thanks for correcting me. " + response.content,
                (
                    ToolStep(
                        "conversation_repair",
                        "passed",
                        "Correction acknowledged and the request was reprocessed.",
                    ),
                    *response.tool_trace,
                ),
                confirmed_order_id=response.confirmed_order_id,
                handover_requested=response.handover_requested,
                conversation_signal=signal,
                handover_decision=response.handover_decision,
                handover_case_id=response.handover_case_id,
                menu_attachments=response.menu_attachments,
            )
        else:
            response = AgentResponse(
                response.content,
                response.tool_trace,
                confirmed_order_id=response.confirmed_order_id,
                handover_requested=response.handover_requested,
                conversation_signal=signal,
                handover_decision=response.handover_decision,
                handover_case_id=response.handover_case_id,
                menu_attachments=response.menu_attachments,
            )
        post_tool_signal = self._conversation_signal(text, session)
        post_tool_decision = self.handover_service.decide(text, session, post_tool_signal)
        if post_tool_decision.should_handover and not response.handover_requested:
            escalation = self._handover_response(post_tool_signal, post_tool_decision)
            response = AgentResponse(
                response.content + "\n\n" + escalation.content,
                (*response.tool_trace, *escalation.tool_trace),
                confirmed_order_id=response.confirmed_order_id,
                handover_requested=True,
                conversation_signal=post_tool_signal,
                handover_decision=post_tool_decision,
                menu_attachments=response.menu_attachments,
            )
        session.last_user_message = text
        return self._finalize(text, session, response, channel)

    def create_handover(
        self,
        session: ConversationSession,
        *,
        customer_state: str,
        unresolved_issue: str,
    ) -> tuple[HandoverCase, AgentResponse]:
        state = customer_state.strip() or "Concerned"
        issue = unresolved_issue.strip() or "Customer requested restaurant staff."
        signal_label = {
            "frustrated": "frustrated",
            "worried": "urgent",
            "confused": "confused",
            "disappointed": "frustrated",
        }.get(state.casefold(), "neutral")
        signal = ConversationSignal(
            signal_label,
            0.9,
            f"The customer selected '{state}' in the handover form.",
            ("customer-handover-form",),
            method="customer-provided",
        )
        decision = HandoverDecision(
            True,
            "The customer requested restaurant staff through the handover control.",
            1.0,
            "explicit_human_request",
        )
        response = self._handover_response(signal, decision)
        self.storage.record_signal(session.session_id, signal)
        self.storage.record_exchange(session, "Requested restaurant staff: " + issue, response)
        case = self.handover_service.create_case(
            session=session,
            decision=decision,
            signal=signal,
            customer_request=issue,
            issue=issue,
        )
        session.handover_active = True
        session.handover_case_id = case.id
        session.pending_action = "handover"
        self.storage.ensure_session(session)
        return case, AgentResponse(
            response.content,
            (*response.tool_trace, ToolStep("queue_handover", "passed", f"Handover {case.id[:8]} is pending.")),
            handover_requested=True,
            conversation_signal=signal,
            handover_decision=decision,
            handover_case_id=case.id,
        )

    @staticmethod
    def _handover_response(signal: ConversationSignal, decision: HandoverDecision) -> AgentResponse:
        return AgentResponse(
            "I'm transferring this request to restaurant staff. I've kept the conversation, current cart, "
            "and tool history so you do not need to start again.",
            (
                ToolStep("evaluate_handover", "passed", f"Matched trigger: {decision.trigger}."),
                ToolStep(
                    "pause_automated_resolution",
                    "passed",
                    "The ordering agent stopped attempting consequential actions.",
                ),
                ToolStep(
                    "preserve_handover_state",
                    "passed",
                    "Conversation, cart, and tool history will be attached.",
                ),
            ),
            handover_requested=True,
            conversation_signal=signal,
            handover_decision=decision,
        )

    def _handle_pending(self, text: str, session: ConversationSession) -> AgentResponse | None:
        choice = _confirmation_choice(text)
        if session.pending_action == "collect_order_reference":
            if self._extract_order_reference(text):
                return self._lookup_order(text, session)
            if set(normalize_text(text).split()).intersection({"context", "explain", "clarify"}):
                session.pending_action = "none"
                return None
            if self._is_menu_browse_request(text) or extract_order_from_text(text, self.catalog):
                session.pending_action = "none"
                return None
            return AgentResponse(
                "Please send the eight-character order number so I can retrieve the correct order. You can also ask to see the menu instead.",
                (ToolStep("lookup_order", "blocked", "The follow-up still did not contain an order reference."),),
            )
        if session.pending_action == "confirm_order":
            if choice == "yes":
                return self._complete_order(session)
            if choice == "no":
                session.pending_action = "none"
                return AgentResponse(
                    "No problem. The draft remains open for changes.",
                    (ToolStep("hold_order", "passed", "No order was placed."),),
                )
            session.confirmation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(
                "Please reply yes to place this exact order, or no to keep editing.",
                (ToolStep("confirmation_gate", "blocked", "A clear yes or no is required."),),
            )
        if session.pending_action == "confirm_cancel":
            if choice == "yes":
                session.order.clear()
                session.pending_action = "none"
                return AgentResponse(
                    "The draft order has been cleared.",
                    (ToolStep("cancel_draft", "passed", "Draft state cleared after explicit confirmation."),),
                )
            if choice == "no":
                session.pending_action = "none"
                return AgentResponse(
                    "I kept the draft order open.",
                    (ToolStep("keep_draft", "passed", "Cancellation was not executed."),),
                )
            session.confirmation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(
                "Please reply yes to continue or no to return to the draft.",
                (ToolStep("confirmation_gate", "blocked", "A clear yes or no is required."),),
            )
        return None

    def _handle_fulfilment_turn(
        self,
        text: str,
        session: ConversationSession,
    ) -> AgentResponse | None:
        normalized = normalize_text(text)
        tokens = set(normalized.split())
        selected: str | None = None
        if tokens.intersection(DELIVERY_WORDS):
            selected = "delivery"
        elif tokens.intersection(PICKUP_WORDS):
            selected = "pickup"

        if selected is not None:
            contains_order = bool(extract_order_from_text(text, self.catalog)) or contains_any(text, ORDER_WORDS)
            if session.pending_action in {"confirm_order", "confirm_cancel"} and contains_order:
                return AgentResponse(
                    "I did not change the draft because that message combines a pending approval with another order action. Reply yes or no first, then make the change.",
                    (
                        ToolStep(
                            "validate_intent",
                            "blocked",
                            "A fulfilment change and order mutation were combined during a pending approval.",
                        ),
                    ),
                )
            session.fulfilment = selected
            session.pending_action = "none"
            if selected == "pickup":
                session.delivery_address = ""
            address = self._extract_delivery_address(text) if selected == "delivery" else ""
            if address:
                session.delivery_address = address
            if contains_order:
                return None
            if not session.order:
                return AgentResponse(
                    f"{selected.title()} selected. Which pizza and size would you like?",
                    (ToolStep("set_fulfilment", "passed", f"Draft fulfilment set to {selected}."),),
                )
            return self._fulfilment_saved_response(session)

        if session.pending_action == "collect_delivery_address":
            address = self._extract_delivery_address(text, pending=True)
            if address:
                session.delivery_address = address
                valid, reason = validate_order(session.order, self.catalog)
                if not valid:
                    session.pending_action = "none"
                    session.validation_failures += 1
                    session.failed_attempts += 1
                    return AgentResponse(reason, (ToolStep("validate_order", "blocked", reason),))
                session.pending_action = "confirm_order"
                return AgentResponse(
                    "Delivery address saved: "
                    + address
                    + ".\n\n"
                    + format_bill(session.order, self.catalog)
                    + "\n\nReply `yes` to place this delivery order, or `no` to keep editing.",
                    (
                        ToolStep("validate_delivery_address", "passed", "A non-empty delivery address was provided."),
                        ToolStep("set_delivery_address", "passed", "The address is attached to this draft."),
                        ToolStep("validate_order", "passed", reason),
                        ToolStep("generate_bill", "passed", "Totals calculated from catalog prices."),
                        ToolStep("confirmation_gate", "info", "Waiting for a separate explicit yes or no."),
                    ),
                    menu_attachments=self._menu_attachments(session.order),
                )
            return AgentResponse(
                "Please send the delivery address for this order, or say `pickup` instead.",
                (ToolStep("validate_delivery_address", "blocked", "No address-like text was found."),),
                menu_attachments=self._menu_attachments(session.order),
            )

        if session.pending_action == "collect_fulfilment":
            return AgentResponse(
                "Would you like delivery or pickup for this order?",
                (ToolStep("set_fulfilment", "blocked", "Delivery or pickup is required before confirmation."),),
                menu_attachments=self._menu_attachments(session.order),
            )
        return None

    def _fulfilment_saved_response(self, session: ConversationSession) -> AgentResponse:
        if session.fulfilment == "delivery" and not session.delivery_address:
            session.pending_action = "collect_delivery_address"
            content = "Delivery selected. What delivery address should I use?"
        else:
            content = (
                f"{session.fulfilment.title()} selected.\n\n"
                + format_bill(session.order, self.catalog)
                + "\n\nSay `confirm order` when this summary is correct."
            )
        return AgentResponse(
            content,
            (ToolStep("set_fulfilment", "passed", f"Draft fulfilment set to {session.fulfilment}."),),
            menu_attachments=self._menu_attachments(session.order),
        )

    @staticmethod
    def _extract_delivery_address(text: str, *, pending: bool = False) -> str:
        candidate = text.strip()
        explicit = re.search(r"\b(?:deliver(?:y)?\s+to|address(?:\s+is)?(?:\s*:)?)[\s,]+(.+)", candidate, re.I)
        if explicit:
            candidate = explicit.group(1)
        elif not pending:
            return ""
        candidate = re.split(
            r"[?.!]\s*(?:how much|what(?:'s| is) the (?:price|total)|show (?:me )?the bill).*$",
            candidate,
            maxsplit=1,
            flags=re.I,
        )[0]
        candidate = re.sub(r"\b(?:delivery|deliver|please)\b", " ", candidate, flags=re.I)
        candidate = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")
        if ConversationalTaskAgent._address_contains_transaction_facts(candidate):
            return ""
        if len(candidate) < 6 or len(candidate.split()) < 2:
            return ""
        if pending and not re.search(r"\d", candidate):
            return ""
        return candidate[:240]

    @staticmethod
    def _address_contains_transaction_facts(value: str) -> bool:
        canonical = unquote(html.unescape(unicodedata.normalize("NFKC", value)))
        canonical = "".join(
            character for character in canonical if unicodedata.category(character) != "Cf"
        )
        canonical = canonical.casefold().translate(_SECURITY_CONFUSABLE_TRANSLATION)
        semantic = security_normalize(value)
        return bool(
            re.search(r"\b(?:pkr|usd|gbp|eur)\s*[\d,.]+\b|[$£€]\s*[\d,.]+", canonical, re.I)
            or re.search(
                r"\b(?:grand\s+total|total|price|cost|bill)\b.{0,20}\b(?:free|zero|\d+)\b",
                canonical,
                re.I,
            )
            or re.search(
                r"\b(?:grand\s+total|total|price|cost|bill)\b.{0,20}"
                r"\b(?:free|zero|nothing|complimentary|waived)\b",
                semantic,
                re.I,
            )
            or re.search(
                r"\b(?:cart|order)\b.{0,20}\b(?:completed|confirmed|free|paid|placed|submitted)\b",
                semantic,
                re.I,
            )
            or re.search(
                r"\b(?:discount|coupon|promo(?:tional)?(?:\s+code)?|subtotal|tax|tip|amount)\b"
                r".{0,30}\b(?:free|zero|\d+|percent|percentage|code)\b|\d+\s*%",
                canonical,
                re.I,
            )
            or re.search(
                r"\b(?:order|payment|transaction)\s*(?:number|no|reference|ref|id|status)\b",
                semantic,
                re.I,
            )
        )

    @staticmethod
    def _intent_conflict(text: str, session: ConversationSession) -> str:
        normalized = normalize_text(text)
        tokens = set(normalized.split())
        has_delivery = bool(tokens.intersection(DELIVERY_WORDS))
        has_pickup = bool(tokens.intersection(PICKUP_WORDS))
        if has_delivery and has_pickup:
            return "Please choose one fulfilment method: delivery or pickup. I have not changed the draft."

        has_remove = bool(tokens.intersection(REMOVE_WORDS))
        has_add = bool(tokens.intersection({"add", "buy"}))
        if has_remove and has_add:
            return "Please send the removal and addition as separate messages so I can apply each cart change exactly."

        has_cart_mutation = has_remove or has_add
        if session.pending_action == "confirm_order" and has_cart_mutation:
            return "Please reply yes to place this exact order, or no to keep editing. I did not apply the requested cart change."
        if session.pending_action == "confirm_cancel" and has_cart_mutation:
            return "Please reply yes to clear the draft, or no to keep it. I did not apply the requested cart change."
        has_cancel = bool(tokens.intersection(CANCEL_WORDS))
        has_confirm = bool(tokens.intersection(CONFIRM_WORDS))
        if has_cart_mutation and (has_cancel or has_confirm):
            return "Please send the cart change separately from confirmation or cancellation. I have not changed the draft."

        return ""

    def _classify(self, text: str) -> str:
        tokens = set(normalize_text(text).split())
        extracted = extract_order_from_text(text, self.catalog)
        if contains_any(text, CANCEL_WORDS):
            return "cancel"
        if self._is_order_history_request(text):
            return "order_lookup"
        if "bill" in tokens or "total" in tokens:
            return "bill"
        if contains_any(text, REMOVE_WORDS):
            return "remove"
        if contains_any(text, MENU_WORDS):
            return "enquiry"
        if extracted:
            return "order"
        if contains_any(text, CONFIRM_WORDS):
            return "confirm"
        if contains_any(text, ORDER_WORDS):
            return "order"
        if tokens.intersection({"hi", "hello", "hey"}):
            return "greeting"
        return "enquiry"

    @staticmethod
    def _extract_order_reference(text: str) -> str:
        match = ORDER_REFERENCE_PATTERN.search(text)
        if match is None:
            return ""
        return (match.group(1) or match.group(2) or "").strip()

    @classmethod
    def _is_order_history_request(cls, text: str) -> bool:
        normalized = normalize_text(text)
        return bool(cls._extract_order_reference(text) or any(cue in normalized for cue in ORDER_HISTORY_CUES))

    @staticmethod
    def _is_menu_browse_request(text: str) -> bool:
        if asks_about_item_details(text):
            return False
        normalized = normalize_text(text)
        tokens = set(normalized.split())
        return bool(
            tokens.intersection({"menu", "options", "choices", "available"})
            or ("pizza" in tokens and bool(tokens.intersection({"which", "what", "have"})))
            or any(
                phrase in normalized
                for phrase in (
                    "what do you have", "what you have", "what have you got", "what you got",
                    "what can i get", "anything good", "show me what", "what is there",
                )
            )
        )

    def _add_to_order(self, text: str, session: ConversationSession) -> AgentResponse:
        items = extract_order_from_text(text, self.catalog)
        trace = [ToolStep("extract_order", "passed" if items else "blocked", f"Matched {len(items)} catalog item(s).")]
        if not items:
            session.validation_failures += 1
            session.failed_attempts += 1
            session.unsupported_attempts += 1
            matches = find_menu_matches(text, self.catalog, limit=3)
            normalized_tokens = set(normalize_text(text).split())
            if "pizza" in normalized_tokens and specific_menu_query_terms(text) and not matches:
                session.menu_context = ()
            suggestion = (
                " Closest catalog choices: " + ", ".join(item.name for item in matches) + "."
                if matches
                else ""
            )
            return AgentResponse(
                missing_order_details_hint(text)
                or "I could not match that to one active catalog pizza. Tell me the pizza name and size."
                + suggestion,
                tuple(trace),
                menu_attachments=self._menu_attachments(item.name for item in matches),
            )
        quantity_error = invalid_order_quantity_reason(text, self.catalog)
        if quantity_error:
            session.validation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(
                quantity_error,
                tuple([*trace, ToolStep("validate_order", "blocked", quantity_error)]),
                menu_attachments=self._menu_attachments(items),
            )
        prospective_order = add_items(session.order, items, self.catalog)
        valid, reason = validate_order(prospective_order, self.catalog)
        if not valid:
            session.validation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(
                reason,
                tuple([*trace, ToolStep("validate_order", "blocked", reason)]),
                menu_attachments=self._menu_attachments(items),
            )
        session.order = prospective_order
        if session.fulfilment == "undecided":
            session.pending_action = "none"
            next_step = "Would you like delivery or pickup?"
        elif session.fulfilment == "delivery" and not session.delivery_address:
            session.pending_action = "collect_delivery_address"
            next_step = "What delivery address should I use?"
        else:
            session.pending_action = "none"
            next_step = "Say `confirm order` when it looks right."
        approved = (
            "Added to the draft:\n"
            + format_order(items)
            + "\n\nCurrent draft:\n"
            + format_order(session.order)
            + "\n\n"
            + next_step
        )
        reply, policy_trace = self.composer.compose(
            strictness=session.strictness,
            user_message=text,
            approved_reply=approved,
            immutable_terms=tuple(items),
            recent_context=self._recent_context(session),
        )
        return AgentResponse(
            reply,
            tuple([*trace, ToolStep("update_draft", "passed", "Draft changed in session state."), *policy_trace]),
            menu_attachments=self._menu_attachments(items),
        )

    def _remove_from_order(self, text: str, session: ConversationSession) -> AgentResponse:
        if not session.order:
            session.validation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(
                "There is no draft order to edit yet.",
                (ToolStep("load_draft", "blocked", "Draft is empty."),),
            )
        items = extract_order_from_text(text, self.catalog)
        if not items:
            session.validation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(
                "Tell me the catalog item to remove, for example `remove one ranch dip`.",
                (ToolStep("extract_order", "blocked", "No removable catalog item matched."),),
            )
        quantity_error = invalid_order_quantity_reason(text, self.catalog)
        if quantity_error:
            session.validation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(
                quantity_error,
                (
                    ToolStep("extract_order", "passed", f"Matched {len(items)} item(s)."),
                    ToolStep("validate_order", "blocked", quantity_error),
                ),
                menu_attachments=self._menu_attachments(items),
            )
        excessive_removals = {
            name: quantity
            for name, quantity in items.items()
            if quantity > session.order.get(name, 0)
        }
        if excessive_removals:
            detail = ", ".join(
                f"{name} has {session.order.get(name, 0)} in the draft"
                for name in excessive_removals
            )
            session.validation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(
                "I did not change the draft because the requested removal exceeds its current quantity. " + detail + ".",
                (
                    ToolStep("extract_order", "passed", f"Matched {len(items)} item(s)."),
                    ToolStep("validate_order", "blocked", "Removal quantity exceeded the current draft quantity."),
                ),
                menu_attachments=self._menu_attachments(items),
            )
        session.order = remove_items(session.order, items)
        session.pending_action = "none"
        return AgentResponse(
            "Updated draft:\n" + format_order(session.order),
            (
                ToolStep("extract_order", "passed", f"Matched {len(items)} item(s)."),
                ToolStep("update_draft", "passed", "Requested quantities were removed."),
            ),
            menu_attachments=self._menu_attachments(session.order),
        )

    def _show_bill(self, session: ConversationSession) -> AgentResponse:
        if not session.order:
            return AgentResponse(
                "There is no draft order yet.",
                (ToolStep("generate_bill", "blocked", "Draft is empty."),),
            )
        return AgentResponse(
            "Current bill:\n\n"
            + format_bill(session.order, self.catalog)
            + f"\n\nFulfilment: {session.fulfilment}."
            + (f" Delivery address: {session.delivery_address}." if session.delivery_address else ""),
            (ToolStep("generate_bill", "passed", "Totals calculated from current catalog prices."),),
            menu_attachments=self._menu_attachments(session.order),
        )

    def _lookup_order(self, text: str, session: ConversationSession) -> AgentResponse:
        reference = self._extract_order_reference(text)
        if not reference:
            session.pending_action = "collect_order_reference"
            return AgentResponse(
                "I can look up a confirmed order using its eight-character order number. Please send that order number; a date alone is not enough to identify an order safely.",
                (ToolStep("lookup_order", "blocked", "A unique order reference is required."),),
            )
        order = self.storage.find_order(reference)
        if order is None:
            session.pending_action = "collect_order_reference"
            return AgentResponse(
                f"I could not find an order with reference {reference.upper()}. Check the eight-character order number and send it again.",
                (ToolStep("lookup_order", "blocked", "No unique persisted order matched the supplied reference."),),
            )

        session.pending_action = "none"
        lines = order.get("lines", [])
        item_names = [str(line["item"]) for line in lines]
        line_summary = ", ".join(f"{line['quantity']} x {line['item']}" for line in lines) or "No items"
        reference_text = str(order["id"])[:8].upper()
        details = (
            f"Order {reference_text} was {order['status']} on {str(order['created_at'])[:10]}. "
            f"Items: {line_summary}. Total: {order['currency']} {int(order['total']):,}. "
            f"Fulfilment: {order['fulfilment']}."
        )
        if order.get("delivery_address"):
            details += f" Delivery address: {order['delivery_address']}."

        normalized = normalize_text(text)
        if any(cue in normalized for cue in REORDER_CUES):
            active_items = {
                str(line["item"]): int(line["quantity"])
                for line in lines
                if str(line["item"]) in self.catalog.by_name and int(line["quantity"]) > 0
            }
            reordered_draft = add_items(session.order, active_items, self.catalog)
            valid, reason = validate_order(reordered_draft, self.catalog)
            if not valid:
                session.validation_failures += 1
                session.failed_attempts += 1
                return AgentResponse(
                    details + " I did not copy it because combining it with the current draft would violate an order limit.",
                    (
                        ToolStep("lookup_order", "passed", f"Matched persisted order {reference_text}."),
                        ToolStep("validate_order", "blocked", reason),
                    ),
                    menu_attachments=self._menu_attachments(active_items),
                )
            session.order = reordered_draft
            session.fulfilment = str(order["fulfilment"]) if order["fulfilment"] in {"delivery", "pickup"} else "undecided"
            session.delivery_address = ""
            session.pending_action = "collect_delivery_address" if session.fulfilment == "delivery" else "none"
            next_step = (
                "Please provide a delivery address for this new draft."
                if session.fulfilment == "delivery"
                else "Say confirm order when the new draft looks right."
            )
            return AgentResponse(
                details + " I copied its currently available items into a new draft. " + next_step,
                (
                    ToolStep("lookup_order", "passed", f"Matched persisted order {reference_text}."),
                    ToolStep("reorder_from_history", "passed", "Available historical items were copied into a new draft."),
                ),
                menu_attachments=self._menu_attachments(active_items),
            )
        return AgentResponse(
            details,
            (ToolStep("lookup_order", "passed", f"Matched persisted order {reference_text}."),),
            menu_attachments=self._menu_attachments(item_names),
        )

    def _start_confirmation(self, session: ConversationSession) -> AgentResponse:
        valid, reason = validate_order(session.order, self.catalog)
        if not valid:
            session.validation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(reason, (ToolStep("validate_order", "blocked", reason),))
        if session.fulfilment == "undecided":
            session.fulfilment = "pickup"
        if session.fulfilment == "delivery" and not session.delivery_address:
            session.pending_action = "collect_delivery_address"
            return AgentResponse(
                "What delivery address should I attach before I prepare the final confirmation?",
                (ToolStep("validate_delivery_address", "blocked", "A delivery address is required."),),
                menu_attachments=self._menu_attachments(session.order),
            )
        session.pending_action = "confirm_order"
        return AgentResponse(
            "Ready to place this order:\n\n"
            + format_bill(session.order, self.catalog)
            + f"\n\nFulfilment: {session.fulfilment}."
            + (f" Delivery address: {session.delivery_address}." if session.delivery_address else "")
            + "\n\nReply `yes` to place it, or `no` to keep editing.",
            (
                ToolStep("validate_order", "passed", reason),
                ToolStep("generate_bill", "passed", "Final bill generated from catalog prices."),
                ToolStep("confirmation_gate", "info", "Waiting for a separate explicit yes or no."),
            ),
            menu_attachments=self._menu_attachments(session.order),
        )

    def _complete_order(self, session: ConversationSession) -> AgentResponse:
        valid, reason = validate_order(session.order, self.catalog)
        if not valid:
            session.pending_action = "none"
            session.validation_failures += 1
            session.failed_attempts += 1
            return AgentResponse(reason, (ToolStep("validate_order", "blocked", reason),))
        bill = generate_bill(session.order, self.catalog)
        attachments = self._menu_attachments(session.order)
        fulfilment = "delivery" if session.fulfilment == "delivery" else "pickup"
        delivery_address = session.delivery_address if fulfilment == "delivery" else ""
        record = OrderRecord(
            session.session_id,
            bill.lines,
            bill.grand_total,
            bill.currency,
            fulfilment=fulfilment,
            delivery_address=delivery_address,
        )
        session.confirmed_orders += 1
        session.failed_attempts = 0
        session.validation_failures = 0
        self.storage.ensure_session(session)
        self.storage.save_order(record)
        session.order.clear()
        session.pending_action = "none"
        session.fulfilment = "undecided"
        session.delivery_address = ""
        return AgentResponse(
            "Order confirmed. Reference `"
            + record.id[:8].upper()
            + "`.\n\n"
            + self._format_final_bill(bill)
            + f"\n\nFulfilment: {fulfilment}."
            + (f" Delivery address: {delivery_address}." if delivery_address else ""),
            (
                ToolStep("validate_order", "passed", reason),
                ToolStep("confirmation_gate", "passed", "A separate explicit yes was received."),
                ToolStep("persist_order", "passed", f"Order {record.id[:8]} saved through the storage adapter."),
            ),
            confirmed_order_id=record.id,
            menu_attachments=attachments,
        )

    def _start_cancel(self, session: ConversationSession) -> AgentResponse:
        if not session.order:
            return AgentResponse(
                "There is no draft order to cancel.",
                (ToolStep("load_draft", "blocked", "Draft is empty."),),
            )
        session.pending_action = "confirm_cancel"
        return AgentResponse(
            "Clear this draft?\n\n" + format_order(session.order) + "\n\nReply `yes` or `no`.",
            (ToolStep("confirmation_gate", "info", "Draft cancellation requires explicit confirmation."),),
            menu_attachments=self._menu_attachments(session.order),
        )

    def _is_pizza_service_request(self, text: str, session: ConversationSession) -> bool:
        normalized = normalize_text(text)
        tokens = set(normalized.split())
        if tokens.intersection(OFF_TOPIC_WORDS) or re.search(r"\d\s*[+*/=]\s*\d", text):
            return False
        if tokens and tokens <= COURTESY_WORDS:
            return True
        if self._is_menu_browse_request(text) or self._is_order_history_request(text):
            return True
        if tokens.intersection(PIZZA_SERVICE_WORDS):
            return True
        if asks_about_item_details(text):
            return True
        return bool(find_menu_matches(text, self.catalog, limit=1))

    @staticmethod
    def _is_explicitly_off_topic(text: str) -> bool:
        normalized = normalize_text(text)
        return bool(
            set(normalized.split()).intersection(OFF_TOPIC_WORDS)
            or re.search(r"\d\s*[+*/=]\s*\d", text)
        )

    @staticmethod
    def _domain_guard_response() -> AgentResponse:
        return AgentResponse(
            "This chat is limited to pizza menu questions, orders, delivery or pickup, billing, and restaurant staff.",
            (
                ToolStep(
                    "pizza_domain_guard",
                    "blocked",
                    "An unrelated request was kept outside the ordering and catalog workflows.",
                ),
            ),
        )

    def _answer_enquiry(self, text: str, session: ConversationSession) -> AgentResponse:
        catalog = self.catalog
        if not self._is_pizza_service_request(text, session):
            return AgentResponse(
                "Tell me what you would like from the menu, what you want changed in your cart, or the order number you want me to check.",
                (ToolStep("clarify_ordering_request", "info", "The request was ambiguous but not explicitly outside pizza service."),),
            )
        browse_request = self._is_menu_browse_request(text)
        normalized = normalize_text(text)
        tokens = set(normalized.split())
        item_words = {
            word
            for item in catalog.active_items
            for value in (item.name, item.title, *item.aliases)
            for word in normalize_text(value).split()
            if word not in {"pizza", "small", "medium", "large"}
        }
        actionable = bool(
            browse_request
            or asks_about_item_details(text)
            or self._is_order_history_request(text)
            or tokens.intersection(MENU_WORDS | ORDER_WORDS | DELIVERY_WORDS | PICKUP_WORDS)
            or tokens.intersection(item_words)
        )
        if "pizza" in tokens and not actionable and tokens.intersection(NON_ORDERING_PIZZA_WORDS):
            return AgentResponse(
                "Which pizza, size, or order detail can I help you with?",
                (
                    ToolStep(
                        "clarify_ordering_request",
                        "info",
                        "Pizza was mentioned without an actionable menu or ordering request.",
                    ),
                ),
            )
        effective_query = "show menu" if browse_request else text
        matches = find_menu_matches(effective_query, catalog, limit=3)
        if len(matches) > 1:
            families = {
                " ".join(
                    word for word in item.name.split() if word.casefold() not in {"small", "medium", "large"}
                )
                for item in matches
            }
            if len(families) == 1:
                catalog_order = {item.sku: index for index, item in enumerate(catalog.active_items)}
                matches.sort(key=lambda item: catalog_order[item.sku])
        unknown_specific_pizza = bool(
            not browse_request
            and "pizza" in tokens
            and specific_menu_query_terms(text)
            and not matches
            and not asks_about_item_details(text)
        )
        if unknown_specific_pizza:
            session.menu_context = ()
            family_names = tuple(
                dict.fromkeys(
                    " ".join(
                        word
                        for word in (item.title or item.name).split()
                        if word.casefold() not in {"small", "medium", "large"}
                    )
                    for item in catalog.active_items
                    if item.category.casefold() == "pizza"
                )
            )
            approved = (
                "I could not find that pizza on the active menu, so I have not substituted another item. "
                "Available pizza families are "
                + ", ".join(family_names)
                + ". Which one would you like?"
            )
            reply, policy_trace = self.composer.compose(
                strictness=session.strictness,
                user_message=text,
                approved_reply=approved,
                immutable_terms=family_names,
                recent_context=self._recent_context(session),
            )
            return AgentResponse(
                reply,
                (
                    ToolStep("catalog_lookup", "blocked", "No active item had sufficient identity evidence."),
                    ToolStep(
                        "catalog_identity_guard",
                        "passed",
                        "An unmatched item name was not replaced by a weak fuzzy catalog result.",
                    ),
                    *policy_trace,
                ),
            )
        if not matches and asks_about_item_details(text):
            context_names = session.menu_context or tuple(reversed(session.order))
            matches = [catalog.by_name[name] for name in context_names if name in catalog.by_name]
        if not matches and asks_about_item_details(text):
            approved = (
                "Which listed pizza do you mean? Send its name, and I will use only the catalog details "
                "instead of guessing from an unmatched item."
            )
            reply, policy_trace = self.composer.compose(
                strictness=session.strictness,
                user_message=text,
                approved_reply=approved,
                recent_context=self._recent_context(session),
            )
            return AgentResponse(
                reply,
                (
                    ToolStep("catalog_lookup", "blocked", "No menu item was identified for the detail request."),
                    ToolStep(
                        "catalog_context_guard",
                        "passed",
                        "Item details were withheld because no confidently matched item or valid cart context exists.",
                    ),
                    *policy_trace,
                ),
            )
        catalog_reply = answer_menu_question(effective_query, catalog, matches)
        trace: list[ToolStep] = [ToolStep("catalog_lookup", "passed", "Checked the active catalog first.")]
        knowledge = (
            self.knowledge_responder(text)
            if self.knowledge_responder and catalog_reply.startswith("I can help")
            else None
        )
        approved = catalog_reply
        if knowledge is not None:
            approved, grounded_trace = knowledge
            trace.extend(grounded_trace)
        reply, policy_trace = self.composer.compose(
            strictness=session.strictness,
            user_message=text,
            approved_reply=approved,
            immutable_terms=tuple(item.name for item in matches if item.name in approved),
            recent_context=self._recent_context(session),
        )
        attachments = (
            self._featured_menu_attachments()
            if browse_request
            else self._menu_attachments(item.name for item in matches)
        )
        return AgentResponse(reply, tuple([*trace, *policy_trace]), menu_attachments=attachments)

    def _finalize(
        self,
        message: str,
        session: ConversationSession,
        response: AgentResponse,
        channel: str,
    ) -> AgentResponse:
        if response.menu_attachments:
            names = tuple(dict.fromkeys(item.title for item in response.menu_attachments))
            session.menu_context = names if len(names) <= 3 else ()
        if any(step.name == "compliance_guard" and step.status == "blocked" for step in response.tool_trace):
            session.compliance_failures += 1
        if response.conversation_signal is not None:
            self.storage.record_signal(session.session_id, response.conversation_signal)
        self.storage.record_exchange(session, message, response, channel)
        decision = response.handover_decision
        if decision is not None and decision.should_handover:
            case = self.handover_service.create_case(
                session=session,
                decision=decision,
                signal=response.conversation_signal,
                customer_request=message,
            )
            session.handover_active = True
            session.handover_case_id = case.id
            session.pending_action = "handover"
            self.storage.ensure_session(session)
            return AgentResponse(
                response.content,
                (*response.tool_trace, ToolStep("queue_handover", "passed", f"Handover {case.id[:8]} is pending.")),
                confirmed_order_id=response.confirmed_order_id,
                handover_requested=True,
                conversation_signal=response.conversation_signal,
                handover_decision=decision,
                handover_case_id=case.id,
                menu_attachments=response.menu_attachments,
            )
        return response

    def _apply_language_layer(
        self,
        user_message: str,
        session: ConversationSession,
        response: AgentResponse,
    ) -> AgentResponse:
        if response.handover_requested or any(
            step.name in {"agent_mode", "model_response"} for step in response.tool_trace
        ):
            return response
        immutable_terms = [attachment.title for attachment in response.menu_attachments]
        if response.confirmed_order_id:
            immutable_terms.append(response.confirmed_order_id[:8].upper())
        reply, policy_trace = self.composer.compose(
            strictness=session.strictness,
            user_message=user_message,
            approved_reply=response.content,
            immutable_terms=tuple(immutable_terms),
            recent_context=self._recent_context(session),
        )
        return AgentResponse(
            reply,
            (*response.tool_trace, *policy_trace),
            confirmed_order_id=response.confirmed_order_id,
            handover_requested=response.handover_requested,
            conversation_signal=response.conversation_signal,
            handover_decision=response.handover_decision,
            handover_case_id=response.handover_case_id,
            menu_attachments=response.menu_attachments,
        )

    def _menu_attachments(
        self,
        item_names: Iterable[str],
        *,
        limit: int = 3,
    ) -> tuple[MenuAttachment, ...]:
        catalog = self.catalog
        attachments: list[MenuAttachment] = []
        for name in item_names:
            item = catalog.by_name.get(name)
            if item is None:
                continue
            attachments.append(
                MenuAttachment(
                    sku=item.sku,
                    title=item.name,
                    description=item.description,
                    ingredients=item.ingredients,
                    image=item.image,
                    price=item.price,
                    currency=catalog.currency,
                )
            )
            if len(attachments) == limit:
                break
        return tuple(attachments)

    def _featured_menu_attachments(self) -> tuple[MenuAttachment, ...]:
        catalog = self.catalog
        names = [item.name for item in catalog.active_items if item.category.casefold() == "pizza"]
        return self._menu_attachments(names, limit=len(names))

    def _recent_context(self, session: ConversationSession) -> tuple[tuple[str, str], ...]:
        turns = self.storage.list_turns(session.session_id)[-6:]
        return tuple(
            ("user" if turn.role == "customer" else "assistant", turn.content)
            for turn in turns
        )

    def _conversation_signal(self, current_message: str, session: ConversationSession) -> ConversationSignal:
        turns = self.storage.list_turns(session.session_id)
        visible_turns = [
            *turns,
            {"id": "current-customer-turn", "role": "customer", "content": current_message},
        ]
        return self.context_analyzer.analyze(
            visible_turns,
            failed_attempts=session.failed_attempts,
            validation_failures=session.validation_failures,
            repair_requests=session.repair_requests,
        )

    @staticmethod
    def _format_final_bill(bill) -> str:
        rows = ["| Item | Qty | Total |", "| --- | ---: | ---: |"]
        rows.extend(
            f"| {line.item} | {line.quantity} | {bill.currency} {line.total:,} |" for line in bill.lines
        )
        rows.append(f"| **Grand total** |  | **{bill.currency} {bill.grand_total:,}** |")
        return "\n".join(rows)
