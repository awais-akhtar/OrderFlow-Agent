"""Grounded, provider-native response streaming for the customer chat."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from orderflow_agent.catalog import Catalog
from orderflow_agent.models import AgentResponse
from orderflow_agent.policy import PROHIBITED_COMMITMENTS, PromptPolicyCompiler
from orderflow_agent.runtime.response_pipeline import (
    LangChainResponsePipeline,
    ResponsePlan,
)
from orderflow_agent.tools import (
    asks_about_ingredient_amount,
    asks_about_ingredients,
    asks_about_item_measurements,
    listed_measurements,
    normalize_text,
)

MONEY_PATTERN = (
    r"\b[A-Z]{3}\s+(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?"
    r"(?![\d,]|\.\d)(?=$|\s|[.!?;:)\]}*])"
)
MEASUREMENT_PATTERN = (
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*[- ]?"
    r"(?:inch(?:es)?|centimet(?:er|re)s?|cm|kg|kilograms?|grams?)\b"
)


class StreamingReplyError(RuntimeError):
    """Raised when a provider cannot produce a safe customer-facing stream."""


@dataclass(frozen=True)
class StreamingReply:
    fragments: Iterator[str]
    provider_label: str


class GroundedStreamingResponder:
    """Turn deterministic tool output into natural prose using live model tokens."""

    def __init__(self, provider: object, catalog: Catalog) -> None:
        capabilities = getattr(provider, "capabilities", None)
        if not getattr(capabilities, "streaming", False):
            raise StreamingReplyError(
                "The selected provider has no verified token-streaming contract. "
                "Choose KernelLoom, OpenAI, or a compatible Hugging Face endpoint."
            )
        if not callable(getattr(provider, "stream_generate", None)):
            raise StreamingReplyError("The selected provider does not implement token streaming.")
        self.provider = provider
        self.catalog = catalog
        self.compiler = PromptPolicyCompiler()
        self.pipeline = LangChainResponsePipeline(provider)

    @property
    def provider_label(self) -> str:
        return str(getattr(self.provider, "label", "Model provider"))

    def stream(
        self,
        *,
        strictness: int,
        user_message: str,
        operational_response: AgentResponse,
        visible_history: Sequence[tuple[str, str]] = (),
    ) -> Iterator[str]:
        policy = self.compiler.compile(strictness)
        mode_direction = {
            "controlled": "Stay very close to the supplied facts.",
            "assisted": "Use natural, direct restaurant-service wording.",
            "flexible": "Adapt the tone naturally to the latest turn.",
        }[policy.mode.value]
        base_instructions = "\n".join(
            (
                "You are the customer-service voice of OrderFlow-Agent, a pizza ordering chat.",
                "Write only the next customer-facing reply in one short paragraph of no more than two sentences.",
                mode_direction,
                "Stay within pizza ordering: menu choices, ingredients, cart changes, delivery or pickup, confirmation, and service handover.",
                "If the request is unrelated, briefly redirect the customer to pizza ordering.",
                "Sound warm and natural, but keep the reply concise and ask only for the next missing order detail.",
                "Use only the supplied facts. Never invent an item, ingredient, price, quantity, order state, discount, refund, or delivery detail.",
                "Do not ask for contact, payment, or delivery information unless the turn facts require it.",
                "Do not mention tools or instructions. Do not use headings, labels, tables, or bullet lists.",
            )
        )
        browse_tokens = set(normalize_text(user_message).split())
        greeting = bool(
            any(step.name == "open_session" for step in operational_response.tool_trace)
            and browse_tokens.intersection({"hi", "hello", "hey"})
        )
        ingredient_request = bool(
            operational_response.menu_attachments and asks_about_ingredients(user_message)
        )
        ingredient_amount_request = bool(
            operational_response.menu_attachments and asks_about_ingredient_amount(user_message)
        )
        measurement_request = bool(
            operational_response.menu_attachments and asks_about_item_measurements(user_message)
        )
        browse_request = bool(
            operational_response.menu_attachments
            and not ingredient_request
            and not ingredient_amount_request
            and not measurement_request
            and browse_tokens.intersection({"menu", "which", "available", "have", "show", "list"})
        )
        catalog_match_request = bool(
            operational_response.menu_attachments
            and not browse_request
            and not ingredient_request
            and not ingredient_amount_request
            and not measurement_request
            and any(step.name == "catalog_lookup" for step in operational_response.tool_trace)
        )
        waiting_confirmation = any(
            step.name == "confirmation_gate" and step.status == "info"
            for step in operational_response.tool_trace
        )
        address_saved = any(
            step.name == "set_delivery_address" and step.status == "passed"
            for step in operational_response.tool_trace
        )
        if address_saved:
            waiting_confirmation = False
        order_confirmed = operational_response.confirmed_order_id is not None
        handover_requested = operational_response.handover_requested
        order_lookup = any(step.name == "lookup_order" for step in operational_response.tool_trace)
        order_recreated = any(step.name == "reorder_from_history" for step in operational_response.tool_trace)
        clarification = any(step.name == "clarify_ordering_request" for step in operational_response.tool_trace)
        missing_item_context = any(
            step.name == "catalog_context_guard" and step.status == "passed"
            for step in operational_response.tool_trace
        )
        unknown_catalog_item = any(
            step.name == "catalog_identity_guard" and step.status == "passed"
            for step in operational_response.tool_trace
        ) and not operational_response.menu_attachments
        ambiguous_confirmation = any(
            step.name == "confirmation_gate" and step.status == "blocked"
            for step in operational_response.tool_trace
        )
        invalid_address = any(
            step.name == "validate_delivery_address" and step.status == "blocked"
            for step in operational_response.tool_trace
        )
        security_redirect = any(
            step.name in {"prompt_injection_guard", "input_length_guard"} and step.status == "blocked"
            for step in operational_response.tool_trace
        )
        domain_redirect = security_redirect or any(
            step.name == "pizza_domain_guard" and step.status == "blocked"
            for step in operational_response.tool_trace
        )
        address_security_redirect = security_redirect and "delivery address" in operational_response.content.casefold()
        draft_updated = any(
            step.name == "update_draft" and step.status == "passed"
            for step in operational_response.tool_trace
        )
        fulfilment_saved = any(
            step.name == "set_fulfilment" and step.status == "passed"
            for step in operational_response.tool_trace
        ) and "say `confirm order`" in operational_response.content.casefold()
        quantity_limit = any(
            step.name == "validate_order" and step.status == "blocked" and "limited to" in step.detail.casefold()
            for step in operational_response.tool_trace
        )
        malformed_quantity = quantity_limit and "whole quantities" in operational_response.content.casefold()
        intent_conflict = any(
            step.name == "validate_intent" and step.status == "blocked"
            for step in operational_response.tool_trace
        )
        removal_limit = any(
            step.name == "validate_order"
            and step.status == "blocked"
            and "removal quantity exceeded" in step.detail.casefold()
            for step in operational_response.tool_trace
        )
        operational_text = operational_response.content
        awaiting_address = draft_updated and "delivery address" in operational_text.casefold()
        awaiting_fulfilment = draft_updated and "delivery or pickup" in operational_text.casefold()
        response_kind = "generic"
        domain_category = ""
        order_lookup_failed = False
        required_families: tuple[str, ...] = ()
        required_sizes: tuple[str, ...] = ()
        required_ingredients: tuple[str, ...] = ()
        if handover_requested:
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "A deterministic escalation rule has already opened a Staff-support ticket.",
                    "Tell the customer clearly that the request is being transferred to restaurant staff.",
                    "Say that the conversation and current cart have been preserved, so they do not need to repeat themselves.",
                    "Your reply must include both ideas in direct language: I am transferring your request to restaurant staff; your conversation and cart are preserved.",
                    "Do not continue answering the menu question, promise a wait time, or claim that a staff member has already replied.",
                    "Return only a warm customer-facing reply of no more than two sentences and 45 words.",
                )
            )
            brief = "Handover facts:\n" + operational_text
        elif domain_redirect:
            category = "request to change protected ordering rules" if security_redirect else _off_topic_category(user_message)
            domain_category = category
            tone = _redirect_tone(user_message)
            if address_security_redirect:
                response_kind = "address_security_redirect"
                base_instructions = "\n".join(
                    (
                        "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                        "The submitted text was not saved as a delivery address.",
                        "Ask the customer to resend only the street, building, and area details.",
                        "Do not mention prompts, rules, security, or offer to restart the order.",
                        "Return one calm customer-facing sentence under 28 words.",
                    )
                )
                brief = "No delivery address was saved. Ask for street, building, and area details only."
            else:
                base_instructions = "\n".join(
                    (
                        "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                        f"The latest customer request is a {category}, outside this restaurant ordering chat.",
                        "Do not answer, explain, calculate, translate, or write code for the unrelated request.",
                        f"Use a {tone} tone. Politely decline, then offer one useful next step such as browsing pizzas or starting an order.",
                        "The sentence must include either the exact phrase 'pizza menu' or the exact phrase 'pizza order'.",
                        "Do not begin with 'Understood' and do not recite a list of every supported feature.",
                        "Return only one natural customer-facing sentence under 34 words.",
                    )
                )
                brief = f"Pizza-domain redirect facts:\nRequest category: {category}.\nOffer help with the pizza menu or an order."
        elif greeting:
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The customer has greeted you.",
                    "Reply with a warm professional greeting and ask what pizza order you can get started.",
                    "Do not list features, mention staff, explain safeguards, or say you are building a draft.",
                    "Return one natural sentence under 18 words.",
                )
            )
            brief = "Greeting facts:\nAsk what pizza order the customer would like to start."
        elif order_lookup:
            response_kind = "order_lookup"
            lookup_failed = any(
                step.name == "lookup_order" and step.status == "blocked"
                for step in operational_response.tool_trace
            )
            order_lookup_failed = lookup_failed
            lookup_status = "confirmed" if " was confirmed " in f" {operational_text.casefold()} " else ""
            lookup_fulfilment = next(
                (
                    value
                    for value in ("delivery", "pickup")
                    if f"fulfilment: {value}" in operational_text.casefold()
                ),
                "",
            )
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "This turn concerns a persisted order lookup, not the current cart bill.",
                    "Use every supplied order fact exactly and do not invent delivery progress, a refund, a replacement, or a different order reference.",
                    (
                        "The customer has not supplied an order number. Ask clearly for the eight-character order number and explain briefly that a date alone cannot identify an order safely; do not thank them for already providing it."
                        if lookup_failed
                        else (
                            "Summarise the matched order naturally, including its exact reference, every item, total, "
                            f"recorded status ({lookup_status}), and fulfilment method ({lookup_fulfilment}). "
                            "Delivery or pickup is a method, not progress or completion status."
                        )
                    ),
                    (
                        "Also explain that the available items were copied into a new draft and state the supplied next step."
                        if order_recreated
                        else "Do not claim that a new order was created."
                    ),
                    "Return no more than two concise customer-facing sentences.",
                )
            )
            if lookup_failed:
                brief = (
                    "No order number was supplied; ask for the eight-character order number and explain that a "
                    "date alone cannot identify an order safely."
                )
            else:
                brief = re.sub(
                    r"\.\s+(Items|Total|Fulfilment|Delivery address):",
                    r"; \1:",
                    re.sub(r"\s+", " ", operational_text).strip(),
                    flags=re.I,
                )
                brief = _sanitize_legacy_address_facts(brief)
        elif unknown_catalog_item:
            response_kind = "unknown_catalog_item"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The requested pizza name was not found and no menu item was selected, changed, or substituted.",
                    "Say briefly that you could not find that pizza, name only the available families supplied in the facts, and ask which one they want.",
                    "Do not imply that the customer selected or changed an option, and do not invent ingredients, prices, sizes, or measurements.",
                    "Return no more than two concise customer-facing sentences.",
                )
            )
            brief = "Unknown-catalog-item facts:\n" + operational_text
        elif missing_item_context:
            response_kind = "missing_item_context"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The customer asked for item details, but no listed pizza was confidently identified.",
                    "Ask which listed pizza they mean and make clear that you will check the catalog rather than guess.",
                    "Do not select, recommend, or name a specific pizza, ingredient, price, size, or measurement.",
                    "Return exactly one question ending in a question mark, under 28 words, with no full stop.",
                )
            )
            brief = "Missing-item-context facts:\nNo listed pizza was identified; ask for its exact menu name."
        elif clarification:
            response_kind = "clarification"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The customer's request is ambiguous, but it is not known to be outside pizza service.",
                    "Ask one friendly clarifying question about the menu, their cart, or an existing order number.",
                    "Do not scold them, say 'Understood', list system capabilities, or mention staff unless they asked for staff.",
                    "Return one natural sentence under 28 words.",
                )
            )
            brief = "Ordering-clarification facts:\n" + operational_text
        elif browse_request:
            response_kind = "browse"
            families: list[str] = []
            for item in operational_response.menu_attachments:
                family = " ".join(
                    word for word in item.title.split() if word.casefold() not in {"small", "medium", "large"}
                )
                if family not in families:
                    families.append(family)
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The customer asked which pizzas are available.",
                    f"The available families are exactly: {', '.join(families)}.",
                    "Write one flowing sentence that names every family, then asks which flavor and size they prefer.",
                    "Repeat every family name in this reply even if the earlier conversation already listed them.",
                    "Product cards beside the reply show the exact sizes, prices, images, and ingredients.",
                    "Do not add a price, size, menu item, heading, label, bullet, or numbered list.",
                    "Return only one natural customer-facing sentence under 55 words.",
                )
            )
            brief = "\n".join(
                (
                    "The customer wants to browse the menu.",
                    f"Available pizza families are exactly: {', '.join(families)}.",
                    "Say which pizza families are available and ask which flavor and size they prefer.",
                    "Product cards beside the reply show every size, exact price, image, description, and ingredient list.",
                    "Write one natural sentence under 55 words. Do not repeat sizes or prices.",
                )
            )
        elif catalog_match_request:
            families: list[str] = []
            sizes: list[str] = []
            for item in operational_response.menu_attachments:
                words = item.title.split()
                size = next((word for word in words if word.casefold() in {"small", "medium", "large"}), "")
                family = " ".join(word for word in words if word.casefold() not in {"small", "medium", "large"})
                if family and family not in families:
                    families.append(family)
                if size and size not in sizes:
                    sizes.append(size)
            sizes.sort(key=lambda value: {"small": 0, "medium": 1, "large": 2}.get(value.casefold(), 99))
            size_phrase = (
                sizes[0]
                if len(sizes) == 1
                else ", ".join(sizes[:-1]) + ", and " + sizes[-1]
            )
            required_opening = f"{', '.join(families)} comes in {size_phrase}."
            response_kind = "catalog_match"
            required_families = tuple(families)
            required_sizes = tuple(sizes)
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    f"The only matched family and its available sizes are expressed in this sentence: {required_opening}",
                    f"Begin with this exact plain sentence: {required_opening}",
                    "Then write one short plain sentence saying the cards show exact prices and ingredients and asking which size they prefer.",
                    "Do not state a price, measurement, or ingredient in this reply.",
                    "Never write 'Pizza Family' or 'Sizes Available'. Do not use markdown, headings, labels, bullets, or line breaks.",
                    "Return only those two natural customer-facing sentences under 45 words total.",
                )
            )
            brief = "\n".join(
                (
                    required_opening,
                    "The product cards contain the exact prices, descriptions, images, and ingredients.",
                )
            )
        elif ingredient_request or ingredient_amount_request or measurement_request:
            item_names = list(dict.fromkeys(item.title for item in operational_response.menu_attachments))
            family_names = list(
                dict.fromkeys(
                    " ".join(
                        word
                        for word in item.title.split()
                        if word.casefold() not in {"small", "medium", "large"}
                    )
                    for item in operational_response.menu_attachments
                )
            )
            if len(family_names) == 1 and (ingredient_request or ingredient_amount_request):
                item_names = family_names
            ingredient_names = list(
                dict.fromkeys(
                    ingredient
                    for item in operational_response.menu_attachments
                    for ingredient in item.ingredients
                )
            )
            detail_instructions = [
                "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                f"The customer asked for catalog details about: {', '.join(item_names)}.",
            ]
            detail_brief = [f"The relevant menu item is {', '.join(item_names)}."]
            if ingredient_request:
                required_ingredients = tuple(ingredient_names)
                detail_instructions.extend(
                    (
                        f"The complete catalog ingredient list is exactly: {', '.join(ingredient_names)}.",
                        "Name every listed ingredient without explaining, expanding, substituting, or inferring it.",
                    )
                )
                detail_brief.append(
                    f"Its complete ingredient list is {', '.join(ingredient_names)}."
                )
            if ingredient_amount_request:
                response_kind = "ingredient_amount"
                required_ingredients = tuple(ingredient_names)
                detail_instructions.extend(
                    (
                        "Name every listed ingredient and say plainly that the catalog does not list per-ingredient amounts.",
                        "Do not say quantities vary by size, estimate a quantity, or include an ingredient outside this list.",
                    )
                )
                if not ingredient_request:
                    detail_instructions.insert(
                        2,
                        f"The complete catalog ingredient list is exactly: {', '.join(ingredient_names)}.",
                    )
                    detail_brief.append(
                        f"Its complete ingredient list is {', '.join(ingredient_names)}."
                    )
                detail_brief.append("The catalog does not list per-ingredient amounts.")
            if measurement_request:
                measurement_values = list(
                    dict.fromkeys(
                        value
                        for item in operational_response.menu_attachments
                        for value in listed_measurements(item.description)
                    )
                )
                if measurement_values:
                    detail_instructions.extend(
                        (
                            f"The exact listed measurement is: {', '.join(measurement_values)}.",
                            "Use only that measurement and do not estimate or infer another one.",
                        )
                    )
                    detail_brief.append(
                        "Exact listed measurement: " + " | ".join(measurement_values)
                    )
                else:
                    detail_instructions.extend(
                        (
                            "The catalog does not list a diameter in inches or any ingredient or portion weight.",
                            "Say plainly that those measurements are not listed; never estimate or infer them.",
                        )
                    )
                    detail_brief.append(
                        "Measurement availability: diameter in inches and ingredient or portion weights are not listed."
                    )
            detail_instructions.append(
                "Do not use markdown, headings, labels, bullets, or line breaks. Return only one natural customer-facing sentence under 55 words."
            )
            base_instructions = "\n".join(detail_instructions)
            brief = "\n".join(detail_brief)
        elif intent_conflict:
            response_kind = "intent_conflict"
            conflict_actions = (
                "delivery and pickup choices"
                if "fulfilment method" in operational_text.casefold()
                else "removal and addition"
            )
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The customer's message combined conflicting consequential actions, so no draft field changed.",
                    "Ask them to send exactly one cart or fulfilment action at a time.",
                    f"Use this exact meaning: No draft changes were made; please send the {conflict_actions} in separate messages, one action at a time.",
                    "Do not say you will handle, apply, proceed with, queue, or remember either action later.",
                    "Do not ask for confirmation because there is no newly applied change to confirm.",
                    "Return one direct customer-facing sentence under 30 words and do not append another clause.",
                )
            )
            brief = (
                "No draft field changed. Conflicting actions: "
                + conflict_actions
                + ". Required next step: send one action at a time."
            )
        elif invalid_address:
            response_kind = "invalid_address"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "No delivery address was saved from this turn.",
                    "Ask for a street, building number, and area, or say the customer may choose pickup instead.",
                    "Do not claim pickup is selected, mention a current location, or repeat rejected text.",
                    "Return one concise customer-facing sentence under 30 words.",
                )
            )
            brief = "No delivery address was saved. Ask for street, building number, and area, or offer pickup instead."
        elif ambiguous_confirmation:
            response_kind = "ambiguous_confirmation"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The customer's last confirmation answer was ambiguous, so no action was taken.",
                    "Ask them to reply only yes to place the exact draft or no to keep editing.",
                    "Your sentence must contain both literal words 'yes' and 'no'.",
                    "Use this target structure: Please reply only yes to place this exact draft, or no to keep editing.",
                    "Do not interpret their choice, tell them to go ahead, or ask what to add to the cart.",
                    "Return one concise customer-facing sentence under 28 words.",
                )
            )
            brief = "No action was taken. Required choice: reply only yes to place the exact draft, or no to keep editing."
        elif removal_limit:
            response_kind = "removal_limit"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The requested removal exceeded the quantity currently in the draft, so the draft did not change.",
                    "State that fact naturally using the supplied item and quantity, then ask for a valid smaller removal amount.",
                    "Use this one-sentence structure: The draft did not change because the requested removal exceeds its current quantity; please choose a smaller removal quantity.",
                    "Do not suggest adding items, claim a cart update, or ask to confirm the order.",
                    "Return exactly one direct customer-facing sentence under 32 words.",
                )
            )
            brief = operational_text
        elif quantity_limit:
            response_kind = "quantity_limit"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The requested quantity was rejected before the cart changed.",
                    (
                        "Explain that the quantity must be a whole number from 1 to 20; do not say the customer exceeded the limit."
                        if malformed_quantity
                        else "State the exact supplied order limit naturally and ask the customer for a smaller quantity."
                    ),
                    "Do not thank them for an order or claim that an item was added, ordered, placed, or confirmed.",
                    "Return one concise customer-facing sentence under 28 words.",
                )
            )
            brief = operational_text
        elif awaiting_address:
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The supplied pizza has already been added to the draft for delivery.",
                    "Write one sentence that acknowledges the exact draft item and asks for the delivery address in the same sentence.",
                    "Do not claim that the order is placed or ask to add the pizza again.",
                    "Return only that warm customer-facing sentence under 35 words.",
                )
            )
            brief = "Draft-awaiting-address facts:\n" + operational_text
        elif awaiting_fulfilment:
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The supplied pizza has already been added to the draft.",
                    "Start with the exact item already being in the draft, then ask the exact short question: Delivery or pickup?",
                    "Use no greeting or thank-you before the item, and keep the whole reply under 22 words.",
                    "Do not claim that the order is placed or ask to add the pizza again.",
                    "Return only that warm customer-facing sentence under 35 words.",
                )
            )
            brief = "Draft-awaiting-fulfilment facts:\n" + operational_text
        elif fulfilment_saved:
            response_kind = "fulfilment_saved"
            fulfilment_total = _trusted_order_total(operational_text) or "the supplied total"
            selected_fulfilment = "delivery" if "delivery" in operational_text.casefold() else "pickup"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    f"The customer selected {selected_fulfilment}; the pizza remains an unconfirmed draft.",
                    f"State the exact draft total {fulfilment_total} and ask the customer to say confirm order when the summary is correct.",
                    "The words 'confirm order' name the command that opens a later approval step; they do not mean the customer has confirmed anything.",
                    f"Use this sentence structure: {selected_fulfilment.title()} is selected and your draft total is {fulfilment_total}; type 'confirm order' to review it before final approval.",
                    "Never say or imply that the order is placed, submitted, completed, or confirmed.",
                    "Do not repeat fact labels or identify the customer, assistant, or agent.",
                    "Return one warm customer-facing sentence under 35 words.",
                )
            )
            brief = (
                f"{selected_fulfilment.title()} is selected. The unconfirmed draft total is "
                f"{fulfilment_total}. The customer must type 'confirm order' to open final approval."
            )
        elif waiting_confirmation:
            confirmation_total = _trusted_order_total(operational_text) or "the supplied total"
            confirmation_fulfilment = "delivery" if "delivery" in operational_text.casefold() else "pickup"
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The draft has passed validation, but the order is not placed or confirmed yet.",
                    "In one sentence, use the exact total and fulfilment facts, say the order still needs approval, and ask the customer to reply yes to place it or no to keep editing.",
                    "Never say or imply that the order is already confirmed, submitted, or placed.",
                    "Return only that warm customer-facing sentence under 45 words.",
                )
            )
            brief = "\n".join(
                (
                    "Confirmation-gate facts:",
                    f"Exact total: {confirmation_total}.",
                    f"Fulfilment: {confirmation_fulfilment}.",
                    "Status: unconfirmed draft.",
                    "Required choice: reply yes to place it or no to keep editing.",
                )
            )
        elif address_saved:
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The pizza is already in the draft and the delivery address has now been saved.",
                    "Repeat the exact saved address, give the exact total supplied below, and ask the customer to say confirm order when ready.",
                    "Keep the currency amount exactly as supplied and do not add decimal places.",
                    "Do not ask to add the pizza to the cart again and do not claim that the order is placed.",
                    "Return only a warm customer-facing reply of no more than two sentences and 45 words.",
                )
            )
            brief = "Address-and-bill facts:\n" + operational_text
        elif order_confirmed:
            response_kind = "confirmed_order"
            confirmed_reference = next(
                iter(re.findall(r"\b[A-F0-9]{8}\b", operational_text, re.I)),
                "",
            )
            confirmed_total = _trusted_order_total(operational_text)
            confirmed_fulfilment = (
                "delivery" if "fulfilment: delivery" in operational_text.casefold() else "pickup"
            )
            confirmed_address_match = re.search(r"Delivery address:\s*([^\n]+)", operational_text, re.I)
            confirmed_address = (
                _sanitize_address_value(confirmed_address_match.group(1).strip().rstrip("."))
                if confirmed_address_match
                else ""
            )
            base_instructions = "\n".join(
                (
                    "Write the next reply for OrderFlow-Agent, a pizza ordering chat.",
                    "The deterministic confirmation tool has placed this order.",
                    f"Write one complete sentence that says order {confirmed_reference} is confirmed for {confirmed_fulfilment} with a total of {confirmed_total}.",
                    "Keep the reference, fulfilment, and total exactly as supplied. Do not put a full stop after only the words 'Order confirmed'.",
                    "Do not add decimal places to the supplied total.",
                    "Return only that warm customer-facing sentence under 35 words.",
                )
            )
            brief = (
                f"Order {confirmed_reference} is confirmed for {confirmed_fulfilment} with an exact total of "
                f"{confirmed_total}"
                + (f" and delivery address {confirmed_address}" if confirmed_address else "")
                + "."
            )
        else:
            attachments = "\n".join(
                f"- {item.title}; {item.currency} {item.price:,}; ingredients: "
                f"{', '.join(item.ingredients) or 'not listed'}; description: {item.description or 'not listed'}"
                for item in operational_response.menu_attachments
            ) or "- No product cards for this turn."
            brief = (
                "Facts for this turn:\n"
                f"{operational_text}\n"
                f"Relevant product facts:\n{attachments}\n"
                "Answer the latest request directly in no more than two sentences and normally fewer than 45 words."
            )
        model_user_message = (
            "Write the next customer-facing reply using only the supplied operational facts and required next step."
        )
        if handover_requested:
            model_user_message = (
                "A staff-support ticket is now pending. Tell me directly that you are transferring my request "
                "to restaurant staff and that my conversation and cart are preserved."
            )
        elif domain_redirect:
            model_user_message = (
                "Tell me the address was not saved and ask only for street, building, and area details."
                if address_security_redirect
                else "Decline that request naturally and offer to help me choose a pizza or start an order."
            )
        elif greeting:
            model_user_message = "Greet me naturally and ask what pizza order you can start."
        elif order_lookup:
            model_user_message = "Help with the persisted order lookup using only the supplied facts."
        elif clarification:
            model_user_message = "Ask one natural question to clarify what pizza-service help I need."
        elif browse_request:
            model_user_message = "Tell me naturally which pizza families are available, using only the supplied menu facts."
        elif catalog_match_request:
            model_user_message = (
                "Use the required opening exactly, do not describe the sizes, then ask which size I prefer."
            )
        elif ingredient_request or ingredient_amount_request or measurement_request:
            model_user_message = (
                "State every supplied ingredient and explicitly say the catalog does not list per-ingredient amounts."
                if ingredient_amount_request
                else "Answer the item-detail question using only the supplied catalog facts and identify anything not listed."
            )
        elif intent_conflict:
            model_user_message = (
                "Write one sentence only: no draft changes were made; ask me to send the "
                f"{conflict_actions} in separate messages, one action at a time. Do not add another clause."
            )
        elif invalid_address:
            model_user_message = (
                "Ask for street, building number, and area details, with pickup as an alternative; do not claim a location."
            )
        elif ambiguous_confirmation:
            model_user_message = (
                "Produce the required neutral clarification sentence now. Include both literal choices, yes and no, "
                "without selecting either one."
            )
        elif removal_limit:
            model_user_message = (
                "Write one sentence saying the draft did not change because the removal exceeds its current quantity, "
                "then ask for a smaller removal quantity."
            )
        elif quantity_limit:
            model_user_message = (
                "Ask me for a whole-number quantity from 1 to 20; do not say I exceeded the limit."
                if malformed_quantity
                else "Explain the exact quantity limit and ask me to choose a smaller amount."
            )
        elif awaiting_address:
            model_user_message = "In one complete sentence, acknowledge the draft item and ask me for the delivery address."
        elif awaiting_fulfilment:
            model_user_message = "In one complete sentence, acknowledge the draft item and ask me to choose delivery or pickup."
        elif fulfilment_saved:
            model_user_message = (
                "Tell me the selected fulfilment and exact draft total, then ask me to type 'confirm order' "
                "to open the final approval step."
            )
        elif waiting_confirmation:
            model_user_message = (
                "The customer requested the checkout review, not final approval. "
                "In one sentence, state the exact total and ask for a separate yes to place it or no to keep editing."
            )
        elif address_saved:
            model_user_message = (
                "The delivery address is already saved. Give the exact total and ask the customer to say confirm order; "
                "do not ask them to confirm the address or add the pizza again."
            )
        elif order_confirmed:
            model_user_message = "Include the exact order reference, total, fulfilment, and address from the supplied facts."
        elif draft_updated:
            model_user_message = "Acknowledge the cart update naturally and give only the next step supplied in the facts."
        # Customer text is evidence for deterministic intent handling, never a second instruction channel to the model.
        isolate_history = True
        normalized_history = tuple(
            (
                "assistant" if str(role).casefold() in {"assistant", "ai"} else "user",
                str(content),
            )
            for role, content in visible_history[-8:]
        )

        def transaction_reviewer(candidate: str) -> str | None:
            return self._guard(candidate, operational_response, final=True)

        def context_reviewer(candidate: str) -> str | None:
            return self._context_guard(
                candidate,
                brief,
                user_message,
                response_kind=response_kind,
                required_families=required_families,
                required_sizes=required_sizes,
                required_ingredients=required_ingredients,
            )

        correction = ""
        max_attempts = 3
        for attempt in range(max_attempts):
            attempt_request = model_user_message
            if correction:
                attempt_request += (
                    "\nThe previous answer was rejected. Follow the correction in the system instructions "
                    "and finish the complete reply."
                )
            plan = ResponsePlan(
                instructions=base_instructions,
                facts=brief,
                user_request=attempt_request,
                history=normalized_history,
                isolate_history=isolate_history,
                retry_feedback=correction,
            )
            native_stream = self.pipeline.stream(plan)
            complete = ""
            buffered_fragments: list[str] = []
            try:
                for fragment in native_stream:
                    text = str(fragment)
                    if not text:
                        continue
                    complete += text
                    buffered_fragments.append(text)
                    if len(complete) > 2000:
                        raise StreamingReplyError(
                            "The model response exceeded the customer-reply length limit."
                        )
            except Exception as exc:
                raise StreamingReplyError(
                    f"{self.provider_label} stopped before it completed the reply: "
                    f"{' '.join(str(exc).split())[:240] or type(exc).__name__}"
                ) from exc
            finally:
                close = getattr(native_stream, "close", None)
                if callable(close):
                    close()
            if not complete.strip():
                raise StreamingReplyError(f"{self.provider_label} returned no customer-facing text.")
            cleaned = _clean_presentation(complete)
            if cleaned != complete:
                complete = cleaned
                buffered_fragments = list(_display_fragments(cleaned))
            cleaned = _normalize_allowed_money_format(complete, operational_response)
            if cleaned != complete:
                complete = cleaned
                buffered_fragments = list(_display_fragments(cleaned))
            if fulfilment_saved:
                cleaned = _compact_service_sentences(complete)
                if cleaned != complete:
                    complete = cleaned
                    buffered_fragments = list(_display_fragments(cleaned))
            if order_lookup and not order_lookup_failed:
                cleaned = _complete_order_lookup_facts(
                    complete,
                    operational_text,
                    operational_response.menu_attachments,
                )
                if cleaned != complete:
                    complete = cleaned
                    buffered_fragments = list(_display_fragments(cleaned))
            if domain_redirect and not address_security_redirect:
                cleaned = _ground_domain_redirect(complete, domain_category)
                if cleaned != complete:
                    complete = cleaned
                    buffered_fragments = list(_display_fragments(cleaned))
            review = self.pipeline.review(
                complete,
                (transaction_reviewer, context_reviewer),
            )
            violation = review.violation
            if violation:
                if attempt < max_attempts - 1:
                    if response_kind == "missing_item_context":
                        correction = (
                            "\nThe previous draft was rejected before display because: "
                            + violation
                            + " Write exactly one question ending with '?', with no other sentence or full stop; "
                            "ask for the listed pizza name and include no item or ingredient."
                        )
                    else:
                        correction = (
                            "\nThe previous draft was rejected before display because: "
                            + violation
                            + " Write a new direct reply using only the brief."
                        )
                    continue
                raise StreamingReplyError(violation)
            yield from buffered_fragments
            return

    def _guard(
        self,
        candidate: str,
        response: AgentResponse,
        *,
        final: bool,
    ) -> str | None:
        lowered = candidate.casefold()
        commitment = next((phrase for phrase in PROHIBITED_COMMITMENTS if phrase in lowered), None)
        if commitment:
            return f"The model stream attempted a prohibited commitment: {commitment}."
        allowed_money = {
            _normal_money(value)
            for value in re.findall(MONEY_PATTERN, response.content)
        }
        allowed_money.update(
            _normal_money(f"{item.currency} {item.price:,}")
            for item in response.menu_attachments
        )
        money_matches = list(re.finditer(MONEY_PATTERN, candidate))
        if not final and money_matches and money_matches[-1].end() == len(candidate):
            money_matches.pop()
        used_money = {_normal_money(match.group(0)) for match in money_matches}
        unsupported_money = sorted(used_money - allowed_money)
        if unsupported_money:
            return "The model stream introduced a price outside the deterministic brief: " + ", ".join(
                unsupported_money
            )
        symbol_money = re.findall(r"[$\u00a3\u20ac]\s*[\d,]+", candidate)
        if symbol_money:
            return "The model stream changed the catalog currency: " + ", ".join(symbol_money[:3])
        confirms_order = bool(
            re.search(
                r"\border\s+(?:is|was|has\s+been)\s+(?:successfully\s+)?"
                r"(?:completed|confirmed|placed|submitted)\b",
                lowered,
            )
            or "placed your order" in lowered
            or "placing the order now" in lowered
            or "thank you for confirming your order" in lowered
            or "thank you for your order" in lowered
            or re.search(
                r"\b(?:order|pizza)\b.{0,35}\b(?:accepted|booked|finali[sz]ed|processed|"
                r"sent\s+to\s+the\s+kitchen|being\s+prepared|on\s+its\s+way)\b",
                lowered,
            )
            or re.search(
                r"\b(?:accepted|booked|finali[sz]ed|processed)\b.{0,25}\b(?:order|pizza)\b",
                lowered,
            )
        )
        if confirms_order and response.confirmed_order_id is None:
            return "The model stream claimed an order was placed before the confirmation tool succeeded."
        draft_changed = any(
            step.name == "update_draft" and step.status == "passed" for step in response.tool_trace
        )
        cart_claim_text = re.sub(
            r"\b(?:did\s+not|didn't|has\s+not|hasn't|was\s+not|wasn't|were\s+not|weren't)\s+"
            r"(?:been\s+)?(?:added|changed|deleted|removed|updated)\b"
            r"|\b(?:remains?|stays?)\s+unchanged\b",
            "",
            lowered,
        )
        claims_cart_change = bool(
            re.search(
                r"\b(?:added|updated|placed|put)\b.{0,45}\b(?:cart|draft|order)\b"
                r"|\b(?:cart|draft)\b.{0,35}\bupdated\b",
                cart_claim_text,
            )
        )
        if not claims_cart_change and re.search(
            r"\b(?:added|changed|deleted|removed|updated)\b.{0,55}\b(?:item|pizza)\b"
            r"|\b(?:item|pizza)\b.{0,55}\b(?:added|changed|deleted|removed|updated)\b",
            cart_claim_text,
        ):
            claims_cart_change = True
        if claims_cart_change and not draft_changed:
            return "The model stream claimed a cart change that no deterministic cart tool performed."
        reference_mentions = re.findall(r"\b[A-F0-9]{8}\b", candidate, re.I)
        if reference_mentions:
            allowed_references = {
                value.casefold()
                for value in re.findall(r"\b[A-F0-9]{8}\b", response.content, re.I)
            }
            if response.confirmed_order_id is not None:
                allowed_references.add(response.confirmed_order_id[:8].casefold())
            if any(value.casefold() not in allowed_references for value in reference_mentions):
                return "The model stream changed or invented an order reference."
        unknown_item = self._unknown_titled_pizza(candidate)
        if unknown_item:
            return f"The model stream introduced an item outside the active catalog: {unknown_item}."
        return None

    def _unknown_titled_pizza(self, candidate: str) -> str:
        allowed: set[str] = {"pizza"}
        for item in self.catalog.active_items:
            for value in (item.name, item.title, *item.aliases):
                normalized = normalize_text(value)
                allowed.add(normalized)
                allowed.add(" ".join(word for word in normalized.split() if word not in {"small", "medium", "large"}))
        for match in re.findall(r"\b(?:[A-Z][A-Za-z0-9.-]*\s+){0,4}Pizza\b", candidate):
            normalized = normalize_text(match)
            words = normalized.split()
            while words and words[0] in {"a", "an", "our", "the", "your"}:
                words.pop(0)
            normalized = " ".join(words)
            if normalized not in allowed and not any(normalized in value for value in allowed):
                return match
        return ""

    def _context_guard(
        self,
        candidate: str,
        brief: str,
        user_message: str,
        *,
        response_kind: str = "generic",
        required_families: Sequence[str] = (),
        required_sizes: Sequence[str] = (),
        required_ingredients: Sequence[str] = (),
    ) -> str:
        lowered = candidate.casefold()
        brief_lower = brief.casefold()
        if len(re.findall(r"[.!?]+[\"']?(?=\s|$)", candidate)) > 2:
            return "The model reply exceeded the two-sentence customer-service limit."
        label = next(
            (
                value
                for value in (
                    "customer facing response",
                    "assistant response",
                    "final answer",
                    "confirmation-gate facts",
                    "facts for this turn",
                    "pizza-domain redirect facts",
                    "provided facts",
                    "supplied facts",
                    "operational facts",
                    "awaiting-address",
                    "awaiting-fulfilment",
                    "draft-awaiting",
                )
                if value in lowered
            ),
            "",
        )
        if label:
            return f"The model added an internal response label: {label}."
        identity = next(
            (value for value in ("ai language model", "as an ai", "i am an ai") if value in lowered),
            "",
        )
        if identity:
            return f"The model discussed its internal identity instead of serving the customer: {identity}."
        unsupported = next(
            (
                value
                for value in ("address", "phone number", "payment details", "delivery address")
                if value in lowered and value not in brief_lower
            ),
            "",
        )
        if unsupported:
            return f"The model asked for unsupported customer information: {unsupported}."
        allowed_measurements = {
            _normal_measurement(value)
            for value in re.findall(MEASUREMENT_PATTERN, brief, re.IGNORECASE)
        }
        unsupported_measurements = [
            value
            for value in re.findall(MEASUREMENT_PATTERN, candidate, re.IGNORECASE)
            if _normal_measurement(value) not in allowed_measurements
        ]
        if unsupported_measurements:
            return (
                "The model added a measurement that is not present in the catalog facts: "
                + ", ".join(unsupported_measurements[:3])
                + "."
            )
        stage_violation = GroundedStreamingResponder._stage_fact_guard(
            candidate,
            brief,
            response_kind=response_kind,
            required_families=required_families,
            required_sizes=required_sizes,
            required_ingredients=required_ingredients,
            required_order_items=tuple(
                normalize_text(item.name)
                for item in self.catalog.active_items
                if normalize_text(item.name) in normalize_text(brief)
            ),
        )
        if stage_violation:
            return stage_violation
        user_tokens = set(normalize_text(user_message).split())
        if response_kind in {"browse", "catalog_match"} and user_tokens.intersection(
            {"menu", "pizza", "available", "have"}
        ):
            denial = next(
                (
                    value
                    for value in (
                        "do not have any specific product information",
                        "don't have any specific product information",
                        "no product information",
                        "cannot provide the menu",
                        "can't provide the menu",
                    )
                    if value in lowered
                ),
                "",
            )
            if denial:
                return f"The model falsely denied the supplied catalog facts: {denial}."
            if not set(normalize_text(candidate).split()).intersection(
                {"menu", "pizza", "options", "cheese", "tandoori", "pepperoni", "garden"}
            ):
                return "The model did not answer the menu question."
        if response_kind == "clarification":
            if "?" not in candidate:
                return "The ambiguous request requires one pizza-service clarifying question."
            if not set(normalize_text(candidate).split()).intersection(
                {"address", "bill", "cart", "delivery", "menu", "order", "pizza", "pickup", "size"}
            ):
                return "The clarification did not return the customer to pizza service."
        if response_kind == "missing_item_context":
            if "?" not in candidate:
                return "The missing-item reply must ask which listed pizza the customer means."
            candidate_tokens = set(normalize_text(candidate).split())
            if not candidate_tokens.intersection({"name", "pizza", "which"}):
                return "The missing-item reply did not ask for the pizza identity."
            catalog_terms = {
                token
                for item in self.catalog.active_items
                for value in (item.name, item.title, *item.aliases, *item.ingredients)
                for token in normalize_text(value).split()
                if token not in {"pizza", "small", "medium", "large"}
            }
            invented_details = sorted(candidate_tokens.intersection(catalog_terms))
            if invented_details:
                return (
                    "The model supplied catalog details without an identified item: "
                    + ", ".join(invented_details[:8])
                    + "."
                )
        if response_kind == "unknown_catalog_item":
            unsupported_state = next(
                (
                    value
                    for value in ("you selected", "you've selected", "you have selected", "you changed", "you switched")
                    if value in lowered
                ),
                "",
            )
            if unsupported_state:
                return f"The model invented a customer selection or change: {unsupported_state}."
            if "?" not in candidate:
                return "The unknown-item reply must ask which available pizza the customer wants."
            if not set(normalize_text(candidate).split()).intersection(
                {"available", "listed", "menu", "pizza"}
            ):
                return "The unknown-item reply did not return the customer to a listed pizza choice."
        if response_kind == "address_security_redirect":
            candidate_tokens = set(normalize_text(candidate).split())
            if not candidate_tokens.intersection({"area", "building", "street"}):
                return "The address retry did not ask for usable address details."
            if not candidate_tokens.intersection({"provide", "resend", "send", "share"}):
                return "The address retry did not ask the customer to provide the details."
        if response_kind == "invalid_address":
            candidate_tokens = set(normalize_text(candidate).split())
            has_address_details = bool(candidate_tokens.intersection({"address", "area", "building", "street"}))
            if not has_address_details:
                return "The invalid-address reply did not ask for usable address details."
            if "current location" in candidate.casefold() or re.search(
                r"\bpickup\b.{0,20}\b(?:is|selected|set)\b|\b(?:is|selected|set)\b.{0,20}\bpickup\b",
                candidate,
                re.I,
            ):
                return "The invalid-address reply invented a pickup selection or location."
        if response_kind == "ambiguous_confirmation":
            candidate_tokens = set(normalize_text(candidate).split())
            if not {"yes", "no"}.issubset(candidate_tokens):
                return "The ambiguous confirmation reply must present both yes and no choices."
            if re.search(r"\b(?:go ahead|please proceed|yes please)\b", candidate.casefold()):
                return "The ambiguous confirmation reply interpreted the customer's choice."
            if "cart" in candidate_tokens and candidate_tokens.intersection({"add", "adding"}):
                return "The ambiguous confirmation reply restarted cart collection."
        if response_kind == "intent_conflict":
            if re.search(
                r"\b(?:i(?:'ll|\s+(?:can|shall|will))|we(?:'ll|\s+(?:can|shall|will)))\b.{0,35}"
                r"\b(?:apply|handle|proceed|queue|remember|update)\b",
                lowered,
            ):
                return "The intent-conflict reply promised to apply a rejected action later."
            if "confirm" in normalize_text(candidate).split():
                return "The intent-conflict reply asked for confirmation despite no draft change."
            if not (
                "one action" in normalize_text(candidate)
                or "separate message" in normalize_text(candidate)
                or "separately" in normalize_text(candidate)
            ):
                return "The intent-conflict reply did not ask for one action at a time."
        if response_kind == "removal_limit":
            if set(normalize_text(candidate).split()).intersection({"add", "adding", "confirm"}):
                return "The removal-limit reply suggested an unrelated cart or confirmation action."
            if not set(normalize_text(candidate).split()).intersection(
                {"amount", "fewer", "quantity", "smaller"}
            ):
                return "The removal-limit reply did not ask for a valid smaller quantity."
            if not re.search(r"\b(?:did not|didn't|has not|hasn't|no)\b.{0,25}\b(?:change|changed|update|updated)\b", lowered):
                return "The removal-limit reply did not state that the draft remained unchanged."
        if response_kind == "quantity_limit":
            limits = set(re.findall(r"\b\d+\b", brief))
            candidate_numbers = set(re.findall(r"\b\d+\b", candidate))
            if limits and not limits.intersection(candidate_numbers):
                return "The quantity-limit reply omitted the deterministic limit."
            if "whole quantities" in brief.casefold():
                if not {"1", "20"}.issubset(candidate_numbers) or "whole" not in normalize_text(candidate).split():
                    return "The malformed-quantity reply omitted the valid whole-number range."
                if "exceed" in lowered:
                    return "The malformed-quantity reply falsely said the customer exceeded the limit."
            elif not set(normalize_text(candidate).split()).intersection(
                {"limit", "limited", "maximum", "smaller", "up"}
            ):
                return "The quantity-limit reply did not explain that a smaller quantity is required."
        if response_kind == "fulfilment_saved":
            required_total = next(iter(re.findall(MONEY_PATTERN, brief)), "")
            if required_total and _normal_money(required_total) not in {
                _normal_money(value) for value in re.findall(MONEY_PATTERN, candidate)
            }:
                return "The fulfilment reply omitted the deterministic draft total."
            if "confirm order" not in normalize_text(candidate):
                return "The fulfilment reply omitted the required review command."
            required_method = "delivery" if "delivery" in normalize_text(brief) else "pickup"
            if required_method not in normalize_text(candidate).split():
                return "The fulfilment reply omitted the selected fulfilment method."
        if required_ingredients and (
            asks_about_ingredients(user_message) or asks_about_ingredient_amount(user_message)
        ):
            ingredient_words = {"mozzarella", "tomato", "chicken", "pepperoni", "pepper", "onion", "olive", "mushroom"}
            if not set(normalize_text(candidate).split()).intersection(ingredient_words):
                return "The model did not answer the ingredient question from the catalog brief."
            if required_ingredients:
                exact_ingredients = [normalize_text(value) for value in required_ingredients]
                normalized_candidate = normalize_text(candidate)
                missing = [value for value in exact_ingredients if value and value not in normalized_candidate]
                if missing:
                    return "The model omitted catalog ingredients: " + ", ".join(missing) + "."
                candidate_words = set(re.findall(r"[a-z0-9]+", normalized_candidate))
                allowed_food_words = set().union(*(set(value.split()) for value in exact_ingredients))
                catalog_food_words = {
                    word
                    for item in self.catalog.active_items
                    for ingredient in item.ingredients
                    for word in normalize_text(ingredient).split()
                }
                common_unlisted_food_words = {
                    "bacon", "basil", "cheddar", "corn", "cream", "egg", "eggs", "flour", "garlic", "ham",
                    "milk", "oregano", "parmesan", "pineapple", "pork", "sausage", "spinach", "tuna", "water",
                }
                family_words = {"cheese", "garden", "heat", "pizza", "special"}
                extras = sorted(
                    candidate_words
                    .intersection(catalog_food_words | common_unlisted_food_words)
                    .difference(allowed_food_words | family_words)
                )
                if extras:
                    return "The model added details outside the catalog ingredient list: " + ", ".join(extras[:8]) + "."
        return ""

    @staticmethod
    def _stage_fact_guard(
        candidate: str,
        brief: str,
        *,
        response_kind: str = "generic",
        required_families: Sequence[str] = (),
        required_sizes: Sequence[str] = (),
        required_ingredients: Sequence[str] = (),
        required_order_items: Sequence[str] = (),
    ) -> str:
        normalized_candidate = normalize_text(candidate)
        normalized_brief = normalize_text(brief)
        money = {_normal_money(value) for value in re.findall(MONEY_PATTERN, brief)}
        if brief.startswith("The customer wants to browse the menu.") and re.search(
            r"(?m)^\s*(?:[-*]|\d+[.)])\s+",
            candidate,
        ):
            return "The menu reply must be one natural sentence rather than a list."
        if brief.startswith("The customer wants to browse the menu.") and "\n" in candidate.strip():
            return "The menu reply must be one natural sentence without an internal preface."
        if brief.startswith("The customer wants to browse the menu.") and re.search(
            r"\bwhich\b.{0,70}\b(?:available|have)\b",
            candidate,
            re.I,
        ):
            return "The menu reply must state the available families before asking for a preference."
        if brief.startswith("Confirmation-gate facts:"):
            if "\n" in candidate.strip() or re.search(r"(?m)^\s*(?:[-*#|]|\d+[.)])\s*", candidate):
                return "The confirmation reply must be one natural sentence without internal labels, lists, or tables."
            words = set(normalized_candidate.split())
            explicit_choice = {"yes", "no"}.issubset(words)
            natural_choice = "place" in words and bool(words.intersection({"edit", "editing"}))
            if not (explicit_choice or natural_choice):
                return "The confirmation reply must offer place-order and keep-editing choices."
            if money and not all(value in {_normal_money(found) for found in re.findall(MONEY_PATTERN, candidate)} for value in money):
                return "The confirmation reply omitted the deterministic total."
        if brief.startswith("Draft-awaiting-address facts:"):
            address_cues = (
                "address" in normalized_candidate.split()
                or "location" in normalized_candidate.split()
                or "where should" in candidate.casefold()
                or "where would" in candidate.casefold()
            )
            if not address_cues:
                return "The draft reply must ask for the delivery address."
        if brief.startswith("Draft-awaiting-fulfilment facts:"):
            words = set(normalized_candidate.split())
            if re.search(
                r"\b(?:would you like|do you want|shall i|should i|can i)\b.{0,45}\badd\b",
                candidate,
                re.IGNORECASE,
            ):
                return "The draft reply asked to add an item that is already in the cart."
            offers_delivery = bool(words.intersection({"delivery", "delivered", "deliver"}))
            offers_pickup = bool(words.intersection({"pickup", "collection", "collect"})) or bool(
                re.search(r"\bpick(?:ed)?\s+(?:it\s+)?up\b", candidate, re.I)
            )
            if not (offers_delivery and offers_pickup):
                return "The draft reply must ask the customer to choose delivery or pickup."
        if brief.startswith("Handover facts:"):
            words = set(normalized_candidate.split())
            identifies_staff = bool(
                words.intersection({"human", "staff", "employee", "person", "operator", "representative"})
            )
            identifies_transfer = bool(
                words.intersection({"transfer", "transferring", "connect", "connecting", "escalate", "escalating"})
            )
            if not (identifies_staff and identifies_transfer):
                return "The handover reply must clearly say that the request is being transferred to staff."
            if not words.intersection({"preserved", "saved", "kept", "intact", "retained"}):
                return "The handover reply must say that the conversation and cart are preserved."
        if brief.startswith("Pizza-domain redirect facts:"):
            words = set(normalized_candidate.split())
            if normalized_candidate.startswith("understood"):
                return "The redirect must not use the repeated canned opening 'Understood'."
            if re.search(r"```|\bdef\s+\w+|\d\s*[+*/=]\s*\d", candidate, re.I):
                return "The redirect answered the unrelated programming or calculation request."
            if "pizza menu" not in normalized_candidate and "pizza order" not in normalized_candidate:
                return "The redirect must keep the customer in pizza menu or ordering support."
            if "can't" not in candidate.casefold() and not words.intersection(
                {"cannot", "cant", "only", "focus", "focused", "instead", "outside", "limited", "handle"}
            ):
                return "The redirect must clearly decline the unrelated request before offering pizza help."
            if re.search(
                r"\b(?:help|assist|focus|details|specify)\b.{0,45}\b(?:programming|coding|science|calculation|creative writing|poetry)\b"
                r"|\b(?:programming|coding|science|calculation|creative writing|poetry)\b.{0,45}\b(?:help|assist|focus|details|specify)\b",
                candidate,
                re.I,
            ):
                return "The redirect offered to continue the unrelated request instead of pizza service."
        if response_kind == "order_lookup":
            supplied_references = set(re.findall(r"\b[A-F0-9]{8}\b", brief, re.I))
            supplied_money = {_normal_money(value) for value in re.findall(MONEY_PATTERN, brief)}
            candidate_money = {_normal_money(value) for value in re.findall(MONEY_PATTERN, candidate)}
            if not supplied_references and re.search(
                r"(?:thanks?|thank you).{0,35}(?:providing|sending|sharing).{0,25}order (?:number|reference)",
                candidate,
                re.I,
            ):
                return "The lookup reply falsely claimed that an order number had already been supplied."
            if supplied_references and not supplied_references.issubset(set(re.findall(r"\b[A-F0-9]{8}\b", candidate, re.I))):
                return "The order-lookup reply omitted the persisted order reference."
            if supplied_money and not supplied_money.issubset(candidate_money):
                return "The order-lookup reply omitted the persisted total."
            if candidate_money - supplied_money:
                return "The order-lookup reply introduced an amount outside the persisted total."
            missing_items = [item for item in required_order_items if item not in normalized_candidate]
            if missing_items:
                return "The order-lookup reply omitted persisted items: " + ", ".join(missing_items) + "."
            if " was confirmed " in f" {normalized_brief} " and "confirmed" not in normalized_candidate.split():
                return "The order-lookup reply omitted the recorded confirmed status."
            if "fulfilment delivery" in normalized_brief and "delivery" not in normalized_candidate.split():
                return "The order-lookup reply omitted the delivery fulfilment method."
            if "fulfilment pickup" in normalized_brief and "pickup" not in normalized_candidate.split():
                return "The order-lookup reply omitted the pickup fulfilment method."
            unsupported_status = next(
                (
                    phrase
                    for phrase in ("active", "fulfilled", "delivered", "completed", "in transit", "preparing")
                    if phrase in normalized_candidate and phrase not in normalized_brief
                ),
                "",
            )
            if unsupported_status:
                return f"The order-lookup reply invented an unsupported order status: {unsupported_status}."
        if response_kind == "catalog_match":
            if "\n" in candidate.strip() or re.search(r"(?m)^\s*(?:[-*#]|\d+[.)])\s*", candidate):
                return "The catalog-match reply must be one natural sentence without headings or lists."
            missing = [
                normalize_text(value)
                for value in (*required_families, *required_sizes)
                if value and normalize_text(value) not in normalized_candidate
            ]
            if missing:
                return "The catalog-match reply omitted a required family or size: " + ", ".join(missing) + "."
            if re.search(MONEY_PATTERN, candidate) or re.search(MEASUREMENT_PATTERN, candidate, re.I):
                return "The catalog-match reply must leave exact prices and measurements in the product cards."
        if required_ingredients and (
            "\n" in candidate.strip() or re.search(r"(?m)^\s*(?:[-*#]|\d+[.)])\s*", candidate)
        ):
            return "The ingredient reply must be one natural sentence without headings or lists."
        if response_kind == "ingredient_amount":
            if re.search(r"\b(?:var(?:y|ies)|depend(?:s|ing)?|included\s+in)\b", candidate, re.I):
                return "The reply inferred how ingredient amounts vary even though the catalog does not provide them."
            unavailable = re.search(
                r"\b(?:not|isn't|aren't|doesn't|does not)\b.{0,45}\b(?:listed|provided|specified|available|give|given)\b",
                candidate,
                re.I,
            )
            if not unavailable:
                return "The reply must say that per-ingredient amounts are not listed in the catalog."
        if brief.startswith("Address-and-bill facts:"):
            if "confirm" not in normalized_candidate.split():
                return "The address reply must ask the customer to confirm the order."
            if money and not all(value in {_normal_money(found) for found in re.findall(MONEY_PATTERN, candidate)} for value in money):
                return "The address reply omitted the deterministic total."
            address_match = re.search(r"Delivery address saved:\s*([^\n]+)", brief, re.I)
            if address_match:
                address = address_match.group(1).strip().rstrip(".")
                if normalize_text(address) not in normalized_candidate:
                    return "The address reply omitted the saved delivery address."
        if response_kind == "confirmed_order":
            references = set(re.findall(r"\b[A-F0-9]{8}\b", brief, re.I))
            candidate_references = set(re.findall(r"\b[A-F0-9]{8}\b", candidate, re.I))
            if references and not references.issubset(candidate_references):
                return "The confirmed-order reply omitted the persisted order reference."
            candidate_money = {_normal_money(value) for value in re.findall(MONEY_PATTERN, candidate)}
            if money and not money.issubset(candidate_money):
                return "The confirmed-order reply omitted the deterministic total."
            if candidate_money - money:
                return "The confirmed-order reply introduced an amount outside the deterministic total."
            requires_delivery = bool(
                "fulfilment delivery" in normalized_brief
                or "confirmed for delivery" in normalized_brief
            )
            requires_pickup = bool(
                "fulfilment pickup" in normalized_brief
                or "confirmed for pickup" in normalized_brief
            )
            if requires_delivery and "pickup" in normalized_candidate.split():
                return "The confirmed-order reply changed delivery to pickup."
            if requires_pickup and "delivery" in normalized_candidate.split():
                return "The confirmed-order reply changed pickup to delivery."
            if requires_delivery and "delivery" not in normalized_candidate.split():
                return "The confirmed-order reply omitted delivery fulfilment."
            if requires_pickup and "pickup" not in normalized_candidate.split():
                return "The confirmed-order reply omitted pickup fulfilment."
            address_match = re.search(r"delivery address\s+(.+?)(?:\.|$)", brief, re.I)
            if address_match:
                address = address_match.group(1).strip().rstrip(".")
                if normalize_text(address) not in normalized_candidate:
                    return "The confirmed-order reply omitted the saved delivery address."
        return ""


def collect_stream(fragments: Iterable[str]) -> str:
    """Collect a provider stream for non-UI callers and tests."""

    return "".join(fragments)


def _compact_service_sentences(value: str) -> str:
    """Join a short model-generated service reply without changing its facts."""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", value.strip()) if part.strip()]
    if len(sentences) <= 2 or len(value.split()) > 45:
        return value
    parts: list[str] = []
    for index, sentence in enumerate(sentences):
        part = sentence.rstrip(".")
        if index and part.split(maxsplit=1)[0] in {"Please", "Say", "The", "Type", "Your"}:
            part = part[:1].lower() + part[1:]
        parts.append(part)
    return "; ".join(parts) + "."


def _normal_money(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace(",", "").strip().upper())


def _trusted_order_total(value: str) -> str:
    """Read a total only from deterministic bill or persisted-total fields."""

    for pattern in (
        rf"Grand\s+total[^\r\n]*?({MONEY_PATTERN})",
        rf"\bTotal:\s*({MONEY_PATTERN})",
    ):
        match = re.search(pattern, value, re.I)
        if match:
            return match.group(1)
    return ""


def _sanitize_address_value(value: str) -> str:
    """Keep legacy customer address text from becoming transaction facts in a model brief."""

    cleaned = re.sub(MONEY_PATTERN, "", value, flags=re.I)
    cleaned = re.sub(
        r"\b(?:grand\s+total|total|price|cost|bill)\b\s*(?::|is)?\s*",
        "",
        cleaned,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")


def _sanitize_legacy_address_facts(value: str) -> str:
    match = re.search(r"(Delivery address:\s*)(.+?)(?=\.(?:\s|$)|$)", value, re.I)
    if not match:
        return value
    safe_address = _sanitize_address_value(match.group(2)) or "address details withheld"
    return value[:match.start(2)] + safe_address + value[match.end(2):]


def _normalize_allowed_money_format(value: str, response: AgentResponse) -> str:
    """Remove model-added .00 only for an exact deterministic integer amount."""

    allowed = set(re.findall(MONEY_PATTERN, response.content))
    allowed.update(f"{item.currency} {item.price:,}" for item in response.menu_attachments)
    cleaned = value
    for amount in sorted(allowed, key=len, reverse=True):
        if "." in amount:
            continue
        cleaned = re.sub(re.escape(amount) + r"\.00(?!\d)", amount, cleaned, flags=re.IGNORECASE)
    return cleaned


def _normal_measurement(value: str) -> str:
    normalized = normalize_text(value)
    number_words = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    parts = normalized.split()
    if parts and parts[0] in number_words:
        parts[0] = number_words[parts[0]]
    unit_aliases = {
        "inches": "inch", "centimeters": "cm", "centimeter": "cm",
        "centimetres": "cm", "centimetre": "cm", "kilograms": "kg",
        "kilogram": "kg", "grams": "gram",
    }
    if parts:
        parts[-1] = unit_aliases.get(parts[-1], parts[-1])
    return " ".join(parts)


def _off_topic_category(value: str) -> str:
    tokens = set(normalize_text(value).split())
    categories = (
        ({"code", "coding", "function", "javascript", "python"}, "programming request"),
        ({"algebra", "calculate", "calculus", "equation", "math", "mathematics"}, "calculation request"),
        ({"biology", "chemistry", "physics", "science"}, "science question"),
        ({"essay", "poem", "story"}, "creative-writing request"),
        ({"politics", "president"}, "politics question"),
        ({"translate", "translation"}, "translation request"),
        ({"weather"}, "weather question"),
    )
    return next((label for words, label in categories if tokens.intersection(words)), "unrelated request")


def _redirect_tone(value: str) -> str:
    tones = ("warm and direct", "friendly and concise", "helpful and matter-of-fact")
    return tones[sum(ord(character) for character in normalize_text(value)) % len(tones)]


def _ground_domain_redirect(value: str, category: str) -> str:
    cleaned = re.sub(r"^\s*(?:understood|sure)[.!,:;-]*\s*", "", value, flags=re.IGNORECASE)
    if not cleaned:
        return value
    if re.search(r"```|\bdef\s+\w+|\d\s*[+*/=]\s*\d", cleaned, re.I):
        return cleaned
    contradictory = re.search(
        r"\b(?:help|assist|focus|details|specify)\b.{0,45}\b(?:programming|coding|science|calculation|creative writing|poetry)\b"
        r"|\b(?:programming|coding|science|calculation|creative writing|poetry)\b.{0,45}\b(?:help|assist|focus|details|specify)\b"
        r"|\b(?:provide|specify|share)\b.{0,30}\b(?:details|information)\b"
        r"|\bassist you better\b",
        cleaned,
        re.I,
    )
    if contradictory:
        clauses = [part.strip(" ,;.") for part in re.split(r"[.;]+", cleaned) if part.strip(" ,;.")]
        safe_clause = next(
            (
                clause
                for clause in clauses
                if re.search(r"\bpizza\s+(?:menu|order|preferences?)\b", clause, re.I)
                and not re.search(
                    r"\b(?:programming|coding|science|calculation|creative writing|poetry)\b",
                    clause,
                    re.I,
                )
            ),
            "I can help you choose from the pizza menu or start a pizza order",
        )
        article = "an" if category[:1].casefold() in {"a", "e", "i", "o", "u"} else "a"
        cleaned = f"I can't handle {article} {category} in this chat, but {safe_clause[0].lower() + safe_clause[1:]}"
    if len(re.findall(r"\b(?:can(?:not|'t)|cant)\b", cleaned, re.I)) > 1:
        safe_matches = re.findall(
            r"\b(?:but\s+)?(i\s+can\s+.{0,120}?\bpizza\s+(?:menu|order)\b[^.!?]*)",
            cleaned,
            re.I,
        )
        safe_clause = safe_matches[-1].strip(" ,;.") if safe_matches else "I can help with the pizza menu or a pizza order"
        cleaned = f"I can't change the ordering rules here, but {safe_clause[0].lower() + safe_clause[1:]}"
    lowered = normalize_text(cleaned)
    has_decline = bool(
        re.search(r"\b(?:can\s*not|cannot|can't|cant|outside|limited|decline)\b", cleaned, re.I)
        or set(lowered.split()).intersection({"only", "focus", "focused", "instead"})
    )
    if not has_decline:
        cleaned = "I can't help with that here, but " + cleaned[0].lower() + cleaned[1:]
    lowered = normalize_text(cleaned)
    if "pizza menu" not in lowered and "pizza order" not in lowered:
        if re.search(r"\bmenu\b", cleaned, re.I):
            cleaned = re.sub(r"\bmenu\b", "pizza menu", cleaned, count=1, flags=re.I)
        elif re.search(r"\border(?:s|ing)?\b", cleaned, re.I):
            cleaned = re.sub(r"\border(?:s|ing)?\b", "pizza order", cleaned, count=1, flags=re.I)
        else:
            cleaned = "I can't help with that request here, but I can help with the pizza menu or start a pizza order."
    return _clean_presentation(cleaned[0].upper() + cleaned[1:])


def _clean_presentation(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith(('"', "'")) and not cleaned.endswith(cleaned[0]):
        cleaned = cleaned[1:].lstrip()
    cleaned = re.sub(
        r"^(?:understood|of course|certainly)(?:\s*[.!,:;-]+|\s+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^based on (?:the )?(?:provided|supplied) facts,?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return re.sub(r"\bi\b", "I", cleaned)


def _complete_order_lookup_facts(
    value: str,
    operational_text: str,
    attachments: Sequence[object],
) -> str:
    normalized = normalize_text(value)
    facts: list[str] = []
    reference = next(iter(re.findall(r"\b[A-F0-9]{8}\b", operational_text, re.I)), "")
    if reference and reference.casefold() not in value.casefold():
        facts.append(f"reference {reference.upper()}")
    missing_items = [
        str(getattr(item, "title", ""))
        for item in attachments
        if str(getattr(item, "title", ""))
        and normalize_text(str(getattr(item, "title", ""))) not in normalized
    ]
    if missing_items:
        facts.append("items: " + ", ".join(missing_items))
    total = next(iter(re.findall(MONEY_PATTERN, operational_text)), "")
    if total and _normal_money(total) not in {
        _normal_money(found) for found in re.findall(MONEY_PATTERN, value)
    }:
        facts.append("total " + total)
    if " was confirmed " in f" {normalize_text(operational_text)} " and "confirmed" not in normalized.split():
        facts.append("recorded status: confirmed")
    fulfilment = next(
        (
            method
            for method in ("delivery", "pickup")
            if f"fulfilment {method}" in normalize_text(operational_text)
        ),
        "",
    )
    if fulfilment and fulfilment not in normalized.split():
        facts.append("fulfilment method: " + fulfilment)
    if not facts:
        return value
    return value.rstrip(" .") + "; " + "; ".join(facts) + "."


def _display_fragments(value: str, words_per_fragment: int = 4) -> Iterator[str]:
    parts = re.findall(r"\S+\s*", value)
    for index in range(0, len(parts), words_per_fragment):
        yield "".join(parts[index:index + words_per_fragment])
