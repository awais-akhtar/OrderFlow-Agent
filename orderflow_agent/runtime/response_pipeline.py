"""LangChain orchestration around provider-native generation and deterministic review."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, field_validator


ConversationRole = Literal["user", "assistant"]
ConversationMessage = tuple[ConversationRole, str]
Reviewer = Callable[[str], str | None]


class ResponsePlan(BaseModel):
    """Typed, fact-only input to the customer-language model."""

    model_config = ConfigDict(frozen=True)

    instructions: str = Field(min_length=1)
    facts: str = Field(min_length=1)
    user_request: str = Field(min_length=1)
    history: tuple[ConversationMessage, ...] = ()
    isolate_history: bool = False
    retry_feedback: str = ""

    @field_validator("instructions", "facts", "user_request", "retry_feedback")
    @classmethod
    def _trim_text(cls, value: str) -> str:
        return value.strip()


@dataclass(frozen=True)
class ProviderCall:
    instructions: str
    conversation: tuple[ConversationMessage, ...]


@dataclass(frozen=True)
class CandidateReview:
    candidate: str
    violation: str = ""

    @property
    def accepted(self) -> bool:
        return not self.violation


class ProviderChatModel(BaseChatModel):
    """Expose an OrderFlow provider through LangChain's chat-model contract."""

    provider: Any = Field(exclude=True)

    @property
    def _llm_type(self) -> str:
        return "orderflow-provider-adapter"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"provider": str(getattr(self.provider, "label", type(self.provider).__name__))}

    @staticmethod
    def _provider_inputs(messages: list[BaseMessage]) -> ProviderCall:
        system_parts: list[str] = []
        conversation: list[ConversationMessage] = []
        for message in messages:
            content = str(message.content)
            if message.type == "system":
                system_parts.append(content)
                continue
            role: ConversationRole = "assistant" if message.type == "ai" else "user"
            conversation.append((role, content))
        if not system_parts:
            raise ValueError("The LangChain model call requires a system instruction.")
        return ProviderCall("\n\n".join(system_parts), tuple(conversation))

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        call = self._provider_inputs(messages)
        for fragment in self.provider.stream_generate(call.instructions, call.conversation):
            text = str(fragment)
            if text:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        chunks = list(self._stream(messages, stop=stop, run_manager=run_manager, **kwargs))
        text = "".join(str(chunk.message.content) for chunk in chunks)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class LangChainResponsePipeline:
    """Build provider calls and run ordered post-generation guardrails."""

    def __init__(self, provider: object | None = None) -> None:
        prompt = ChatPromptTemplate.from_messages(
            (
                ("system", "{system_content}"),
                MessagesPlaceholder("history"),
                ("human", "{user_request}"),
            )
        )
        self._prompt_chain = (
            RunnableLambda(self._coerce_plan)
            | RunnableLambda(self._prompt_values)
            | prompt
        )
        self._planning_chain = self._prompt_chain | RunnableLambda(self._provider_call)
        self._generation_chain = (
            self._prompt_chain | ProviderChatModel(provider=provider)
            if provider is not None
            else None
        )
        self._review_chain = (
            RunnableLambda(self._coerce_review_payload)
            | RunnableLambda(self._run_reviewers)
        )

    def prepare(self, plan: ResponsePlan) -> ProviderCall:
        return self._planning_chain.invoke(plan)

    def stream(self, plan: ResponsePlan) -> Iterator[str]:
        if self._generation_chain is None:
            raise RuntimeError("A provider is required for LangChain generation.")
        for chunk in self._generation_chain.stream(plan):
            content = chunk.content
            if isinstance(content, str) and content:
                yield content

    def review(self, candidate: str, reviewers: Sequence[Reviewer]) -> CandidateReview:
        return self._review_chain.invoke(
            {"candidate": candidate, "reviewers": tuple(reviewers)}
        )

    @staticmethod
    def _coerce_plan(value: ResponsePlan | dict[str, object]) -> ResponsePlan:
        return value if isinstance(value, ResponsePlan) else ResponsePlan.model_validate(value)

    @staticmethod
    def _prompt_values(plan: ResponsePlan) -> dict[str, object]:
        system_parts = [plan.instructions, plan.facts]
        if plan.retry_feedback:
            system_parts.append(plan.retry_feedback)
        history = () if plan.isolate_history else plan.history
        return {
            "system_content": "\n\n".join(system_parts),
            "history": [
                ("human" if role == "user" else "ai", content)
                for role, content in history
            ],
            "user_request": plan.user_request,
        }

    @staticmethod
    def _provider_call(prompt_value) -> ProviderCall:
        messages = prompt_value.to_messages()
        if not messages or messages[0].type != "system":
            raise ValueError("The response plan did not produce a system instruction.")
        conversation: list[ConversationMessage] = []
        for message in messages[1:]:
            role: ConversationRole = "assistant" if message.type == "ai" else "user"
            conversation.append((role, str(message.content)))
        return ProviderCall(str(messages[0].content), tuple(conversation))

    @staticmethod
    def _coerce_review_payload(value: dict[str, object]) -> dict[str, object]:
        candidate = str(value.get("candidate", "")).strip()
        if not candidate:
            raise ValueError("A candidate reply is required for review.")
        reviewers = value.get("reviewers", ())
        if not isinstance(reviewers, Sequence):
            raise TypeError("Reviewers must be an ordered sequence.")
        return {"candidate": candidate, "reviewers": reviewers}

    @staticmethod
    def _run_reviewers(value: dict[str, object]) -> CandidateReview:
        candidate = str(value["candidate"])
        reviewers = value["reviewers"]
        assert isinstance(reviewers, Sequence)
        for reviewer in reviewers:
            if not callable(reviewer):
                raise TypeError("Every response reviewer must be callable.")
            violation = reviewer(candidate)
            if violation:
                return CandidateReview(candidate, str(violation))
        return CandidateReview(candidate)
