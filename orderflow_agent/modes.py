"""Agent-control modes for the same deterministic ordering workflow."""

from __future__ import annotations

from enum import Enum


class AgentMode(str, Enum):
    CONTROLLED = "controlled"
    ASSISTED = "assisted"
    FLEXIBLE = "flexible"

    @property
    def strictness(self) -> int:
        return {
            AgentMode.CONTROLLED: 80,
            AgentMode.ASSISTED: 50,
            AgentMode.FLEXIBLE: 20,
        }[self]


def mode_from_strictness(strictness: int) -> AgentMode:
    if strictness >= 67:
        return AgentMode.CONTROLLED
    if strictness >= 34:
        return AgentMode.ASSISTED
    return AgentMode.FLEXIBLE


def coerce_mode(value: AgentMode | str | int) -> AgentMode:
    if isinstance(value, AgentMode):
        return value
    if isinstance(value, int):
        return mode_from_strictness(value)
    normalized = value.strip().casefold()
    aliases = {
        "controlled": AgentMode.CONTROLLED,
        "scripted": AgentMode.CONTROLLED,
        "assisted": AgentMode.ASSISTED,
        "guided": AgentMode.ASSISTED,
        "flexible": AgentMode.FLEXIBLE,
        "adaptive": AgentMode.FLEXIBLE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown agent mode: {value}") from exc
