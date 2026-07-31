"""Gold-format data structures shared by loaders, the eval scorer, and the runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Polarity = Literal["positive", "negative", "neutral", "conflict"]


@dataclass(frozen=True)
class Aspect:
    term: str
    polarity: Polarity


@dataclass(frozen=True)
class GoldExample:
    id: str
    text: str
    domain: str
    aspects: list[Aspect] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "domain": self.domain,
            "aspects": [{"term": a.term, "polarity": a.polarity} for a in self.aspects],
        }

    @staticmethod
    def from_json(d: dict) -> "GoldExample":
        return GoldExample(
            id=d["id"],
            text=d["text"],
            domain=d["domain"],
            aspects=[Aspect(term=a["term"], polarity=a["polarity"]) for a in d["aspects"]],
        )
