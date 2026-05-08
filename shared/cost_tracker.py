"""Token + dollar accounting for Anthropic Messages API calls.

Pricing snapshot per 1M tokens (USD). Update as Anthropic publishes new rates.
The `cached_input` rate applies to input tokens served from prompt cache.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PRICES: dict[str, dict[str, float]] = {
    # input / cached_input / output, all USD per 1M tokens
    "claude-opus-4-6":           {"input": 15.00, "cached_input": 1.50,  "output": 75.00},
    "claude-sonnet-4-6":         {"input":  3.00, "cached_input": 0.30,  "output": 15.00},
    "claude-haiku-4-5-20251001": {"input":  1.00, "cached_input": 0.10,  "output":  5.00},
    "claude-haiku-4-5":          {"input":  1.00, "cached_input": 0.10,  "output":  5.00},
}


def _price_for(model: str) -> dict[str, float]:
    if model in PRICES:
        return PRICES[model]
    # Fall back to Sonnet pricing for unknown model strings rather than crashing.
    return PRICES["claude-sonnet-4-6"]


@dataclass
class CostEntry:
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, usage: dict[str, int]) -> None:
        self.calls += 1
        self.input_tokens += usage.get("input_tokens", 0)
        self.cached_input_tokens += usage.get("cache_read_input_tokens", 0)
        self.cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)

    def cost_usd(self) -> float:
        p = _price_for(self.model)
        # Anthropic bills cache_creation at the standard input rate, cached reads at the lower rate.
        billed_input = self.input_tokens + self.cache_creation_input_tokens
        return (
            billed_input * p["input"] / 1_000_000
            + self.cached_input_tokens * p["cached_input"] / 1_000_000
            + self.output_tokens * p["output"] / 1_000_000
        )

    def cache_hit_rate(self) -> float:
        total = self.input_tokens + self.cached_input_tokens + self.cache_creation_input_tokens
        if total == 0:
            return 0.0
        return self.cached_input_tokens / total


@dataclass
class CostLedger:
    by_model: dict[str, CostEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, usage: dict[str, int]) -> None:
        with self._lock:
            entry = self.by_model.setdefault(model, CostEntry(model=model))
            entry.add(usage)

    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(e.cost_usd() for e in self.by_model.values())

    def total_calls(self) -> int:
        with self._lock:
            return sum(e.calls for e in self.by_model.values())

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_cost_usd": round(self.total_cost_usd(), 6),
                "total_calls": self.total_calls(),
                "by_model": {
                    m: {
                        **{k: v for k, v in asdict(e).items() if k != "model"},
                        "cost_usd": round(e.cost_usd(), 6),
                        "cache_hit_rate": round(e.cache_hit_rate(), 4),
                    }
                    for m, e in self.by_model.items()
                },
            }

    def write_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    def format_summary(self) -> str:
        d = self.to_dict()
        lines = [
            f"Cost summary",
            f"  total calls           : {d['total_calls']}",
            f"  total cost (USD)      : ${d['total_cost_usd']:.4f}",
        ]
        for model, m in d["by_model"].items():
            cache_pct = m["cache_hit_rate"] * 100
            lines.append("")
            lines.append(f"  model: {model}")
            lines.append(f"    calls               : {m['calls']}")
            lines.append(f"    input tokens        : {m['input_tokens']:,}")
            lines.append(f"    cached input tokens : {m['cached_input_tokens']:,}  ({cache_pct:.1f}% hit)")
            lines.append(f"    cache creation tok  : {m['cache_creation_input_tokens']:,}")
            lines.append(f"    output tokens       : {m['output_tokens']:,}")
            lines.append(f"    cost (USD)          : ${m['cost_usd']:.4f}")
        return "\n".join(lines)
