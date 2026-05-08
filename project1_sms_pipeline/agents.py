"""SMS pipeline: segmentation -> persona framing -> variant generation."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any

from shared import AsyncLLMClient, EventLog, LLMClient

from .context_retrieval import retrieve_context
from .schemas import PERSONA_TOOL, SEGMENT_TOOL, VARIANT_TOOL


# ---- Stage 1: segmentation -------------------------------------------------

SEGMENTATION_SYSTEM = """You are a marketing analytics agent for an SMS \
platform. Given a single subscriber's behavioral features, classify them \
into one of six micro-segments. Use ONLY the features supplied. Call the \
emit_segment_classification tool exactly once.

Segment definitions:
- high_intent_browser : visited recently (<=14d), few/no purchases.
- price_sensitive     : multiple low-AOV purchases, responds to discount cues.
- vip_loyalist        : high frequency (10+ purchases / 90d), high AOV.
- new_subscriber      : signed up <14d ago, no purchases yet.
- post_purchase       : recent purchase (<14d), in satisfaction window.
- winback_dormant     : 60+ days since last visit, prior purchase history.

Confidence should reflect feature ambiguity. Treat overlapping signals \
(e.g., recent low-AOV purchase) as ~0.55 confidence, not 0.9."""


@dataclass
class SegmentationResult:
    segment: str
    confidence: float
    reasoning: str


class SegmentationAgent:
    def __init__(self, client: LLMClient, log: EventLog):
        self.client = client
        self.log = log

    def run(self, subscriber: dict[str, Any]) -> SegmentationResult:
        with self.log.span(
            "segmentation", subscriber_id=subscriber.get("subscriber_id")
        ) as ctx:
            user_msg = (
                "Classify this subscriber:\n\n"
                f"{json.dumps(self._features_only(subscriber), indent=2)}"
            )
            resp = self.client.call(
                system=SEGMENTATION_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                tools=[SEGMENT_TOOL],
                tool_choice={"type": "tool", "name": SEGMENT_TOOL["name"]},
                cache_system=True,
                temperature=0.2,
            )
            data = resp.first_tool_input()
            if not data:
                raise RuntimeError("Segmentation agent returned no tool call.")
            ctx["segment"] = data["segment"]
            ctx["confidence"] = data["confidence"]
            ctx["cache_hit"] = resp.cached
            return SegmentationResult(**data)

    @staticmethod
    def _features_only(subscriber: dict[str, Any]) -> dict[str, Any]:
        # Strip ground-truth label so the agent can't cheat at eval time.
        return {k: v for k, v in subscriber.items() if k != "segment_truth"}


# ---- Stage 2: persona framing ---------------------------------------------

PERSONA_SYSTEM = """You build marketing persona frames for SMS personalization. \
You receive: (a) the subscriber's segment label and behavioral features, and \
(b) optionally a 'campaign learnings' snippet retrieved for this cohort.

If a learnings snippet is present, treat it as authoritative. Otherwise rely \
on general best practice. Output one persona frame via emit_persona_frame. \
Keep voice_pillars to 2-3 concise tone words."""


@dataclass
class PersonaFrame:
    headline_motivation: str
    primary_barrier: str
    voice_pillars: list[str]
    context_used: str | None = None


class PersonaFramingAgent:
    def __init__(self, client: LLMClient, log: EventLog):
        self.client = client
        self.log = log

    def run(self, subscriber: dict[str, Any], segment: str) -> PersonaFrame:
        with self.log.span(
            "persona_framing",
            subscriber_id=subscriber.get("subscriber_id"),
            segment=segment,
        ) as ctx:
            ctx_snippet = retrieve_context(segment, subscriber.get("last_category"))
            ctx["context_used"] = bool(ctx_snippet)

            features = {k: v for k, v in subscriber.items() if k != "segment_truth"}
            user_parts = [
                f"Segment: {segment}",
                f"Features: {json.dumps(features, indent=2)}",
            ]
            if ctx_snippet:
                user_parts.append(
                    "Campaign learnings (authoritative):\n" + ctx_snippet.text
                )
            user_msg = "\n\n".join(user_parts)

            resp = self.client.call(
                system=PERSONA_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                tools=[PERSONA_TOOL],
                tool_choice={"type": "tool", "name": PERSONA_TOOL["name"]},
                cache_system=True,
                temperature=0.5,
            )
            data = resp.first_tool_input()
            if not data:
                raise RuntimeError("Persona agent returned no tool call.")
            return PersonaFrame(
                **data,
                context_used=ctx_snippet.text if ctx_snippet else None,
            )


# ---- Stage 3: variant generation ------------------------------------------

VARIANT_SYSTEM = """You are an SMS copywriter for a US D2C brand. Generate \
exactly THREE distinct SMS variants given the persona frame and brand \
guardrails.

Hard constraints:
- Each variant <= 160 characters total.
- Each variant has a single, concrete CTA.
- Variants must differ in angle, not just word swaps.
- No emojis unless voice pillars include 'playful'.
- No legal-review-required claims (no 'guaranteed', 'best price', etc.).

Brand voice baseline: confident, direct, never desperate."""


@dataclass
class SMSVariant:
    body: str
    tone: str
    cta_action: str
    expected_char_count: int

    @property
    def actual_char_count(self) -> int:
        return len(self.body)

    @property
    def char_limit_ok(self) -> bool:
        return self.actual_char_count <= 160


class VariantGenerationAgent:
    def __init__(self, client: LLMClient, log: EventLog):
        self.client = client
        self.log = log

    def run(self, persona: PersonaFrame, segment: str) -> list[SMSVariant]:
        with self.log.span("variant_generation", segment=segment) as ctx:
            user_msg = (
                "Persona frame:\n"
                f"  motivation: {persona.headline_motivation}\n"
                f"  barrier:    {persona.primary_barrier}\n"
                f"  voice:      {', '.join(persona.voice_pillars)}\n"
                f"  segment:    {segment}\n\n"
                "Generate 3 SMS variants now."
            )
            resp = self.client.call(
                system=VARIANT_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                tools=[VARIANT_TOOL],
                tool_choice={"type": "tool", "name": VARIANT_TOOL["name"]},
                cache_system=True,
                temperature=0.8,
                max_tokens=1024,
            )
            data = resp.first_tool_input()
            if not data:
                raise RuntimeError("Variant agent returned no tool call.")
            variants = [SMSVariant(**v) for v in data["variants"]]
            ctx["variant_count"] = len(variants)
            ctx["any_over_limit"] = any(not v.char_limit_ok for v in variants)
            return variants


# ---- orchestrator ----------------------------------------------------------


@dataclass
class PipelineOutput:
    subscriber_id: str
    segment_truth: str | None
    segment_predicted: str
    segment_confidence: float
    persona: PersonaFrame
    variants: list[SMSVariant]


class SMSPipeline:
    def __init__(
        self,
        client: LLMClient | None = None,
        log_path: str = "logs/pipeline.jsonl",
    ):
        self.client = client or LLMClient()
        self.log = EventLog(log_path)
        self.segmentation = SegmentationAgent(self.client, self.log)
        self.persona = PersonaFramingAgent(self.client, self.log)
        self.variant = VariantGenerationAgent(self.client, self.log)

    def run_one(self, subscriber: dict[str, Any]) -> PipelineOutput:
        with self.log.span(
            "pipeline.run_one", subscriber_id=subscriber.get("subscriber_id")
        ):
            seg = self.segmentation.run(subscriber)
            persona = self.persona.run(subscriber, seg.segment)
            variants = self.variant.run(persona, seg.segment)
            return PipelineOutput(
                subscriber_id=subscriber["subscriber_id"],
                segment_truth=subscriber.get("segment_truth"),
                segment_predicted=seg.segment,
                segment_confidence=seg.confidence,
                persona=persona,
                variants=variants,
            )

    def run_one_to_dict(self, subscriber: dict[str, Any]) -> dict[str, Any]:
        out = self.run_one(subscriber)
        return _output_to_dict(out)


# ---- Async variant ---------------------------------------------------------


async def _segmentation_call(
    client: AsyncLLMClient,
    log: EventLog,
    subscriber: dict[str, Any],
) -> SegmentationResult:
    with log.span("segmentation.async", subscriber_id=subscriber.get("subscriber_id")) as ctx:
        features = {k: v for k, v in subscriber.items() if k != "segment_truth"}
        user_msg = "Classify this subscriber:\n\n" + json.dumps(features, indent=2)
        resp = await client.call(
            system=SEGMENTATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=[SEGMENT_TOOL],
            tool_choice={"type": "tool", "name": SEGMENT_TOOL["name"]},
            cache_system=True,
            temperature=0.2,
        )
        data = resp.first_tool_input()
        if not data:
            raise RuntimeError("Segmentation agent returned no tool call.")
        ctx["segment"] = data["segment"]
        ctx["confidence"] = data["confidence"]
        ctx["cache_hit"] = resp.cached
        return SegmentationResult(**data)


async def _persona_call(
    client: AsyncLLMClient,
    log: EventLog,
    subscriber: dict[str, Any],
    segment: str,
) -> PersonaFrame:
    with log.span(
        "persona_framing.async",
        subscriber_id=subscriber.get("subscriber_id"),
        segment=segment,
    ) as ctx:
        ctx_snippet = retrieve_context(segment, subscriber.get("last_category"))
        ctx["context_used"] = bool(ctx_snippet)
        features = {k: v for k, v in subscriber.items() if k != "segment_truth"}
        parts = [
            f"Segment: {segment}",
            f"Features: {json.dumps(features, indent=2)}",
        ]
        if ctx_snippet:
            parts.append("Campaign learnings (authoritative):\n" + ctx_snippet.text)
        resp = await client.call(
            system=PERSONA_SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
            tools=[PERSONA_TOOL],
            tool_choice={"type": "tool", "name": PERSONA_TOOL["name"]},
            cache_system=True,
            temperature=0.5,
        )
        data = resp.first_tool_input()
        if not data:
            raise RuntimeError("Persona agent returned no tool call.")
        return PersonaFrame(**data, context_used=ctx_snippet.text if ctx_snippet else None)


async def _variant_call(
    client: AsyncLLMClient,
    log: EventLog,
    persona: PersonaFrame,
    segment: str,
) -> list[SMSVariant]:
    with log.span("variant_generation.async", segment=segment) as ctx:
        user_msg = (
            "Persona frame:\n"
            f"  motivation: {persona.headline_motivation}\n"
            f"  barrier:    {persona.primary_barrier}\n"
            f"  voice:      {', '.join(persona.voice_pillars)}\n"
            f"  segment:    {segment}\n\n"
            "Generate 3 SMS variants now."
        )
        resp = await client.call(
            system=VARIANT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=[VARIANT_TOOL],
            tool_choice={"type": "tool", "name": VARIANT_TOOL["name"]},
            cache_system=True,
            temperature=0.8,
            max_tokens=1024,
        )
        data = resp.first_tool_input()
        if not data:
            raise RuntimeError("Variant agent returned no tool call.")
        variants = [SMSVariant(**v) for v in data["variants"]]
        ctx["variant_count"] = len(variants)
        ctx["any_over_limit"] = any(not v.char_limit_ok for v in variants)
        return variants


class AsyncSMSPipeline:
    """Async orchestrator. Use with asyncio.Semaphore to bound concurrency."""

    def __init__(
        self,
        client: AsyncLLMClient,
        log_path: str = "logs/pipeline.jsonl",
    ):
        self.client = client
        self.log = EventLog(log_path)

    async def run_one(self, subscriber: dict[str, Any]) -> PipelineOutput:
        with self.log.span(
            "pipeline.run_one.async", subscriber_id=subscriber.get("subscriber_id")
        ):
            seg = await _segmentation_call(self.client, self.log, subscriber)
            persona = await _persona_call(self.client, self.log, subscriber, seg.segment)
            variants = await _variant_call(self.client, self.log, persona, seg.segment)
            return PipelineOutput(
                subscriber_id=subscriber["subscriber_id"],
                segment_truth=subscriber.get("segment_truth"),
                segment_predicted=seg.segment,
                segment_confidence=seg.confidence,
                persona=persona,
                variants=variants,
            )

    async def run_many(
        self,
        subscribers: list[dict[str, Any]],
        max_concurrency: int = 8,
    ) -> list[PipelineOutput]:
        sem = asyncio.Semaphore(max_concurrency)

        async def bounded(s: dict[str, Any]) -> PipelineOutput:
            async with sem:
                return await self.run_one(s)

        return await asyncio.gather(*(bounded(s) for s in subscribers))


def _output_to_dict(out: PipelineOutput) -> dict[str, Any]:
    return {
        "subscriber_id": out.subscriber_id,
        "segment_truth": out.segment_truth,
        "segment_predicted": out.segment_predicted,
        "segment": out.segment_predicted,
        "segment_confidence": out.segment_confidence,
        "persona": asdict(out.persona),
        "variants": [asdict(v) for v in out.variants],
    }


def output_to_dict(out: PipelineOutput) -> dict[str, Any]:  # public alias
    return _output_to_dict(out)
