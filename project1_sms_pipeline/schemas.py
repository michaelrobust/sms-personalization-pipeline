"""Tool-use schemas for the SMS pipeline."""
from __future__ import annotations

SEGMENT_LABELS = [
    "high_intent_browser",
    "price_sensitive",
    "vip_loyalist",
    "new_subscriber",
    "post_purchase",
    "winback_dormant",
]


SEGMENT_TOOL = {
    "name": "emit_segment_classification",
    "description": "Classify a subscriber into one micro-segment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "segment": {"type": "string", "enum": SEGMENT_LABELS},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
        },
        "required": ["segment", "confidence", "reasoning"],
    },
}


PERSONA_TOOL = {
    "name": "emit_persona_frame",
    "description": "Emit motivation, barrier, and 2-3 voice pillars for the subscriber.",
    "input_schema": {
        "type": "object",
        "properties": {
            "headline_motivation": {"type": "string"},
            "primary_barrier": {"type": "string"},
            "voice_pillars": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            },
        },
        "required": ["headline_motivation", "primary_barrier", "voice_pillars"],
    },
}


VARIANT_TOOL = {
    "name": "emit_sms_variants",
    "description": "Emit exactly 3 distinct SMS variants, each <=160 chars with a single CTA.",
    "input_schema": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "body": {"type": "string"},
                        "tone": {
                            "type": "string",
                            "enum": [
                                "urgent",
                                "warm",
                                "playful",
                                "premium",
                                "direct",
                                "empathetic",
                            ],
                        },
                        "cta_action": {"type": "string"},
                        "expected_char_count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 160,
                        },
                    },
                    "required": ["body", "tone", "cta_action", "expected_char_count"],
                },
            }
        },
        "required": ["variants"],
    },
}


FAILURE_CATEGORIES = [
    "tone_off",            # voice doesn't match the persona's voice_pillars
    "cta_unclear",         # CTA missing, fuzzy, or multi-action
    "off_brand",           # makes claims that need legal review or breaks brand voice
    "segment_mismatch",    # could be for any cohort; not specific to this segment
    "char_limit",          # exceeds 160 chars
    "duplicate_angle",     # not meaningfully different from another variant
    "other",
]


JUDGE_TOOL = {
    "name": "emit_acceptability_score",
    "description": "Score each variant on the rubric. When a variant is not "
                   "would_send_unedited, list the failure_categories that apply.",
    "input_schema": {
        "type": "object",
        "properties": {
            "per_variant": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "variant_index": {"type": "integer", "minimum": 0},
                        "tone_match": {"type": "integer", "minimum": 0, "maximum": 3},
                        "cta_clarity": {"type": "integer", "minimum": 0, "maximum": 3},
                        "segment_relevance": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                        },
                        "char_limit_ok": {"type": "boolean"},
                        "would_send_unedited": {"type": "boolean"},
                        "failure_categories": {
                            "type": "array",
                            "items": {"type": "string", "enum": FAILURE_CATEGORIES},
                            "description": (
                                "Empty if would_send_unedited is true. "
                                "Otherwise list every category that applies."
                            ),
                        },
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "variant_index",
                        "tone_match",
                        "cta_clarity",
                        "segment_relevance",
                        "char_limit_ok",
                        "would_send_unedited",
                        "failure_categories",
                        "notes",
                    ],
                },
            },
            "overall_pass": {"type": "boolean"},
        },
        "required": ["per_variant", "overall_pass"],
    },
}
