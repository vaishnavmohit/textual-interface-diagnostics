"""Pydantic schemas for structured LLM output on the natural-image benchmarks.

The reasoner is asked to return parseable JSON so per-sample correctness can be
scored without brittle text matching. One schema per task family:

- BongardDecision   : Bongard-OW / Bongard-HOI query classification (pos/neg)
- WinogroundChoice  : Winoground caption<->image assignment (per sub-decision)

Field names mirror the columns the pipeline records and the COLM prompts, so
the description-generation and reasoning stages stay compatible with the
inherited results format.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BongardDecision(BaseModel):
    """Reasoner output for a Bongard-style query classification.

    ``Conclusion`` uses the benchmark's own category labels **cat_1 / cat_2**,
    matching the reasoner prompt and the COLM `ImageDescription` schema. This
    consistency is load-bearing: Ollama's ``format=<schema>`` *enforces* the
    output vocabulary, so a prompt that says "cat_1 or cat_2" must be paired
    with a cat_1/cat_2 schema — otherwise open-weight (Ollama) reasoners are
    forced to emit a different vocabulary than the prompt asks for and diverge
    from the COLM results (see docs/OLLAMA_REPRODUCTION_NOTE.md). Scoring maps
    cat_2→pos, cat_1→neg via pipeline._norm.
    """

    Analysis: str
    Rule: str
    TestImage: str = Field(alias="Test Image")
    Conclusion: Literal["cat_1", "cat_2"]

    class Config:
        populate_by_name = True


class WinogroundChoice(BaseModel):
    """One Winoground sub-decision: which candidate (cat_0/cat_1) best matches
    the probe. Issued for both the text-score (caption chosen per image
    description) and image-score (description chosen per caption) directions;
    Text/Image/Group scores are composed from these.
    """

    analysis: str
    category: Literal["cat_0", "cat_1"]

    class Config:
        populate_by_name = True


# Registry so the pipeline/llm layer can select a schema by task family without
# importing task-specific names everywhere.
SCHEMA_BY_TASK = {
    "bongard": BongardDecision,
    "winoground": WinogroundChoice,
}
