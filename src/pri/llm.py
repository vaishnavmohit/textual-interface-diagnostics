"""Unified LLM inference client.

Routes text and multimodal calls to one of: OpenAI, Mistral, Google Gemini,
or Ollama. Ollama runs in two modes side-by-side -- local (your daemon) and
cloud (https://ollama.com) -- and the model name decides which one each call
uses, so the same process can mix local and cloud freely.

Routing rules (in order):
    model contains 'gpt'    -> OpenAI         (needs OPENAI_API_KEY)
    model contains 'gemini' -> Gemini         (needs GEMINI_API_KEY)
    model contains 'xtral'  -> Mistral        (needs MISTRAL_API_KEY)
    Ollama cloud model      -> Ollama Cloud   (needs OLLAMA_API_KEY)
    everything else         -> Ollama local   (uses OLLAMA_HOST or localhost)

A model is treated as cloud if any of these holds:
    - it appears in CLOUD_MODELS below
    - the name (or its tag) ends with '-cloud' -- Ollama's own convention

Env vars:
    OPENAI_API_KEY    required for OpenAI calls
    GEMINI_API_KEY    required for Gemini calls
    MISTRAL_API_KEY   required for Mistral / Pixtral calls
    OLLAMA_API_KEY    required for any cloud-Ollama call
    OLLAMA_HOST       local daemon URL (e.g. http://10.10.101.10:11434);
                      defaults to http://localhost:11434 when unset
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any

import ollama
import openai
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

try:
    from mistralai import Mistral
except ImportError:  # keep non-Mistral backends usable if the optional client is absent
    Mistral = None

from .io_utils import encode_image_to_base64, encode_image_to_base64_shrink
# Bongard reasoning uses BongardDecision (same Analysis/Rule/TestImage/Conclusion
# shape as hmr's LOGO schema). Winoground schema selection is added with its
# pipeline (see docs/NEW_PACKAGE_BUILD_PLAN.md increment 9).
from .schemas import BongardDecision as ImageDescriptionSynthLogo

load_dotenv()

logger = logging.getLogger(__name__)

MAX_JSON_RETRIES = 5
TEMP_INCREMENT = 0.2
SEED = 42
MAX_TEMP = 1.0


class ParseFailure(ValueError):
    """All parse/repair attempts failed. Carries the last full model output so
    callers can persist what actually came back -- the exception message alone
    reduces it to a 200-character snippet."""

    def __init__(self, msg: str, raw_output: str = ""):
        super().__init__(msg)
        self.raw_output = raw_output

# When a response comes back empty (the think trace ate the whole output
# budget), reseeding can't help -- the budget is the bottleneck. Instead grow
# the output (num_predict) and context (num_ctx) budgets for the next attempt.
EMPTY_OUTPUT_GROWTH = 2.0   # multiply budgets after an empty response
NUM_CTX_CAP = 32768         # never grow num_ctx beyond this
# Growth was previously capped by num_ctx, which is an Ollama context notion and
# reaches 32768 -- well past what hosted APIs accept as a COMPLETION budget
# (gpt-4o: 16384). Exceeding it turns a recoverable empty response into a hard
# 400, so the output budget needs its own ceiling.
MAX_OUTPUT_TOKENS_CAP = 16384

# JSON repair fallback: by default, reuse the SAME model/client that produced
# the response (no separate model needed). Set JSON_REPAIR_MODEL to a specific
# tag (e.g. a small local model) only if you want repairs routed elsewhere.
JSON_REPAIR_MODEL = os.getenv("JSON_REPAIR_MODEL", "")
JSON_REPAIR_TEMP = 0.0
JSON_REPAIR_MAX_TOKENS = 3000


# ---------------------------------------------------------------------------
# Ollama Cloud model registry
# ---------------------------------------------------------------------------
# Add any model that should always be routed to Ollama Cloud, regardless of
# whether it has a '-cloud' suffix. The '-cloud' suffix on the model name (or
# its tag, e.g. 'gpt-oss:120b-cloud') is also recognized automatically and
# matches Ollama's own naming convention.
CLOUD_MODELS: set[str] = {
    "gpt-oss:120b-cloud",
    "deepseek-v3.1:671b-cloud",
    "qwen3-coder:480b-cloud",
    "kimi-k2:1t-cloud",
    "glm-4.6:cloud",
    "minimax-m2:cloud",
    # NOTE: gemini-3-flash-preview is deliberately NOT listed here. The
    # Bongard-HOI generation grid that every Gemini-3-Flash number in the paper
    # comes from was produced through the Google Gemini API
    # (genai.Client(api_key=GEMINI_API_KEY) in the legacy runner), so routing it
    # to Ollama Cloud would serve a different stack than the results it is
    # compared against -- and on a machine without an Ollama Cloud key it simply
    # 404s. Leaving it out sends it to the "gemini" backend, matching provenance.
    "nemotron-3-super",
    "qwen3.5:397b-cloud",
    "qwen3.5:cloud",
    "gemma4:31b-cloud",
    "glm-5:cloud",
    "minimax-m2.5:cloud",
    "deepseek-v3.2:cloud",
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "kimi-k2.5:cloud",
    "qwen3-vl:235b-cloud"
}


def is_cloud_model(model_name: str) -> bool:
    """Whether the given Ollama model name should route to Ollama Cloud."""
    if model_name in CLOUD_MODELS:
        return True
    # Ollama Cloud convention: the tag ends with -cloud
    if model_name.endswith("-cloud") or model_name.endswith(":cloud"):
        return True
    return False


# ---------------------------------------------------------------------------
# Client construction (lazy: only built when first used)
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None
_gemini_client: genai.Client | None = None
_mistral_client: Any | None = None
_ollama_local_client: ollama.Client | None = None
_ollama_cloud_client: ollama.Client | None = None


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"{name} is not set in environment or .env")
    return val


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    return _openai_client


def _gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))
    return _gemini_client


def _mistral() -> Any:
    global _mistral_client
    if Mistral is None:
        raise RuntimeError(
            "the installed mistralai does not export `Mistral`, which Pixtral/Mistral "
            "models need. mistralai 2.x removed it; install the 1.x line: "
            "pip install 'mistralai>=1.2,<2'"
        )
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=_require_env("MISTRAL_API_KEY"))
    return _mistral_client


def _ollama_local() -> ollama.Client:
    """Return an Ollama client pointed at a local/remote daemon (not cloud).

    Reads OLLAMA_HOST if set; otherwise defaults to http://localhost:11434.
    """
    global _ollama_local_client
    if _ollama_local_client is not None:
        return _ollama_local_client

    host = os.getenv("OLLAMA_HOST")
    if host and not host.startswith("http"):
        host = f"http://{host}"
    _ollama_local_client = ollama.Client(host=host) if host else ollama.Client()
    logger.info("Ollama (local): %s", host or "http://localhost:11434")
    return _ollama_local_client


def _ollama_cloud() -> ollama.Client:
    """Return an Ollama client pointed at https://ollama.com."""
    global _ollama_cloud_client
    if _ollama_cloud_client is not None:
        return _ollama_cloud_client

    api_key = _require_env("OLLAMA_API_KEY")
    _ollama_cloud_client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    logger.info("Ollama (cloud): ollama.com")
    return _ollama_cloud_client


def _ollama_for(model_name: str) -> ollama.Client:
    """Pick the right Ollama client for this model.

    Cloud-hosted models can be reached two ways: directly at ollama.com with an
    OLLAMA_API_KEY, or through a local daemon that proxies them using the
    machine's `ollama signin` identity. We prefer the direct route when a key is
    configured and fall back to the local daemon otherwise, because requiring
    the key would break the (perfectly valid) signed-in-local setup.
    """
    if is_cloud_model(model_name) and os.getenv("OLLAMA_API_KEY"):
        return _ollama_cloud()
    return _ollama_local()


# ---------------------------------------------------------------------------
# Helpers for JSON and Thoughts
# ---------------------------------------------------------------------------

# def _extract_think(content: str) -> tuple[str, str]:
#     """
#     Safely extracts <think> blocks typical of DeepSeek/Ollama models.
#     Returns (cleaned_content, concatenated_thoughts).
#     """
#     parts = re.split(r"<think>", content, flags=re.IGNORECASE)
#     if len(parts) == 1:
#         return content, ""
    
#     cleaned_content = parts[0]
#     thoughts = []
    
#     for part in parts[1:]:
#         if "</think>" in part.lower():
#             thought, rest = re.split(r"</think>", part, maxsplit=1, flags=re.IGNORECASE)
#             thoughts.append(thought.strip())
#             cleaned_content += rest
#         else:
#             # Unclosed <think> tag, treat the rest of the payload as thought
#             thoughts.append(part.strip())
            
#     return cleaned_content.strip(), "\n\n".join(thoughts)

import re
from typing import Any

# Ollama pacing. A local daemon has no rate limit, so the default is 0 and a
# 6,500-image describe run is not padded with hours of sleeping. Set
# OLLAMA_CALL_DELAY (seconds) when pointing at Ollama Cloud, which does.
_OLLAMA_DELAY = float(os.getenv("OLLAMA_CALL_DELAY", "0"))
# Gemini's free tier rate-limits hard, hence a non-zero default; a paid key can
# set GEMINI_CALL_DELAY=0 and run at full speed.
_GEMINI_DELAY = float(os.getenv("GEMINI_CALL_DELAY", "15"))


def _gemini_pace() -> None:
    if _GEMINI_DELAY > 0:
        time.sleep(_GEMINI_DELAY)


def _ollama_pace() -> None:
    if _OLLAMA_DELAY > 0:
        time.sleep(_OLLAMA_DELAY)


def _extract_think(response_data: Any) -> tuple[str, str]:
    """
    Extracts content and thinking from either an Ollama SDK response object
    or a raw string containing <think> tags.
    
    Returns: (cleaned_content, thoughts)
    """
    # 1. THE EASY WAY: If you pass the raw Ollama SDK response object
    if hasattr(response_data, 'message'):
        # The SDK message object HAS these fields and sets them to None when
        # unused, so getattr's default never fires -- `or ""` is what actually
        # guards the strip() below. Without it every non-thinking call raises
        # "'NoneType' object has no attribute 'strip'".
        content = getattr(response_data.message, 'content', None) or ""
        thinking = getattr(response_data.message, 'thinking', None) or ""

        # In case an older model embeds tags inside the content attribute anyway
        if '<think>' in content and not thinking:
            return _parse_raw_tags(content)

        return content.strip(), thinking.strip()

    # 2. THE HARD WAY: If you pass a raw text string (e.g., from an HTTP request)
    if isinstance(response_data, str):
        return _parse_raw_tags(response_data)
        
    # Fallback if an unknown type is passed
    return str(response_data), ""

def _parse_raw_tags(text: str) -> tuple[str, str]:
    """Helper function to parse raw <think> tags using regex."""
    # re.DOTALL is required so the dot (.) matches newline characters (\n)
    
    # 1. Find all properly closed <think>...</think> blocks
    thoughts = re.findall(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    
    # 2. Remove those closed blocks from the main content
    cleaned_content = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # 3. Catch unclosed <think> tags (common if the model hits a max_token limit)
    unclosed_match = re.search(r"<think>(.*)", cleaned_content, flags=re.IGNORECASE | re.DOTALL)
    if unclosed_match:
        thoughts.append(unclosed_match.group(1))
        # Remove the unclosed tag and everything after it from the content
        cleaned_content = re.sub(r"<think>.*", "", cleaned_content, flags=re.IGNORECASE | re.DOTALL)
        
    # Clean up whitespace and join multiple thoughts (if any)
    final_thoughts = "\n\n".join(t.strip() for t in thoughts).strip()
    return cleaned_content.strip(), final_thoughts


def _clean_json(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):]
    elif content.startswith("```"):
        content = content[len("```"):]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _elapsed_ns(start: float) -> int:
    return int((time.perf_counter() - start) * 1e9)


def _usage_value(usage: Any, name: str) -> Any:
    if usage is None:
        return None
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    return value


def _openai_usage_metadata(usage: Any) -> dict[str, int | None]:
    completion_details = _usage_value(usage, "completion_tokens_details")
    prompt_details = _usage_value(usage, "prompt_tokens_details")
    return {
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
        "reasoning_tokens": _usage_value(completion_details, "reasoning_tokens"),
        "cached_tokens": _usage_value(prompt_details, "cached_tokens"),
    }


def _usage_delta(usage: Any) -> int | None:
    """total - (prompt + candidates), or None if any term is unavailable."""
    total = _usage_value(usage, "total_token_count")
    prompt = _usage_value(usage, "prompt_token_count")
    cand = _usage_value(usage, "candidates_token_count")
    if total is None or prompt is None or cand is None:
        return None
    return total - (prompt + cand)


def _gemini_usage_metadata(usage: Any) -> dict[str, int | None]:
    return {
        "prompt_tokens": _usage_value(usage, "prompt_token_count"),
        "candidate_tokens": _usage_value(usage, "candidates_token_count"),
        # None-safe: Gemini omits candidates_token_count when a response carries
        # no candidate content, and raw arithmetic on it raised TypeError AFTER
        # the model had already answered -- discarding 13 otherwise good E8
        # results as "unparseable". Accounting must never fail a scored call.
        "thoughts_tokens": _usage_delta(usage),
        "total_tokens": _usage_value(usage, "total_token_count"),
    }


#  Helper to convert nanoseconds to seconds (up to 2 decimal places)
def ns_to_sec(ns_val: int | None) -> float | None:
    if ns_val is not None:
        return round(ns_val / 1_000_000_000, 2)
    return None

def _ollama_usage_metadata(resp: Any) -> dict[str, int | float | None]:
    """
    Extracts token counts and execution times from an Ollama response object.
    Includes both raw nanoseconds and formatted seconds (2 decimal places).
    """
    # 1. Token Counts
    prompt_tokens = getattr(resp, "prompt_eval_count", None)
    completion_tokens = getattr(resp, "eval_count", None)
    total_tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    # 2. Raw Nanosecond Durations
    total_ns = getattr(resp, "total_duration", None)
    load_ns = getattr(resp, "load_duration", None)
    prompt_ns = getattr(resp, "prompt_eval_duration", None)
    eval_ns = getattr(resp, "eval_duration", None)

    # 3. Build the final metadata dictionary
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        
        # Raw Nanoseconds
        "total_duration_ns": total_ns,
        "load_duration_ns": load_ns,
        "prompt_eval_duration_ns": prompt_ns,
        "eval_duration_ns": eval_ns,
        
        # Converted Seconds
        "total_duration_sec": ns_to_sec(total_ns),
        "load_duration_sec": ns_to_sec(load_ns),
        "prompt_eval_duration_sec": ns_to_sec(prompt_ns),
        "eval_duration_sec": ns_to_sec(eval_ns),
    }


def _total_tokens(metadata: dict[str, int | None]) -> int | None:
    total = metadata.get("total_tokens")
    if total is not None:
        return total
    parts = [
        metadata.get("prompt_tokens"),
        metadata.get("completion_tokens"),
        metadata.get("candidate_tokens"),
    ]
    if any(part is not None for part in parts):
        return sum(part or 0 for part in parts)
    return None


def _gemini_text_and_thought(resp: Any) -> tuple[str, str]:
    answer_parts, thought_parts = [], []
    for part in (resp.candidates[0].content.parts if resp.candidates else []):
        text = getattr(part, "text", None)
        if getattr(part, "thought", False):
            if text:
                thought_parts.append(text)
        elif text:
            answer_parts.append(text)

    answer_text = "".join(answer_parts) or getattr(resp, "text", "")
    thought_text = "".join(thought_parts)
    if not thought_text and resp.candidates:
        thought_text = getattr(resp.candidates[0], "thought", "") or ""
    return answer_text, thought_text


REQUIRED_KEYS = ("Analysis", "Rule", "Test Image", "Conclusion")

# Common key spellings models emit instead of the canonical ones.
_KEY_ALIASES = {
    "testimage": "Test Image",
    "test_image": "Test Image",
    "test image": "Test Image",
    "analysis": "Analysis",
    "rule": "Rule",
    "conclusion": "Conclusion",
}

# Repairing a DESCRIPTION must not impose any key set. The decision-schema
# variant below mandates Analysis/Rule/Test Image/Conclusion and even defaults a
# missing Conclusion to "neg" -- applied to a description that produces a
# fabricated decision wrapping the real content, which is a silent corruption.
_JSON_REPAIR_SYSTEM_GENERIC = """
You are a strict JSON repair tool.

Your job is to repair malformed or truncated JSON so that it parses.

Rules:
1. Return ONLY valid JSON.
2. Do not wrap the JSON in markdown.
3. Do not add comments.
4. Do not explain anything.
5. Preserve every key and value exactly as written.
6. Do not rename, add, drop, reorder, summarize or reinterpret anything.
7. Do not change factual meaning.
8. Only make the minimum edits needed to produce valid JSON.
9. Do NOT impose any particular schema or set of top-level keys.
10. If the input is truncated, close the open structures minimally and keep
    whatever content is present.

You are not solving any task. You are only repairing JSON.
""".strip()

_JSON_REPAIR_SYSTEM = """
You are a strict JSON repair tool.

Your job is to repair malformed JSON so that it parses successfully and matches the required schema.

Rules:
1. Return ONLY valid JSON.
2. Do not wrap the JSON in markdown.
3. Do not add comments.
4. Do not explain anything.
5. Preserve the original data as much as possible.
6. Do not rewrite, improve, summarize, or reinterpret the content.
7. Do not change factual meaning.
8. Only make the minimum edits needed to produce valid JSON.
9. The final object must contain exactly these top-level keys:
   - "Analysis"
   - "Rule"
   - "Test Image"
   - "Conclusion"
10. "Conclusion" must be exactly one of "pos", "neg", "cat_1" or "cat_2".
11. Keep the vocabulary the original used. If it says cat_1 or cat_2, keep that; only map obvious variants ("Category 2" -> "cat_2").
12. Otherwise, if the conclusion is clearly positive/yes/match/true, normalize it to "pos"; if clearly negative/no/non-match/false, to "neg".
13. If a required key is missing but its content clearly exists under another similar key, rename that key.
14. If a required key is missing and cannot be recovered, use an empty string for that key, except "Conclusion".
15. If "Conclusion" is missing and cannot be recovered, use "neg".
16. Do not invent new analytical content.

You are not solving the original task. You are only repairing JSON.
""".strip()


def _normalize_keys(data: dict) -> dict:
    """Map common key aliases to the canonical schema keys."""
    out: dict = {}
    for key, value in data.items():
        canonical = _KEY_ALIASES.get(str(key).strip().lower(), key)
        out.setdefault(canonical, value)
    return out


# Both vocabularies are legitimate answers. Bongard-OW labels its classes
# cat_1/cat_2 and prompts.system_eval asks the reasoner for exactly those, so
# insisting on pos/neg here rejected correct output -- the mirror image of the
# schema_vocabulary failure recorded in provenance/E1_AUDIT.md, where a
# constrained decoder forced pos/neg onto a benchmark that uses cat_1/cat_2.
# pipeline._norm maps both families to a common form for scoring.
VALID_CONCLUSIONS = ("pos", "neg", "cat_1", "cat_2")


def _normalize_conclusion(value: Any) -> Any:
    """Coerce variants like 'Positive', 'NEG.', 'Category 2' into a valid label."""
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    if "cat_2" in low or low in ("category 2", "category_2", "class 2"):
        return "cat_2"
    if "cat_1" in low or low in ("category 1", "category_1", "class 1"):
        return "cat_1"
    if low.startswith("pos"):
        return "pos"
    if low.startswith("neg"):
        return "neg"
    return value


def _coerce_valid(data: Any, required_keys: tuple[str, ...] | None = None) -> dict | None:
    """Return a usable dict, or None if it cannot be salvaged.

    With ``required_keys`` this enforces the decision schema: keys are normalized
    to the canonical spellings, Conclusion is mapped to pos/neg, and anything
    missing a required key is rejected.

    Without it -- which is every description call -- any JSON object is valid.
    Enforcing the decision keys here is what made well-formed descriptions look
    unparseable: they contain none of Analysis/Rule/Test Image/Conclusion, so the
    whole repair ladder ran and failed on output that json.loads accepts.
    """
    if not isinstance(data, dict):
        return None
    if not required_keys:
        return data
    data = _normalize_keys(data)
    if "Conclusion" in data:
        data["Conclusion"] = _normalize_conclusion(data["Conclusion"])
    if not set(required_keys) <= set(data.keys()):
        return None
    if data["Conclusion"] not in VALID_CONCLUSIONS:
        return None
    return data


def _extract_json_block(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, ignoring braces in strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # unbalanced — likely truncated output


def _regex_repairs(text: str) -> str:
    """Cheap textual fixes for the most common malformations."""
    # Smart quotes -> straight quotes
    text = text.translate(str.maketrans({"“": '"', "”": '"',
                                         "‘": "'", "’": "'"}))
    # Trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _try_load(content: str, required_keys: tuple[str, ...] | None = None) -> dict | None:
    try:
        return _coerce_valid(json.loads(content), required_keys)
    except (json.JSONDecodeError, ValueError):
        return None


def _raw_chat(model_name: str, system_prompt: str, user_prompt: str,
              temp: float = 0.0, max_tokens: int = 2048,
              json_schema: dict | None = None) -> str:
    """Minimal text-only completion across backends (used by the JSON repairer).

    Deliberately does NOT route through ``parse_or_repair_json`` to avoid
    recursion; the caller post-processes the result.
    """
    backend = _backend(model_name)

    if backend == "openai":
        resp = _openai().chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            response_format={"type": "json_object"},
            **_openai_token_limit_kwargs(model_name, max_tokens),
        )
        return resp.choices[0].message.content or ""

    if backend == "mistral":
        resp = _mistral().chat.complete(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    if backend == "gemini":
        resp = _gemini().models.generate_content(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temp,
                response_mime_type="application/json",
            ),
            contents=[user_prompt],
        )
        return resp.text or ""

    # ollama (local or cloud)
    # Constrain generation ONLY when the caller wants that schema. Passing the
    # decision schema unconditionally is how a description repair turns into a
    # decision -- the same constrained-decoding failure the E1 audit documents.
    kwargs = {"format": json_schema} if json_schema else {}
    resp = _ollama_for(model_name).chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temp, "seed": SEED, "num_predict": max_tokens},
        **kwargs,
    )
    return _extract_think(resp)[0]


def _ai_repair_json(broken: str, model_name: str,
                    required_keys: tuple[str, ...] | None = None) -> dict | None:
    """Last-resort fallback: ask an LLM to fix the JSON. Returns dict or None.

    By default reuses the same model/client that produced the response
    (JSON_REPAIR_MODEL is empty); set that env var to route repairs elsewhere.
    """
    repair_model = JSON_REPAIR_MODEL or model_name
    if required_keys:
        user_prompt = (
            "Repair the following into a single valid JSON object with the required keys "
            f"({', '.join(required_keys)}):\n\n" + broken
        )
        schema = ImageDescriptionSynthLogo.model_json_schema()
        system = _JSON_REPAIR_SYSTEM
    else:
        # Descriptions have no fixed key set. Naming one here would rewrite the
        # payload into that shape instead of repairing it.
        user_prompt = (
            "The following is meant to be a single valid JSON object but is "
            "malformed or truncated. Repair the syntax and return one valid JSON "
            "object. Preserve every key and value exactly as written; do not "
            "rename, add, drop or summarise anything. If the text is cut off, "
            "close the structure minimally.\n\n" + broken
        )
        schema = None
        system = _JSON_REPAIR_SYSTEM_GENERIC
    try:
        fixed = _raw_chat(
            repair_model,
            system,
            user_prompt,
            temp=JSON_REPAIR_TEMP,
            max_tokens=JSON_REPAIR_MAX_TOKENS,
            json_schema=schema,
        )
    except Exception as e:  # network/backend error during repair
        logger.warning("AI JSON repair call failed: %s", e)
        return None

    fixed = _clean_json(fixed)
    for candidate in (fixed, _extract_json_block(fixed) or "", _regex_repairs(fixed)):
        if candidate:
            data = _try_load(candidate, required_keys)
            if data is not None:
                return data
    return None


def parse_or_repair_json(
    content: str | None,
    model_name: str,
    *,
    allow_ai_repair: bool = True,
    required_keys: tuple[str, ...] | None = None,
) -> str:
    """Return a guaranteed schema-valid canonical JSON string, or raise.

    Escalating fallback ladder:
        1. strip <think> tags + ``` fences
        2. direct json.loads + validate
        3. extract first balanced {...} block
        4. cheap regex repairs (trailing commas, smart quotes)
        5. AI repair call (one extra LLM round-trip)
    """
    text = content or ""
    # 1. strip reasoning tags then code fences
    text = _parse_raw_tags(text)[0]
    cleaned = _clean_json(text)

    # Empty output -> nothing to parse or repair (commonly the model spent its
    # whole num_predict budget on the <think> trace, leaving content empty).
    # Fail immediately instead of wasting an AI-repair round-trip on "".
    if not cleaned.strip():
        raise ValueError(
            "Empty model output (no content to parse). If think=True, the "
            "reasoning trace likely consumed the entire max_tokens budget; "
            "raise max_tokens so the JSON answer fits after the reasoning."
        )

    # 2. direct parse
    data = _try_load(cleaned, required_keys)

    # 3. extract a balanced object
    if data is None:
        block = _extract_json_block(cleaned)
        if block is not None:
            data = _try_load(block, required_keys)
            # 4. regex repairs on the extracted block
            if data is None:
                data = _try_load(_regex_repairs(block), required_keys)

    # 4b. regex repairs on the whole cleaned text as a fallback
    if data is None:
        data = _try_load(_regex_repairs(cleaned), required_keys)

    # 5. AI repair
    if data is None and allow_ai_repair:
        logger.info("JSON unparseable after regex fallbacks; attempting AI repair.")
        data = _ai_repair_json(cleaned, model_name, required_keys)

    if data is None:
        snippet = cleaned[:200] + ("..." if len(cleaned) > 200 else "")
        raise ValueError(f"Unable to parse or repair JSON. Snippet: {snippet!r}")

    return json.dumps(data, ensure_ascii=False)


def parse_json_dict(
    content: str | None,
    model_name: str,
    *,
    required_keys: tuple[str, ...] | None = None,
    allow_ai_repair: bool = True,
) -> dict:
    """``parse_or_repair_json`` as a mapping, returning {} instead of raising.

    The underlying function returns a JSON *string* and raises when the output
    cannot be salvaged. Callers want a dict, and a single unparseable response in
    a 500-problem run must be recorded as an invalid prediction -- not abort the
    run. Pass ``required_keys`` for the decision schema so aliases are normalized
    and Conclusion is mapped to pos/neg.
    """
    try:
        return json.loads(parse_or_repair_json(
            content, model_name,
            allow_ai_repair=allow_ai_repair, required_keys=required_keys))
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        logger.info("unparseable model output recorded as invalid: %s", e)
        return {}


def _validate_synthlogo_json(content: str) -> str:
    """Backwards-compatible strict validator (no repair)."""
    content = _clean_json(content)
    data = json.loads(content)
    required = set(REQUIRED_KEYS)
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    if data["Conclusion"] not in ("pos", "neg"):
        raise ValueError(f"Invalid Conclusion: {data['Conclusion']!r}")
    return content


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def _backend(model_name: str) -> str:
    lower = model_name.lower()
    if model_name in CLOUD_MODELS:
        return "ollama"
    if "gpt" in lower:
        return "openai"
    if "xtral" in lower:
        return "mistral"
    if "gemini" in lower:
        return "gemini"
    return "ollama"


def _openai_token_limit_kwargs(model_name: str, max_tokens: int) -> dict[str, int]:
    lower = model_name.lower()
    if lower.startswith("gpt-5") or lower.startswith(("o1", "o3", "o4")):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


# ---------------------------------------------------------------------------
# Shared generate-with-repair retry driver (used by every backend/function)
# ---------------------------------------------------------------------------

def _grow(value: int | None, cap: int | None) -> int | None:
    """Multiply a budget by EMPTY_OUTPUT_GROWTH, clamped to cap (None passes through)."""
    if value is None:
        return None
    grown = int(value * EMPTY_OUTPUT_GROWTH)
    return min(grown, cap) if cap is not None else grown


def _looks_truncated(raw: str | None) -> bool:
    """Whether the text reads as a JSON object that was cut off mid-generation.

    Opens a brace and never balances it. Distinguishes "the budget ran out" --
    which only a bigger budget fixes -- from "the model emitted something
    malformed", where a reseed is worth trying.
    """
    t = (raw or "").strip()
    if not t.startswith("{"):
        return False
    depth, in_str, esc = 0, False, False
    for c in t:
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
    return depth > 0 or in_str


def _generate_with_repair(
    model_name: str,
    *,
    think: bool,
    base_temp: float,
    base_max_tokens: int,
    base_num_ctx: int | None,
    generate_fn: Any,
    required_keys: tuple[str, ...] | None = None,
    expect_json: bool = True,
) -> str | tuple[str, int | None, int | float | None, str, dict[str, Any]]:
    """Generate, then run the full parse/repair ladder; regenerate on failure.

    ``generate_fn(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx)``
    performs one backend call and returns
    ``(raw_content, thought_text, usage_metadata, duration)``.

    Failure handling per attempt:
      * Empty output (no content -- usually the think trace consumed the whole
        output budget): grow num_predict and num_ctx and regenerate. Reseeding
        alone can't help when the budget is the bottleneck.
      * Non-empty but unparseable: run the full parse/repair ladder; if even
        that fails, raise temperature, reseed, and regenerate.

    Returns the parsed JSON string (``think=False``) or the
    ``(content, tokens, duration, thought, usage)`` tuple (``think=True``).
    Non-JSON exceptions from ``generate_fn`` (e.g. network errors) propagate.
    """
    cur_temp = base_temp
    cur_seed = SEED
    cur_max_tokens = base_max_tokens
    cur_num_ctx = base_num_ctx
    last_exc: Exception | None = None
    last_raw: str = ""

    for attempt in range(MAX_JSON_RETRIES):
        raw, thought, usage, duration = generate_fn(
            cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx
        )
        last_raw = raw or last_raw

        # Empty response -> grow the budgets (num_predict, and num_ctx which is
        # the real ceiling) instead of reseeding with the same exhausted budget.
        if not (raw or "").strip():
            new_ctx = _grow(cur_num_ctx, NUM_CTX_CAP)
            new_max = _grow(cur_max_tokens, MAX_OUTPUT_TOKENS_CAP)
            last_exc = ValueError("empty model output")
            logger.info(
                "[retry %d/%d] empty output; growing budget "
                "num_predict %s->%s, num_ctx %s->%s",
                attempt + 1, MAX_JSON_RETRIES,
                cur_max_tokens, new_max, cur_num_ctx, new_ctx,
            )
            cur_max_tokens, cur_num_ctx = new_max, new_ctx
            cur_seed = SEED + random.randint(1, 1000)
            continue

        try:
            if not expect_json:
                # Prose schemas (C1) have nothing to parse. Validating them as
                # JSON makes the repair ladder REWRITE the prose into a JSON
                # object -- silently converting the condition into the one it is
                # supposed to contrast against. Accept any non-empty text.
                content = raw
            else:
                content = parse_or_repair_json(raw, model_name,
                                               required_keys=required_keys)
        except (json.JSONDecodeError, ValueError) as e:
            last_exc = e
            # A response that starts as valid JSON and simply stops is truncated,
            # not malformed. Reseeding at a higher temperature regenerates the
            # same over-long answer and burns every retry; growing the output
            # budget is the only thing that can help.
            if _looks_truncated(raw):
                new_ctx = _grow(cur_num_ctx, NUM_CTX_CAP)
                new_max = _grow(cur_max_tokens, MAX_OUTPUT_TOKENS_CAP)
                logger.info(
                    "[retry %d/%d] output truncated; growing budget "
                    "num_predict %s->%s, num_ctx %s->%s",
                    attempt + 1, MAX_JSON_RETRIES,
                    cur_max_tokens, new_max, cur_num_ctx, new_ctx,
                )
                cur_max_tokens, cur_num_ctx = new_max, new_ctx
            else:
                logger.info(
                    "[retry %d/%d] parse/repair failed: %s",
                    attempt + 1, MAX_JSON_RETRIES, e,
                )
                cur_temp = min(cur_temp + TEMP_INCREMENT, MAX_TEMP)
            cur_seed = SEED + random.randint(1, 1000)
            continue

        if think:
            return (content, _total_tokens(usage), duration, thought, usage)
        return content

    raise ParseFailure(
        f"Failed after {MAX_JSON_RETRIES} parse/repair attempts. Last error: {last_exc}",
        raw_output=last_raw,
    )


# ---------------------------------------------------------------------------
# Public API: multimodal inference (text + image)
# ---------------------------------------------------------------------------

def llm_inference_multimodal(
    model_name: str,
    prompt: str,
    image_path: str | None = None,
    temp: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
    json_output: bool = True,
) -> str | tuple[str, int | None, int, str, dict[str, Any]]:
    """Returns content, or content plus token/timing/thought metadata if think=True."""

    if image_path is None:
        raise ValueError("image_path must be provided")

    backend = _backend(model_name)

    if backend == "openai":
        image = encode_image_to_base64(image_path)

        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _t0 = time.perf_counter()
            resp = _openai().chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": "auto",
                        }},
                    ],
                }],
                temperature=cur_temp,
                # Only constrain the output to JSON when the schema actually
                # asks for it. The flat/C1 condition requests plain prose, and
                # forcing json_object there makes the model return nothing --
                # which then drives the empty-output budget growth into a
                # provider limit error rather than surfacing the conflict.
                **({"response_format": {"type": "json_object"}} if json_output else {}),
                **_openai_token_limit_kwargs(model_name, cur_max_tokens),
            )
            return (resp.choices[0].message.content, "",
                    _openai_usage_metadata(resp.usage), _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, expect_json=json_output)

    if backend == "mistral":
        image = encode_image_to_base64_shrink(image_path)

        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _t0 = time.perf_counter()
            resp = _mistral().chat.complete(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": "auto",
                        }},
                    ],
                }],
                temperature=cur_temp,
                max_tokens=cur_max_tokens,
            )
            return (resp.choices[0].message.content, "",
                    _openai_usage_metadata(resp.usage), _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, expect_json=json_output)

    if backend == "gemini":
        image = PIL.Image.open(image_path)
        thinking_config = types.ThinkingConfig(include_thoughts=True) if think else None

        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _gemini_pace()
            _t0 = time.perf_counter()
            resp = _gemini().models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    max_output_tokens=cur_max_tokens,
                    temperature=cur_temp if not think else None,
                    **({"response_mime_type": "application/json"} if json_output else {}),
                    thinking_config=thinking_config,
                ),
                contents=[prompt, image],
            )
            usage = _gemini_usage_metadata(resp.usage_metadata)
            if think:
                answer_text, thought_text = _gemini_text_and_thought(resp)
                return (answer_text, thought_text, usage, _elapsed_ns(_t0))
            return (resp.text, "", usage, _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, expect_json=json_output)

    # ollama (local daemon or cloud)
    image = encode_image_to_base64_shrink(image_path)

    def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
        _ollama_pace()
        resp = _ollama_for(model_name).chat(
            model=model_name,
            messages=[{"role": "system", "content": prompt, "images": [image]}],
            think=think,
            options={
                "temperature": cur_temp,
                "seed": cur_seed,
                # Both are required. Ollama defaults num_ctx to 2048, and a single
                # image consumes most of that, so the structured description is
                # truncated mid-JSON and every parse/repair attempt fails on a
                # fragment. The config's num_ctx/max_tokens must reach the server.
                **({"num_ctx": cur_num_ctx} if cur_num_ctx else {}),
                **({"num_predict": cur_max_tokens} if cur_max_tokens else {}),
            },
        )
        cleaned_content, thoughts = _extract_think(resp)
        usage = _ollama_usage_metadata(resp)
        return (cleaned_content, thoughts, usage, usage["eval_duration_sec"])

    return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, expect_json=json_output)


# ---------------------------------------------------------------------------
# Public API: multi-image inference (DVRL — Direct Visual Rule Learning)
# ---------------------------------------------------------------------------

def llm_inference_multiimage(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[str],
    temp: float = 0.0,
    max_tokens: int = 3000,
    num_ctx: int | None = None,
    think: bool = False,
) -> str | tuple[str, int | None, int, str, dict[str, Any]]:
    
    backend = _backend(model_name)

    if backend == "openai":
        content: list = [{"type": "text", "text": user_prompt}]
        for path in image_paths:
            b64 = encode_image_to_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "auto"},
            })

        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _t0 = time.perf_counter()
            resp = _openai().chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=cur_temp,
                response_format={"type": "json_object"},
                **_openai_token_limit_kwargs(model_name, cur_max_tokens),
            )
            return (resp.choices[0].message.content, "",
                    _openai_usage_metadata(resp.usage), _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, required_keys=REQUIRED_KEYS)

    if backend == "mistral":
        content = [{"type": "text", "text": user_prompt}]
        for path in image_paths:
            b64 = encode_image_to_base64_shrink(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _t0 = time.perf_counter()
            resp = _mistral().chat.complete(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=cur_temp,
                max_tokens=cur_max_tokens,
            )
            return (resp.choices[0].message.content, "",
                    _openai_usage_metadata(resp.usage), _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, required_keys=REQUIRED_KEYS)

    if backend == "gemini":
        pil_images = [PIL.Image.open(p) for p in image_paths]
        # Gemini suppresses thoughts under JSON mode; we keep JSON mime and rely
        # on the system prompt for JSON either way.
        thinking_config = types.ThinkingConfig(include_thoughts=True) if think else None

        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _gemini_pace()
            _t0 = time.perf_counter()
            resp = _gemini().models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=cur_temp if not think else None,
                    response_mime_type="application/json",
                    thinking_config=thinking_config,
                ),
                contents=[user_prompt, *pil_images],
            )
            usage = _gemini_usage_metadata(resp.usage_metadata)
            if think:
                answer_text, thought_text = _gemini_text_and_thought(resp)
                return (answer_text, thought_text, usage, _elapsed_ns(_t0))
            return (resp.text, "", usage, _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, required_keys=REQUIRED_KEYS)

    # ollama (local or cloud)
    images_b64 = [encode_image_to_base64_shrink(p) for p in image_paths]

    def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
        _ollama_pace()
        resp = _ollama_for(model_name).chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": images_b64},
            ],
            think=think,
            options={
                "temperature": cur_temp,
                "seed": cur_seed,
                # Both are required. Ollama defaults num_ctx to 2048, and a single
                # image consumes most of that, so the structured description is
                # truncated mid-JSON and every parse/repair attempt fails on a
                # fragment. The config's num_ctx/max_tokens must reach the server.
                **({"num_ctx": cur_num_ctx} if cur_num_ctx else {}),
                **({"num_predict": cur_max_tokens} if cur_max_tokens else {}),
            },
        )
        cleaned_content, thoughts = _extract_think(resp)
        usage = _ollama_usage_metadata(resp)
        return (cleaned_content, thoughts, usage, usage["eval_duration_sec"])

    return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, required_keys=REQUIRED_KEYS)


# ---------------------------------------------------------------------------
# Public API: text-only inference
# ---------------------------------------------------------------------------

def llm_inference(
    model_name: str,
    user_prompt: str,
    system_prompt: str,
    temp: float = 0.0,
    max_tokens: int = 30000,
    num_ctx: int | None = None,
    think: bool = False,
) -> str | tuple[str, int | None, int, str, dict[str, Any]]:
    
    backend = _backend(model_name)

    if backend == "openai":
        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _t0 = time.perf_counter()
            resp = _openai().chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cur_temp,
                **_openai_token_limit_kwargs(model_name, cur_max_tokens),
            )
            return (resp.choices[0].message.content, "",
                    _openai_usage_metadata(resp.usage), _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, required_keys=REQUIRED_KEYS)

    if backend == "mistral":
        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _t0 = time.perf_counter()
            resp = _mistral().chat.complete(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cur_temp,
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content, "",
                    _openai_usage_metadata(resp.usage), _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, required_keys=REQUIRED_KEYS)

    if backend == "gemini":
        thinking_cfg = types.ThinkingConfig(include_thoughts=True) if think else None

        def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
            _t0 = time.perf_counter()
            resp = _gemini().models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    # max_output_tokens=cur_max_tokens,
                    temperature=cur_temp if not think else None,
                    response_mime_type="application/json",
                    thinking_config=thinking_cfg,
                ),
                contents=[user_prompt],
            )
            usage = _gemini_usage_metadata(resp.usage_metadata)
            if think:
                answer_text, thought_text = _gemini_text_and_thought(resp)
                return (answer_text, thought_text, usage, _elapsed_ns(_t0))
            return (resp.text, "", usage, _elapsed_ns(_t0))

        return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, required_keys=REQUIRED_KEYS)

    # ollama (local or cloud): schema-constrained generation; full parse/repair
    # ladder each attempt, temperature/seed bumped only when even repair fails.
    client = _ollama_for(model_name)

    def generate(cur_temp, cur_seed, attempt, cur_max_tokens, cur_num_ctx):
        resp = client.chat(
            model=model_name,
            format=ImageDescriptionSynthLogo.model_json_schema(),
            think=think,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": cur_temp,
                "seed": cur_seed,
                # Same requirement as the multimodal path: without these the
                # server keeps its own defaults, the retry ladder grows numbers
                # that never reach it, and a long answer is truncated at the
                # context edge on every attempt identically.
                **({"num_ctx": cur_num_ctx} if cur_num_ctx else {}),
                **({"num_predict": cur_max_tokens} if cur_max_tokens else {}),
            },
        )
        cleaned_content, thoughts = _extract_think(resp)
        usage = _ollama_usage_metadata(resp)
        return (cleaned_content, thoughts, usage, usage["eval_duration_sec"])

    return _generate_with_repair(model_name, think=think, base_temp=temp, base_max_tokens=max_tokens, base_num_ctx=num_ctx, generate_fn=generate, required_keys=REQUIRED_KEYS)
