"""The Thales dialect of Bria VGL, as prompt text: no LLM call, no ComfyUI imports.

Each payload in `bria_vgl.dialects.thales` picks a prompt pair from `thales/prompts/`, a field
guide from `thales/input_schema/` (rendered into `{llm_input_schema}`), and an output schema from
`thales/output_schema/` (sent as the LLM's response schema, not as text). This module reproduces
that choice and the rendering exactly, so an LLM node in ComfyUI receives the same system and
final prompt `bria_vgl.create` / `bria_vgl.modify` would send.

The text files are vendored from the upstream repo at the commit in `thales/UPSTREAM.json`
(see the README to update them). The functions below are ports of upstream's:

    render / compose_system_prompt   bria_vgl/prompting.py
    clean_vgl / vgl_json             bria_vgl/payload.py (Payload.parse + vgl_json)
    to_standard_json_schema          bria_vgl/gateway.py (_to_standard_json_schema)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent / "thales"

#: The slot a system prompt declares for a style preset. Without it the preset is appended.
STYLE_PRESET_KEY = "style_preset_block"

#: Introduces the preset so it reads as instructions: the user's prompt owns the content, the preset owns the look.
BLOCK_PREAMBLE = (
    "Apply the following style guidelines to the scene you describe. The user's request decides "
    "what is in the scene; these guidelines decide how it looks, and override any default stated above."
)

#: Narrow on purpose: an inline JSON example, a padded ``{ user_prompt }`` or a lone brace stays literal.
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Every edit payload accepts at most this many images (``EditImages.images`` max_length).
MAX_EDIT_IMAGES = 4

#: Upstream's Request default; every Thales payload uses it.
TEMPERATURE = 0.2


@dataclass(frozen=True)
class Payload:
    """One Thales payload's static choices, as in ``payloads.py``."""

    prompts: str
    input_schema: str
    output_schema: str
    #: Takes a user prompt. Only Caption does not.
    prompt: bool = True
    #: How many images it sends: (min, max).
    images: tuple[int, int] = (0, 0)
    #: Refine's baseline VGL, rendered into ``{input_vgl}``.
    input_vgl: bool = False
    #: Edits only: an optional VGL of the first image switches to the ``_input_vgl`` prompt twin.
    input_image_vgl: bool = False
    #: Edit with mask: the mask is painted mid-grey into the image before it is sent.
    mask: bool = False


PAYLOADS: dict[str, Payload] = {
    "generate": Payload("generate", "generate", "generate"),
    "inspire": Payload("inspire", "inspire", "generate", images=(1, 1)),
    "caption": Payload("caption", "inspire", "generate", prompt=False, images=(1, 1)),
    # inspire's schema: upstream notes it is the one refine had, without generate's diversity wording.
    "refine": Payload("refine", "inspire", "generate", input_vgl=True),
    # one image -> edit_image/edit; more -> edit_multi_images/edit_multi_ref (see prompt_dir)
    "edit": Payload(
        "edit_image", "edit", "edit", images=(1, MAX_EDIT_IMAGES), input_image_vgl=True
    ),
    "edit_with_mask": Payload(
        "edit_image_mask", "inpaint", "edit", images=(1, 1), input_image_vgl=True, mask=True
    ),
}


@cache
def _text(*parts: str) -> str:
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


@cache
def output_schema(name: str) -> dict:
    return json.loads(_text("output_schema", f"{name}.json"))


def render(template: str, context: Mapping[str, object]) -> tuple[str, frozenset[str]]:
    """Fill every placeholder ``context`` has a value for and leave the rest. ``None`` counts as unsupplied."""
    unfilled: set[str] = set()

    def substitute(match: re.Match[str]) -> str:
        value = context.get(match.group(1))
        if value is None:
            unfilled.add(match.group(1))
            return match.group(0)
        return str(value)

    return _PLACEHOLDER.sub(substitute, template), frozenset(unfilled)


def compose_system_prompt(
    base_template: str, style_preset: str, context: Mapping[str, object]
) -> tuple[str, frozenset[str]]:
    """Render ``base_template`` with the preset in its slot, or appended if it has none."""
    block = f"{BLOCK_PREAMBLE}\n\n{style_preset.strip()}"
    if f"{{{STYLE_PRESET_KEY}}}" in base_template:
        return render(base_template, {**context, STYLE_PRESET_KEY: f"{block}\n\n"})
    rendered, unfilled = render(base_template, context)
    return f"{rendered.rstrip()}\n\n{block}", unfilled


@dataclass(frozen=True)
class Prompts:
    system_prompt: str
    final_prompt: str
    #: Which ``thales/prompts/`` pair was used, e.g. ``edit_multi_images_input_vgl``.
    prompt_dir: str
    #: The output schema name (``generate`` / ``edit``); see ``output_schema``.
    schema: str
    temperature: float = TEMPERATURE


def build(
    payload: str,
    prompt: str = "",
    *,
    images: int = 0,
    input_vgl: str = "",
    input_image_vgl: str = "",
    style_preset: str = "",
) -> Prompts:
    """The system and final prompt a Thales payload would send.

    ``images`` is how many images go with it (it picks the single/multi edit prompt).
    ``input_vgl`` / ``input_image_vgl`` are VGL JSON; they are normalised with ``vgl_json`` first,
    as upstream renders them from a validated model. An empty ``style_preset`` means none.
    """
    spec = PAYLOADS[payload]
    lo, hi = spec.images
    if not lo <= images <= hi:
        want = f"{lo}" if lo == hi else f"{lo} to {hi}"
        raise ValueError(f"{payload} sends {want} image(s), got {images}")
    if spec.prompt and not prompt.strip():
        raise ValueError(f"{payload} needs a prompt")
    if spec.input_vgl and not input_vgl.strip():
        raise ValueError(f"{payload} needs input_vgl: the VGL to change")

    prompt_dir, input_schema = spec.prompts, spec.input_schema
    if payload == "edit" and images > 1:
        prompt_dir, input_schema = "edit_multi_images", "edit_multi_ref"

    context: dict[str, object] = {"llm_input_schema": _text("input_schema", f"{input_schema}.txt")}
    if spec.prompt:
        context["user_prompt"] = prompt
    if spec.input_vgl:
        context["input_vgl"] = vgl_json(input_vgl)
    if spec.input_image_vgl and input_image_vgl.strip():
        # Every edit prompt has an _input_vgl twin that takes the input image's VGL as its baseline.
        prompt_dir = f"{prompt_dir}_input_vgl"
        context["input_image_vgl"] = vgl_json(input_image_vgl)

    # Supplied even with no preset, so a prompt that declares the slot never ships it.
    full_context = {STYLE_PRESET_KEY: "", **context}
    system_template = _text("prompts", prompt_dir, "system_prompt.txt")
    if style_preset.strip():
        system_prompt, system_unfilled = compose_system_prompt(
            system_template, style_preset, full_context
        )
    else:
        system_prompt, system_unfilled = render(system_template, full_context)
    final_prompt, final_unfilled = render(
        _text("prompts", prompt_dir, "final_prompt.txt"), full_context
    )
    if unfilled := system_unfilled | final_unfilled:
        logger.warning("unfilled prompt placeholders %s in %s", sorted(unfilled), prompt_dir)
    return Prompts(system_prompt, final_prompt, prompt_dir, spec.output_schema)


# ---------------------------------------------------------------- VGL JSON


def _is_filled(value: object) -> bool:
    return not (
        value is None or value in ("", [], {}) or (isinstance(value, float) and math.isnan(value))
    )


def _prune(value: object, schema: Mapping | None) -> object:
    """Bottom-up: drop empty values (as ``boltons.iterutils.remap`` with upstream's ``_is_filled``
    visits containers after rebuilding them), and keep only the fields ``schema`` names, in its
    order -- what validating into the pydantic model and dumping it does."""
    if isinstance(value, dict):
        props = (schema or {}).get("properties")
        keys = [k for k in props if k in value] if props is not None else list(value)
        out = {}
        for k in keys:
            v = _prune(value[k], props.get(k) if props is not None else None)
            if _is_filled(v):
                out[k] = v
        return out
    if isinstance(value, list):
        item = (schema or {}).get("items")
        return [v for v in (_prune(x, item) for x in value) if _is_filled(v)]
    return value


_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


def clean_vgl(text: str, schema: str = "edit") -> dict:
    """An LLM reply (or any VGL JSON) as upstream ``Payload.parse`` would keep it.

    Tolerates a Markdown code fence around the JSON. Fields the schema does not name are dropped,
    as are empty values. ``edit`` is the full Thales VGL (generate's fields + edit_instruction).
    """
    match = _FENCE.match(text)
    body = match.group(1) if match else text
    try:
        data = json.loads(body)
    except json.JSONDecodeError as invalid:
        raise ValueError(f"not valid VGL JSON: {invalid}") from invalid
    if not isinstance(data, dict):
        raise ValueError(f"VGL JSON must be an object, got {type(data).__name__}")
    return _prune(data, output_schema(schema))  # type: ignore[return-value]


def vgl_json(vgl: str | Mapping) -> str:
    """A VGL as it goes into a prompt: cleaned, then ``json.dumps`` spacing like the prompts were tuned on."""
    return json.dumps(
        clean_vgl(vgl) if isinstance(vgl, str) else _prune(dict(vgl), output_schema("edit"))
    )


# ---------------------------------------------------------------- response schema

#: Gemini's OpenAPI dialect spells types in upper case. Standard JSON Schema does not.
_JSON_SCHEMA_TYPES = {
    "OBJECT": "object",
    "ARRAY": "array",
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
}


def to_standard_json_schema(node: object) -> object:
    """The bundled schema, in Gemini's dialect, rewritten as standard JSON Schema."""
    if isinstance(node, list):
        return [to_standard_json_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    converted = {}
    for key, value in node.items():
        if key == "nullable":
            continue
        if key == "type" and isinstance(value, str):
            converted[key] = _JSON_SCHEMA_TYPES.get(value.upper(), value.lower())
        elif key == "properties" and isinstance(value, dict):
            # Keys here are field names, not keywords, so only the values are schemas.
            converted[key] = {name: to_standard_json_schema(sub) for name, sub in value.items()}
        else:
            converted[key] = to_standard_json_schema(value)
    if node.get("nullable") and isinstance(converted.get("type"), str):
        converted["type"] = [converted["type"], "null"]
    if "properties" in converted and "additionalProperties" not in converted:
        converted["additionalProperties"] = False
    return converted


def response_schema(name: str, dialect: str = "gemini") -> dict:
    """The output schema as Gemini takes it (``gemini``, the bytes upstream tuned against) or as
    standard JSON Schema (``json_schema``, for OpenAI-style structured output)."""
    schema = output_schema(name)
    return schema if dialect == "gemini" else to_standard_json_schema(schema)  # type: ignore[return-value]


#: Appended to the system prompt when an LLM node cannot enforce a response schema itself.
SCHEMA_IN_PROMPT = (
    "Your reply must be a single JSON object that validates against this JSON Schema:\n{schema}"
)
