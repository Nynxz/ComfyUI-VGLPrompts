"""Check thales.py renders byte-for-byte what Bria-AI/vgl's own renderer does.

    git clone -b dev https://github.com/Bria-AI/vgl /path/to/vgl
    python tools/verify_upstream.py /path/to/vgl

Runs upstream's prompting.py (stdlib only; its one in-package import is stubbed) on every payload
variant, with and without a style preset, against upstream's own prompt files, and compares.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

S = str(Path(sys.argv[1]) / "src" / "bria_vgl")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


exc = types.ModuleType("bria_vgl.exceptions")
exc.PromptNotFoundError = FileNotFoundError
sys.modules["bria_vgl"] = types.ModuleType("bria_vgl")
sys.modules["bria_vgl.exceptions"] = exc
up = load("up_prompting", S + "/prompting.py")
th = load("thales", str(Path(__file__).resolve().parent.parent / "thales.py"))
T = Path(S) / "dialects/thales"


def schema(name):
    return (T / "input_schema" / f"{name}.txt").read_text()


vgl = {
    "short_description": "a cat",
    "objects": [
        {
            "description": "cat",
            "location": "center",
            "relationship": "alone",
            "pose": None,
            "texture": "",
        }
    ],
    "background_setting": "room",
    "edit_instruction": "make it blue",
    "lighting": {"conditions": "soft", "direction": "front", "shadows": ""},
    "bogus": 1,
}
vj = json.dumps(
    {
        "short_description": "a cat",
        "objects": [{"description": "cat", "location": "center", "relationship": "alone"}],
        "background_setting": "room",
        "lighting": {"conditions": "soft", "direction": "front"},
        "edit_instruction": "make it blue",
    }
)
P = "a red fox in snow"
cases = [
    (("generate", P, {}), "generate", {"user_prompt": P, "llm_input_schema": schema("generate")}),
    (
        ("inspire", P, {"images": 1}),
        "inspire",
        {"user_prompt": P, "llm_input_schema": schema("inspire")},
    ),
    (("caption", "", {"images": 1}), "caption", {"llm_input_schema": schema("inspire")}),
    (
        ("refine", P, {"input_vgl": json.dumps(vgl)}),
        "refine",
        {"user_prompt": P, "llm_input_schema": schema("inspire"), "input_vgl": vj},
    ),
    (
        ("edit", P, {"images": 1}),
        "edit_image",
        {"user_prompt": P, "llm_input_schema": schema("edit")},
    ),
    (
        ("edit", P, {"images": 3}),
        "edit_multi_images",
        {"user_prompt": P, "llm_input_schema": schema("edit_multi_ref")},
    ),
    (
        ("edit", P, {"images": 1, "input_image_vgl": json.dumps(vgl)}),
        "edit_image_input_vgl",
        {"user_prompt": P, "llm_input_schema": schema("edit"), "input_image_vgl": vj},
    ),
    (
        ("edit", P, {"images": 2, "input_image_vgl": json.dumps(vgl)}),
        "edit_multi_images_input_vgl",
        {"user_prompt": P, "llm_input_schema": schema("edit_multi_ref"), "input_image_vgl": vj},
    ),
    (
        ("edit_with_mask", P, {"images": 1}),
        "edit_image_mask",
        {"user_prompt": P, "llm_input_schema": schema("inpaint")},
    ),
    (
        ("edit_with_mask", P, {"images": 1, "input_image_vgl": json.dumps(vgl)}),
        "edit_image_mask_input_vgl",
        {"user_prompt": P, "llm_input_schema": schema("inpaint"), "input_image_vgl": vj},
    ),
]
ok = 0
for preset in (None, "Moody film noir, high contrast black and white, 35mm grain."):
    for (pl, prompt, kw), d, ctx in cases:
        mine = th.build(pl, prompt, style_preset=preset or "", **kw)
        ref = up.render_prompts(T / "prompts" / d, ctx, style_preset=preset)
        same = (mine.system_prompt, mine.final_prompt) == ref and mine.prompt_dir == d
        ok += same
        if not same:
            print("MISMATCH", pl, list(kw), preset is not None, mine.prompt_dir, d)
print(f"{ok}/{2 * len(cases)} prompt pairs byte-identical to upstream render_prompts")
assert ok == 2 * len(cases)
print("vgl_json == upstream model_dump(exclude_none) form:", th.vgl_json(json.dumps(vgl)) == vj)
gw = Path(S, "gateway.py").read_text()
ns = {}
exec(
    "from typing import Any\n" + gw[gw.find("_JSON_SCHEMA_TYPES =") : gw.find("def _is_anthropic")],
    ns,
)
for n in ("generate", "edit"):
    print(
        n,
        "json_schema identical to upstream converter:",
        ns["_to_standard_json_schema"](th.output_schema(n)) == th.response_schema(n, "json_schema"),
    )
