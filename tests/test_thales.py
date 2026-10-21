"""The Thales prompts and the two nodes, against a real ComfyUI checkout.

    COMFY_ROOT=/path/to/ComfyUI python tests/test_thales.py

Byte-for-byte parity with upstream's renderer is checked separately by
`tools/verify_upstream.py`, which needs a checkout of Bria-AI/vgl.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness

pack = _harness.load()

import importlib
import json

import torch

thales = importlib.import_module(f"{_harness.PACK_NAME}.thales")
nodes = importlib.import_module(f"{_harness.PACK_NAME}.nodes")

VGL = {
    "bogus_field": 1,
    "background_setting": "a kitchen",
    "short_description": "a cat on a table",
    "objects": [
        {
            "location": "center",
            "description": "cat",
            "relationship": "alone",
            "pose": None,
            "texture": "",
        }
    ],
    "lighting": {"conditions": "soft", "direction": "front", "shadows": ""},
    "text_render": [],
}
CLEAN = {
    "short_description": "a cat on a table",
    "objects": [{"description": "cat", "location": "center", "relationship": "alone"}],
    "background_setting": "a kitchen",
    "lighting": {"conditions": "soft", "direction": "front"},
}


def image(w=64, h=48, b=1, value=0.5):
    return torch.full((b, h, w, 3), value)


def run(payload, **kw):
    return nodes.ThalesPrompt.execute(payload, **kw).result


# ---------------------------------------------------------------- routing


def test_every_payload_routes_to_its_upstream_prompt_pair():
    vgl = json.dumps(VGL)
    cases = [
        (("generate", "p"), {}, "generate", "generate"),
        (("inspire", "p"), {"images": 1}, "inspire", "generate"),
        (("caption", ""), {"images": 1}, "caption", "generate"),
        (("refine", "p"), {"input_vgl": vgl}, "refine", "generate"),
        (("edit", "p"), {"images": 1}, "edit_image", "edit"),
        (("edit", "p"), {"images": 4}, "edit_multi_images", "edit"),
        (("edit", "p"), {"images": 1, "input_image_vgl": vgl}, "edit_image_input_vgl", "edit"),
        (
            ("edit", "p"),
            {"images": 2, "input_image_vgl": vgl},
            "edit_multi_images_input_vgl",
            "edit",
        ),
        (("edit_with_mask", "p"), {"images": 1}, "edit_image_mask", "edit"),
        (
            ("edit_with_mask", "p"),
            {"images": 1, "input_image_vgl": vgl},
            "edit_image_mask_input_vgl",
            "edit",
        ),
    ]
    for args, kw, prompt_dir, schema in cases:
        p = thales.build(*args, **kw)
        assert (p.prompt_dir, p.schema) == (prompt_dir, schema), (args, kw, p.prompt_dir)
        # nothing a payload supplies is left as a literal {placeholder}
        for text in (p.system_prompt, p.final_prompt):
            for key in (
                "user_prompt",
                "llm_input_schema",
                "input_vgl",
                "input_image_vgl",
                "style_preset_block",
            ):
                assert "{" + key + "}" not in text, (prompt_dir, key)


def test_baseline_vgls_go_in_cleaned():
    p = thales.build("refine", "make it night", input_vgl=json.dumps(VGL))
    assert json.dumps(CLEAN) in p.final_prompt
    assert "bogus_field" not in p.final_prompt


def test_style_preset_is_composed_into_the_system_prompt_only():
    plain = thales.build("generate", "a fox")
    styled = thales.build("generate", "a fox", style_preset="Ink wash, muted greens.")
    assert (
        thales.BLOCK_PREAMBLE in styled.system_prompt
        and "Ink wash, muted greens." in styled.system_prompt
    )
    assert thales.BLOCK_PREAMBLE not in plain.system_prompt
    assert styled.final_prompt == plain.final_prompt


def test_missing_inputs_are_refused_like_the_payload_models():
    for args, kw, needle in [
        (("generate", "  "), {}, "needs a prompt"),
        (("refine", "p"), {}, "needs input_vgl"),
        (("inspire", "p"), {"images": 0}, "1 image"),
        (("edit", "p"), {"images": 5}, "1 to 4"),
        (("generate", "p"), {"images": 1}, "0 image"),
    ]:
        try:
            thales.build(*args, **kw)
        except ValueError as exc:
            assert needle in str(exc), (args, str(exc))
        else:
            raise AssertionError(f"accepted {args} {kw}")


# ---------------------------------------------------------------- VGL JSON


def test_clean_vgl_matches_payload_parse():
    assert thales.clean_vgl(json.dumps(VGL)) == CLEAN
    assert list(thales.clean_vgl(json.dumps(VGL))) == list(CLEAN), "field order follows the model"
    fenced = "```json\n" + json.dumps({**VGL, "edit_instruction": "make it blue"}) + "\n```"
    assert thales.clean_vgl(fenced) == {**CLEAN, "edit_instruction": "make it blue"}


def test_clean_vgl_drops_containers_left_empty():
    """remap visits a container after rebuilding it, so one emptied by the cleaning goes too."""
    out = thales.clean_vgl(
        json.dumps(
            {"short_description": "x", "lighting": {"shadows": ""}, "objects": [{"pose": None}]}
        )
    )
    assert out == {"short_description": "x"}


def test_clean_vgl_rejects_non_objects():
    for text in ("not json", "[1, 2]"):
        try:
            thales.clean_vgl(text)
        except ValueError:
            pass
        else:
            raise AssertionError(text)


def test_response_schema_dialects():
    gemini = thales.response_schema("edit", "gemini")
    standard = thales.response_schema("edit", "json_schema")
    assert gemini["type"] == "OBJECT" and "edit_instruction" in gemini["properties"]
    assert standard["type"] == "object" and standard["additionalProperties"] is False
    texture = standard["properties"]["objects"]["items"]["properties"]["texture"]
    assert texture["type"] == ["string", "null"] and "nullable" not in texture
    assert "edit_instruction" not in thales.response_schema("generate")["properties"]


# ---------------------------------------------------------------- nodes


def test_node_generate_has_no_images():
    _system, final, images, schema, temperature = run(
        {"payload": "generate", "prompt": "a red fox in snow"}
    )
    assert "a red fox in snow" in final and images is None and temperature == 0.2
    assert json.loads(schema)["type"] == "object"


def test_node_downscales_like_bria_vgl():
    *_, images, _schema, _t = run({"payload": "caption", "image": image(w=1024, h=1024)})
    assert tuple(images.shape) == (1, 512, 512, 3), "1 MP -> 0.26 MP, aspect kept"
    *_, small, _schema, _t = run({"payload": "inspire", "prompt": "p", "image": image(w=64, h=48)})
    assert tuple(small.shape) == (1, 48, 64, 3), "under the cap: untouched"


def test_node_paints_the_mask_grey_before_downscaling():
    mask = torch.zeros(1, 48, 64)
    mask[:, :, :32] = 1.0
    _s, final, images, _schema, _t = run(
        {
            "payload": "edit_with_mask",
            "prompt": "add a hat",
            "image": image(value=1.0),
            "mask": mask,
        }
    )
    assert "grey masked areas" in final
    grey = round(128 / 255, 4)
    assert round(images[0, 0, 0, 0].item(), 4) == grey and images[0, 0, -1, 0].item() == 1.0


def test_node_refuses_a_mask_of_another_size():
    try:
        run(
            {
                "payload": "edit_with_mask",
                "prompt": "p",
                "image": image(),
                "mask": torch.ones(1, 10, 10),
            }
        )
    except ValueError as exc:
        assert "must match" in str(exc)
    else:
        raise AssertionError("mismatched mask accepted")


def test_node_edit_counts_batch_frames_and_switches_to_multi():
    system, _f, images, schema, _t = run(
        {
            "payload": "edit",
            "prompt": "swap <image_1> and <image_3>",
            "image_1": image(b=2),
            "image_2": image(w=32, h=32),
        },
        schema_in_prompt=False,
    )
    single, *_ = run({"payload": "edit", "prompt": "p", "image_1": image()}, schema_in_prompt=False)
    assert images.shape[0] == 3
    assert tuple(images.shape[1:3]) == (48, 64), "padded to the largest frame"
    assert system == thales.build("edit", "p", images=3).system_prompt != single
    assert "edit_instruction" in json.loads(schema)["properties"]


def test_node_input_image_vgl_selects_the_baseline_twin():
    _s, final, *_ = run(
        {"payload": "edit", "prompt": "p", "image_1": image(), "input_image_vgl": json.dumps(VGL)}
    )
    assert "[EXISTING JSON]" in final and json.dumps(CLEAN) in final


def test_node_writes_the_compact_json_schema_into_the_prompt_by_default():
    upstream = thales.build("generate", "p").system_prompt
    system, *_, schema, _t = run({"payload": "generate", "prompt": "p"})
    compact = json.dumps(json.loads(schema), separators=(",", ":"))
    assert system.startswith(upstream.rstrip()) and system.endswith(compact)
    assert '"type":"object"' in compact, "json_schema is the default dialect"


def test_node_schema_in_prompt_off_is_bria_vgls_exact_prompt():
    system, *_, schema, _t = run(
        {"payload": "generate", "prompt": "p"}, schema_dialect="gemini", schema_in_prompt=False
    )
    assert system == thales.build("generate", "p").system_prompt
    assert '"OBJECT"' in schema, "the output still carries the chosen dialect"


def test_parse_node_outputs():
    reply = "```json\n" + json.dumps({**VGL, "edit_instruction": "make it blue"}) + "\n```"
    vgl, short, instruction = nodes.ParseVGL.execute(reply).result
    assert json.loads(vgl) == {**CLEAN, "edit_instruction": "make it blue"}
    assert (short, instruction) == ("a cat on a table", "make it blue")


def test_extension_exposes_both_nodes():
    import asyncio

    extension = asyncio.run(pack.comfy_entrypoint())
    found = asyncio.run(extension.get_node_list())
    ids = sorted(n.define_schema().node_id for n in found)
    assert ids == ["vgl.ParseVGL", "vgl.ThalesPrompt"], ids


if __name__ == "__main__":
    raise SystemExit(_harness.run(dict(globals())))
