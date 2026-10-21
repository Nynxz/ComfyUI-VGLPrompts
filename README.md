# ComfyUI-VGLPrompts

Bria VGL's **Thales** prompts for any LLM node in ComfyUI.

Bria's FIBO models work best when the prompt isn't a sentence but a detailed JSON description of
the picture — objects, lighting, camera, style. That JSON is a **VGL**. Writing one by hand is
tedious, so an LLM writes it, and [`bria_vgl`](https://github.com/Bria-AI/vgl) holds the
instructions Bria gives that LLM. This pack renders those exact instructions inside ComfyUI, so
whatever LLM node you use — Generate Text with a local VLM, Gemini, OpenAI, Claude — does the
writing.

```
VGL Prompt (Thales) ──► LLM node ──► Parse VGL ──► TextEncodeFibo2 ──► sampler
  the instructions      writes JSON   tidies it     FIBO reads it
```

## Quick start with Generate Text

1. Add **VGL Prompt (Thales)**, pick `generate`, type your idea: `a red fox in fresh snow at dawn`.
2. Add **Generate Text** (with a vision-language text model as its `clip`) and wire:

   | VGL Prompt (Thales) | → Generate Text |
   |---|---|
   | `system_prompt` | `system_prompt` — keep `use_default_template` **on**, or it is ignored |
   | `final_prompt` | `prompt` |
   | `images` | `image` (only for payloads that take an image) |

   Then on Generate Text set **`max_length` ≈ 4096** (a VGL is long; the default 512 cuts the JSON
   off) and **`sampling_mode` on, temperature 0.2** (what Bria uses).
3. Add **Parse VGL**: `generated_text` → `text`.
4. Wire Parse VGL's `vgl` → `TextEncodeFibo2`'s `prompt`, and sample as usual.

`response_schema` and `temperature` stay unwired with Generate Text — it has nowhere to take them.
See [The output schema](#the-output-schema).

## Payloads

| payload | you give it | the VGL describes |
|---|---|---|
| **generate** | a prompt | a new scene, invented from the prompt |
| **inspire** | a prompt + a reference image | a new scene; the prompt says what to borrow from the image |
| **caption** | an image | that image, as it is |
| **refine** | a prompt + a VGL | that VGL, changed as the prompt says (no images) |
| **edit** | a prompt + 1–4 images | the image(s) with the change applied |
| **edit_with_mask** | a prompt + an image + a mask | the image with the white mask area changed |

**inspire** — write the prompt as *what to keep and what to change*: `the same dog and pose, but at
the beach`, `a cat in the same pose and lighting`, `this dog as a knight, oil painting`. The LLM
studies the reference's subject, pose, composition, lighting and medium, and keeps what you name.
A photo reference stays a photograph unless the prompt asks for another style. Use **edit** when
the result must line up with your actual image.

**edit** — refer to the images as `<image_1>` … `<image_4>`. A batch on `image_1` counts as several
images, in order. For a more faithful edit, caption the image first and feed that VGL into
`input_image_vgl`: the LLM then starts from an accurate description of it.

**style_preset** (optional, every payload) — look-and-feel guidelines, e.g. `moody film noir, high
contrast, 35mm grain`. The prompt decides what is in the picture; the preset decides how it looks.

## Nodes

**VGL Prompt (Thales)** — outputs `system_prompt`, `final_prompt`, `images` (as the LLM should see
them: RGB, the mask painted mid-grey, downscaled to about 0.26 MP — exactly as `bria_vgl` sends
them), `response_schema`, `temperature` (0.2).

**Parse VGL** — the LLM's reply is rarely clean: a ```` ```json ```` fence around it, empty fields,
fields that don't exist. `TextEncodeFibo2` silently treats anything that isn't valid JSON as plain
text, so a messy reply loses its structure without an error. Parse VGL cleans it the way
`bria_vgl`'s `Payload.parse` does — strips the fence, drops empty and unknown fields, orders the
fields like the model — and errors loudly if the reply isn't JSON at all. Outputs:

* `vgl` — the clean JSON: into `TextEncodeFibo2`, or into another VGL Prompt's `input_vgl` /
  `input_image_vgl` to chain (caption → edit, generate → refine)
* `short_description`, `edit_instruction` — handy to preview

## The output schema

`bria_vgl` makes its LLM follow a JSON schema through the API's *structured output*, so the reply
can't stray. Most ComfyUI LLM nodes — Generate Text included — can't enforce a schema, so:

* **Default — `schema_in_prompt` on:** the schema is also written, compact (~700 tokens), at the end
  of the system prompt: the nearest thing to upstream's guarantee. `schema_dialect` defaults to
  standard JSON Schema.
* **Your LLM node enforces a JSON schema:** wire `response_schema` into it and turn
  `schema_in_prompt` off. The prompts are then upstream's byte for byte; `schema_dialect: gemini`
  gives the exact bytes Bria sends Gemini.

## Fidelity

With `schema_in_prompt` off, every payload renders byte-for-byte the system and final prompt
`bria_vgl`'s own renderer produces, with and without a style preset; Parse VGL and the JSON Schema
conversion match upstream exactly too. `tools/verify_upstream.py` checks all of it.

## Development

`thales/` is vendored from [Bria-AI/vgl](https://github.com/Bria-AI/vgl) (MIT,
`thales/LICENSE.bria-vgl`) at the commit in `thales/UPSTREAM.json` — the prompt text only, so
`bria-vgl` and its litellm/openai dependencies are not needed.

```bash
python tools/sync_thales.py [ref]                      # refresh thales/ from upstream (default: dev)
python tools/verify_upstream.py /path/to/vgl           # byte-for-byte parity with a vgl checkout
COMFY_ROOT=/path/to/ComfyUI python tests/test_thales.py
```

A new payload or placeholder upstream needs a matching entry in `thales.py`; `verify_upstream.py`
is how you find out.
