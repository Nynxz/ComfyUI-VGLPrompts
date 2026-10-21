"""The pack's two nodes: render a Thales payload's prompts, and read a VGL back out of a reply.

Neither calls an LLM. `VGL Prompt (Thales)` produces what `bria_vgl` would send -- the system
prompt, the final (user) prompt, the images as sent, the response schema and temperature -- for
whichever LLM node the workflow uses. `Parse VGL` cleans that node's reply the way
`Payload.parse` does, so it can be chained into the next payload or a Fibo text encoder.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image

from . import thales
from .pack import PackNode

#: Images are downscaled to about this many pixels before they're sent (bria_vgl MAX_PIXELS).
MAX_PIXELS = 262144

_PROMPT_TOOLTIP = "The request: what to generate, or for edits the change to make. Edits refer to their images as <image_1> to <image_4>."
_VGL_TOOLTIP = "VGL JSON, e.g. from Parse VGL. Empty values and unknown fields are dropped first, as bria_vgl does."


def _prompt():
    return io.String.Input("prompt", multiline=True, tooltip=_PROMPT_TOOLTIP)


def _input_image_vgl():
    return io.String.Input(
        "input_image_vgl",
        multiline=True,
        optional=True,
        tooltip="Optional VGL of the first image. When given, the edit uses the prompt variant that "
        "treats it as the baseline instead of describing the image from scratch. " + _VGL_TOOLTIP,
    )


def _to_pil(image: torch.Tensor) -> Image.Image:
    """One [H, W, C] ComfyUI image -> RGB PIL."""
    array = (image.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array[..., :3] if array.shape[-1] >= 3 else array[..., 0]).convert("RGB")


def _mask_to_pil(mask: torch.Tensor) -> Image.Image:
    array = (mask.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="L")


def _as_sent(image: Image.Image) -> Image.Image:
    """bria_vgl's to_base64_jpeg, minus the JPEG: RGB, downscaled to MAX_PIXELS keeping the aspect ratio."""
    image = image.convert("RGB")
    if image.size[0] * image.size[1] > MAX_PIXELS:
        ratio = (MAX_PIXELS / (image.size[0] * image.size[1])) ** 0.5
        image = image.resize(
            (int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.Resampling.LANCZOS
        )
    return image


def _apply_gray_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Paint the masked area mid-grey. Burned in before downscaling, so the mask is resized with its image."""
    gray = Image.new("RGB", image.size, (128, 128, 128))
    return Image.composite(gray, image.convert("RGB"), mask.convert("L"))


def _to_batch(images: list[Image.Image]) -> torch.Tensor | None:
    """PIL images -> one IMAGE batch. A batch has one size, so differing images are centred on a
    black canvas of the largest width and height (only multi-image edits can differ)."""
    if not images:
        return None
    w, h = max(i.size[0] for i in images), max(i.size[1] for i in images)
    out = torch.zeros(len(images), h, w, 3)
    for n, image in enumerate(images):
        x, y = (w - image.size[0]) // 2, (h - image.size[1]) // 2
        out[n, y : y + image.size[1], x : x + image.size[0]] = torch.from_numpy(
            np.asarray(image, dtype=np.float32) / 255.0
        )
    return out


def _frames(*images: torch.Tensor | None) -> list[torch.Tensor]:
    """Every frame of every connected IMAGE input, in order: a batch of 3 counts as 3 images."""
    return [batch[i] for batch in images if batch is not None for i in range(batch.shape[0])]


class ThalesPrompt(PackNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="ThalesPrompt",
            display_name="VGL Prompt (Thales)",
            description=(
                "Renders the system prompt and final prompt that Bria VGL's Thales dialect sends to "
                "its LLM for a payload, byte for byte, with the images prepared the same way. Wire "
                "them into any LLM node (system prompt, user prompt, images); its reply is a VGL "
                "JSON caption -- clean it with Parse VGL.\n\n"
                "response_schema is the JSON schema bria_vgl enforces through the LLM's structured "
                "output. Most ComfyUI LLM nodes (Generate Text included) cannot enforce one, so by "
                "default it is also written into the system prompt (schema_in_prompt). Turn that off "
                "when the LLM node enforces response_schema itself, for bria_vgl's exact prompts."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "payload",
                    options=[
                        io.DynamicCombo.Option("generate", [_prompt()]),
                        io.DynamicCombo.Option(
                            "inspire",
                            [
                                _prompt(),
                                io.Image.Input(
                                    "image",
                                    tooltip="A visual reference only: the VGL describes the prompt's scene, not this image.",
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "caption",
                            [
                                io.Image.Input(
                                    "image",
                                    tooltip="The image the VGL describes. Caption takes no prompt.",
                                )
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "refine",
                            [
                                _prompt(),
                                io.String.Input(
                                    "input_vgl",
                                    multiline=True,
                                    tooltip="The VGL to change. " + _VGL_TOOLTIP,
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "edit",
                            [
                                _prompt(),
                                io.Image.Input(
                                    "image_1",
                                    tooltip="<image_1>. A batch counts as several images, in order.",
                                ),
                                io.Image.Input(
                                    "image_2",
                                    optional=True,
                                    tooltip="<image_2>, and so on. 1 to 4 images in all.",
                                ),
                                io.Image.Input("image_3", optional=True),
                                io.Image.Input("image_4", optional=True),
                                _input_image_vgl(),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "edit_with_mask",
                            [
                                _prompt(),
                                io.Image.Input("image"),
                                io.Mask.Input(
                                    "mask",
                                    tooltip="White is the area to change. Must match the image's size.",
                                ),
                                _input_image_vgl(),
                            ],
                        ),
                    ],
                    tooltip=(
                        "generate: a new VGL from a prompt. inspire: from a prompt, with an image as "
                        "visual reference. caption: describe an image. refine: change an existing VGL "
                        "by a prompt, no images. edit: edit 1-4 images. edit_with_mask: edit the masked "
                        "area of one image."
                    ),
                ),
                io.String.Input(
                    "style_preset",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Optional style guidelines, composed into the system prompt as bria_vgl's "
                    "style_preset_prompt: the prompt owns the content, the preset owns the look.",
                ),
                io.Combo.Input(
                    "schema_dialect",
                    options=["json_schema", "gemini"],
                    default="json_schema",
                    advanced=True,
                    tooltip="json_schema: standard JSON Schema, for OpenAI-style structured output. "
                    "gemini: the schema exactly as bria_vgl sends Gemini (upper-case types, nullable).",
                ),
                io.Boolean.Input(
                    "schema_in_prompt",
                    default=True,
                    advanced=True,
                    tooltip="Append the response schema (compact) to the system prompt. bria_vgl never "
                    "lets its LLM stray from the schema -- it enforces it as structured output -- and "
                    "an LLM node that cannot enforce one gets the nearest thing in writing. Turn off "
                    "when wiring response_schema into a node that enforces it: then the prompts are "
                    "bria_vgl's exactly.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt"),
                io.String.Output(
                    display_name="final_prompt",
                    tooltip="The user message text; the images go with it.",
                ),
                io.Image.Output(
                    display_name="images",
                    tooltip="The images as sent: RGB, the mask painted grey, downscaled to about 0.26 MP. "
                    "None for generate and refine. Multi-image edits of different sizes are centred on a "
                    "black canvas, since a batch has one size.",
                ),
                io.String.Output(display_name="response_schema"),
                io.Float.Output(
                    display_name="temperature", tooltip="The temperature bria_vgl uses (0.2)."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        payload: dict,
        style_preset: str = "",
        schema_dialect: str = "json_schema",
        schema_in_prompt: bool = True,
    ) -> io.NodeOutput:
        name = payload["payload"]
        spec = thales.PAYLOADS[name]
        if name == "edit":
            frames = _frames(
                *(payload.get(f"image_{i}") for i in range(1, thales.MAX_EDIT_IMAGES + 1))
            )
        else:
            frames = _frames(payload.get("image"))
        images = [_to_pil(f) for f in frames]
        if spec.mask and images:
            mask = _mask_to_pil(payload["mask"].reshape(-1, *payload["mask"].shape[-2:])[0])
            if mask.size != images[0].size:
                raise ValueError(
                    f"mask is {mask.size} but image is {images[0].size}; they must match"
                )
            images = [_apply_gray_mask(images[0], mask)]

        prompts = thales.build(
            name,
            payload.get("prompt") or "",
            images=len(images),
            input_vgl=payload.get("input_vgl") or "",
            input_image_vgl=payload.get("input_image_vgl") or "",
            style_preset=style_preset or "",
        )
        schema_dict = thales.response_schema(prompts.schema, schema_dialect)
        schema = json.dumps(schema_dict, indent=1)
        system_prompt = prompts.system_prompt
        if schema_in_prompt:
            # Compact in the prompt: ~2.5k characters instead of ~3.9k, and no easier to misread.
            compact = json.dumps(schema_dict, separators=(",", ":"))
            system_prompt = (
                f"{system_prompt.rstrip()}\n\n{thales.SCHEMA_IN_PROMPT.format(schema=compact)}"
            )
        return io.NodeOutput(
            system_prompt,
            prompts.final_prompt,
            _to_batch([_as_sent(i) for i in images]),
            schema,
            prompts.temperature,
        )


class ParseVGL(PackNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="ParseVGL",
            display_name="Parse VGL",
            description=(
                "Cleans an LLM's VGL reply the way bria_vgl's Payload.parse does: strips a Markdown "
                "code fence, drops empty values and fields the Thales VGL does not have, and orders "
                "the fields like the model. The result chains into another VGL Prompt (input_vgl / "
                "input_image_vgl) or straight into Fibo's text encoder as its prompt."
            ),
            inputs=[
                io.String.Input(
                    "text", multiline=True, force_input=True, tooltip="The LLM node's reply."
                )
            ],
            outputs=[
                io.String.Output(
                    display_name="vgl",
                    tooltip="Compact VGL JSON, as bria_vgl puts it into a prompt.",
                ),
                io.String.Output(display_name="short_description"),
                io.String.Output(
                    display_name="edit_instruction",
                    tooltip="Empty unless the reply came from an edit.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        vgl = thales.clean_vgl(text)
        return io.NodeOutput(
            json.dumps(vgl), vgl.get("short_description", ""), vgl.get("edit_instruction", "")
        )
