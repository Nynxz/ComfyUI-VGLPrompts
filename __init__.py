"""ComfyUI-VGLPrompts — Bria VGL's Thales prompts for any LLM node in ComfyUI.

`bria_vgl` turns a request (a prompt, images, a VGL to change) into one LLM call that returns a
VGL: the structured JSON caption Bria's FIBO models are conditioned on. This pack does only the
prompt half: it renders the exact system and final prompt `bria_vgl` would send, and prepares the
images the same way, so any LLM node (Gemini, OpenAI, Claude, a local VLM) can make the call.

    pack.json   the pack's identity: namespace, display name, menu category.
    pack.py     that identity and the node base class.
    thales.py   payload routing, prompt rendering, VGL cleaning -- no ComfyUI imports.
    thales/     the prompt, field-guide and output-schema text, vendored from Bria-AI/vgl.
    nodes.py    the two nodes.
"""

from comfy_api.latest import ComfyExtension, io

from .nodes import ParseVGL, ThalesPrompt


class VGLPromptsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ThalesPrompt, ParseVGL]


async def comfy_entrypoint() -> ComfyExtension:
    return VGLPromptsExtension()


__all__ = ["VGLPromptsExtension", "comfy_entrypoint"]
