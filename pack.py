"""Pack identity and the node base class.

Everything that carries the pack's name comes from `pack.json`, so renaming the pack is a
one-file edit:

    pack.json { "namespace": "vgl" }
      ->  node id        vgl.ThalesPrompt   (node_id / comfyClass)
"""

from __future__ import annotations

import json
import os

from comfy_api.latest import io

_ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_ROOT, "pack.json"), encoding="utf-8") as _fh:
    _PACK: dict = json.load(_fh)

#: Node-id prefix: "vgl" -> node id "vgl.ThalesPrompt".
NAMESPACE: str = _PACK["namespace"]
#: Root ComfyUI menu category.
CATEGORY: str = _PACK["category"]


class PackNode(io.ComfyNode):
    """Base for this pack's nodes, so neither repeats the namespace or the menu category."""

    CATEGORY = CATEGORY

    @classmethod
    def make_schema(cls, node_id: str, display_name: str, **kwargs) -> io.Schema:
        return io.Schema(
            node_id=f"{NAMESPACE}.{node_id}",
            display_name=display_name,
            category=cls.CATEGORY,
            **kwargs,
        )
