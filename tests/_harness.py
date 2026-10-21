"""Load the pack against a real ComfyUI checkout, and a tiny runner so tests need no pytest.

    COMFY_ROOT=/path/to/ComfyUI python tests/test_graft.py

The pack has to be imported as a *package* under a stable name, because its modules use relative
imports (`from .pack import PackNode`). Loading `nodes.py` as a loose file cannot resolve those,
which is why this goes through `spec_from_file_location` with `submodule_search_locations`
rather than just adding the folder to `sys.path`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback

PACK_NAME = "vglprompts_pack"
PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    """Import the pack and return the module. Idempotent."""
    if PACK_NAME in sys.modules:
        return sys.modules[PACK_NAME]

    comfy = os.environ.get("COMFY_ROOT")
    if not comfy:
        raise SystemExit("set COMFY_ROOT to a ComfyUI checkout, e.g. dev_environment/ComfyUI")
    sys.path.insert(0, comfy)
    # ComfyUI reads argv at import time; --cpu keeps these tests off the GPU.
    sys.argv = [sys.argv[0], "--cpu"]

    spec = importlib.util.spec_from_file_location(
        PACK_NAME,
        os.path.join(PACK_ROOT, "__init__.py"),
        submodule_search_locations=[PACK_ROOT],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACK_NAME] = module
    spec.loader.exec_module(module)
    return module


def run(namespace: dict) -> int:
    """Run every `test_*` callable in `namespace`. Returns a process exit code."""
    tests = sorted(
        (name, fn) for name, fn in namespace.items() if name.startswith("test_") and callable(fn)
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
        # A test runner catching everything is the point: one broken test must not stop the
        # rest from reporting.
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0
