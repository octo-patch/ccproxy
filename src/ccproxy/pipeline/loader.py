"""Dynamic hook loading from config entries.

Imports hook modules by dotted path (triggering @hook registration),
then filters the global registry by the entries the caller declared.
Validates YAML-supplied params against each hook's declared Pydantic
model (if any) and drops params for hooks that declare no model.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import replace
from typing import Any, cast

from pydantic import ValidationError

from ccproxy.pipeline.hook import HookParams, HookSpec, RegisteredHandlerFn, get_registry

logger = logging.getLogger(__name__)


def load_hooks(entries: list[str | dict[str, Any]]) -> list[HookSpec]:
    """Resolve a config hook-list into a list of HookSpec objects.

    Each entry is either a dotted module path string (the hook fn's
    module) or a dict ``{"hook": "<module_path>", "params": {...}}``.

    Side effects:
    - Imports each module, triggering @hook registration.
    - Returns per-load HookSpec copies with ``params`` and ``priority`` resolved
      from the given config entries.
    """
    hook_priority_map: dict[str, int] = {}
    hook_params_map: dict[str, HookParams] = {}

    for idx, entry in enumerate(entries):
        params: HookParams = {}
        if isinstance(entry, str):
            module_path = entry
        else:
            module_path = str(entry.get("hook", ""))
            raw_params = entry.get("params", {})
            params = raw_params if isinstance(raw_params, dict) else {}
            if not module_path:
                continue

        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            logger.error("Failed to import hook module: %s", module_path)
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if callable(obj) and hasattr(obj, "_hook_spec"):
                hook_fn = cast(RegisteredHandlerFn, obj)
                hook_name = hook_fn._hook_spec.name
                hook_priority_map[hook_name] = idx
                if params:
                    hook_params_map[hook_name] = params

    all_specs = get_registry().get_all_specs()
    hook_specs: list[HookSpec] = []
    max_priority = len(entries)

    for name, spec in all_specs.items():
        if name not in hook_priority_map:
            continue
        params = hook_params_map.get(name, {})
        resolved_params: HookParams = {}
        if params and spec.model is not None:
            try:
                validated = spec.model(**params)
            except ValidationError as exc:
                raise ValueError(f"Hook {spec.name!r} params failed validation: {exc}") from exc
            resolved_params = validated.model_dump()
        elif params and spec.model is None:
            logger.warning(
                "Hook %r received YAML params but declares no model=; ignoring",
                name,
            )
        hook_specs.append(
            replace(
                spec,
                params=resolved_params,
                priority=hook_priority_map.get(name, max_priority),
            )
        )

    return hook_specs
