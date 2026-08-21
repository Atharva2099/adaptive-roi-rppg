"""Explicit injectable extraction contract for Gate 9.

The production runner accepts a callable, never a precomputed per-hop file.
An implementation may use MediaPipe/MAT or a test double, but it must return
primary rows plus independent Oracle B/C rows before publication can begin.
"""
from __future__ import annotations
import importlib
from typing import Any, Callable, Mapping
from adaptive_roi_rppg.contracts.errors import ContractValidationError

def load_extractor(spec: str) -> Callable[..., Mapping[str, Any]]:
    if ":" not in spec: raise ContractValidationError("Gate 9 extractor must be module:function")
    module_name, function_name = spec.split(":", 1)
    try: function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc: raise ContractValidationError("Gate 9 extractor cannot be loaded") from exc
    if not callable(function): raise ContractValidationError("Gate 9 extractor is not callable")
    return function

def execute_extractor(extractor: Callable[..., Mapping[str, Any]], plan: Any, provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    try: result = extractor(plan=plan, provenance=dict(provenance))
    except TypeError as exc: raise ContractValidationError("Gate 9 extractor must accept plan= and provenance=") from exc
    if not isinstance(result, Mapping) or set(result) != {"per_hop", "oracle_b", "oracle_c"}: raise ContractValidationError("Gate 9 extractor must return per_hop, oracle_b, and oracle_c")
    for name in ("per_hop", "oracle_b", "oracle_c"):
        if not isinstance(result[name], (bytes, str)): raise ContractValidationError(f"Gate 9 extractor output {name} must be CSV bytes or text")
    return result
