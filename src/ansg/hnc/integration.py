"""Connecting HNC to the ANSG generation runtime without changing it.

* ``hnc_enabled = false`` -> :func:`generate` calls ``ansg.generation.runtime.run_generation``
  with the caller's request **unchanged** and only adds ``{"hnc": {"enabled": false}}``
  to the returned record (tested to be otherwise identical, token for token).
* ``hnc_enabled = true``  -> :func:`runtime_generate_fn` turns one loaded backbone (plus
  optional cultural modules, i.e. CSAM/DAL with their own unchanged equations) into a
  ``generate_fn`` for :func:`ansg.hnc.realize.realise`. HNC never builds, wraps or
  updates model parameters; it only decides *what* to ask for, segment by segment.

Requires the ``model`` and ``pretrained`` extras only when a model is actually used.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .realize import HNCConfig


def generate(request, hnc: HNCConfig) -> dict:
    """The original single-prompt path, with an explicit HNC flag in the record."""
    from ..generation.runtime import run_generation
    if hnc.hnc_enabled:
        raise ValueError("generate() is the HNC-disabled path; use realise() with runtime_generate_fn for HNC")
    record = run_generation(request)
    record["hnc"] = {"enabled": False, "variant": "A"}
    return record


def runtime_generate_fn(request, records: list[dict] | None = None):
    """``generate_fn`` that runs one ``run_generation`` per segment (seed + segment index).

    Each segment's full generation record (hashes, label, software) is appended to ``records``.
    """
    from ..generation.runtime import run_generation

    def fn(prompt: str, segment_index: int) -> str:
        seg_req = replace(request, prompt=prompt,
                          sampling=replace(request.sampling, seed=request.sampling.seed + segment_index, num_samples=1))
        rec: dict[str, Any] = run_generation(seg_req)
        if records is not None:
            records.append(rec)
        return rec["samples"][0]["continuation"]

    return fn
