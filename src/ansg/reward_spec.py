"""Machine-checkable reward / RL specifications of the dissertation (torch-free).

``configs/specification/reward_rl_variants.yaml`` transcribes every reward and
PPO statement of the thesis as a separate *variant* with section, equation and
page. This module loads and validates it and evaluates a variant's arithmetic
for **caller-supplied** component values. It deliberately:

* selects no variant (``selected_variant`` must be chosen explicitly by the
  author; :func:`load_reward_spec` exposes it, nothing defaults it);
* computes no component. CMPI, virtue coherence, coreference, PCI and the other
  sub-metrics have no operational definition in the thesis; their status is
  recorded and :meth:`RewardSpec.evaluate` refuses them unless the caller opts
  in with ``allow_undefined=True`` (used to re-check the printed worked
  examples). Results are always marked ``reproduces_thesis_result: False``.

``ansg.training.cultural_reward`` remains the existing top-level aggregator; its
default weights coincide with the top level of Eq. 4.29 / Eq. 5.1 only.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "configs" / "specification" / "reward_rl_variants.yaml"
TRAINING_PY = Path(__file__).resolve().parent / "training.py"

COMPONENT_STATUSES = ("undefined", "underspecified", "normalisation_unspecified", "standard_metric")
VARIANT_KINDS = ("full", "fragment", "ratio")
TRANSFORMS = {None: lambda x: x, "one_minus": lambda x: 1.0 - x}


class RewardSpecError(ValueError):
    """The specification file is structurally invalid."""


class UndefinedComponentError(RuntimeError):
    """A variant needs components the thesis does not define."""


@dataclass(frozen=True)
class Term:
    name: str
    weight: float
    transform: str | None = None


def _terms(node: Mapping[str, Any]) -> list[Term]:
    out = []
    for name, spec in (node.get("sum") or {}).items():
        if isinstance(spec, Mapping):
            out.append(Term(name, float(spec["weight"]), spec.get("transform")))
        else:
            out.append(Term(name, float(spec)))
    return out


class RewardSpec:
    def __init__(self, data: Mapping[str, Any], path: Path | None = None):
        self.data = dict(data)
        self.path = path

    # ------------------------------------------------------------ accessors
    @property
    def components(self) -> dict[str, dict[str, Any]]:
        return dict(self.data.get("components") or {})

    @property
    def variants(self) -> dict[str, dict[str, Any]]:
        return dict(self.data.get("variants") or {})

    @property
    def ppo_variants(self) -> dict[str, dict[str, Any]]:
        return dict(self.data.get("ppo_variants") or {})

    @property
    def selected_variant(self) -> str | None:
        return self.data.get("selected_variant")

    def variant(self, variant_id: str) -> dict[str, Any]:
        if variant_id not in self.variants:
            raise KeyError(f"unknown reward variant {variant_id!r}; known: {sorted(self.variants)}")
        return self.variants[variant_id]

    def nodes(self, variant_id: str) -> dict[str, list[Term]]:
        return {name: _terms(node) for name, node in self.variant(variant_id)["formula"].items()}

    def leaf_components(self, variant_id: str) -> list[str]:
        v = self.variant(variant_id)
        nodes = self.nodes(variant_id)
        leaves = {t.name for terms in nodes.values() for t in terms if t.name not in nodes}
        leaves |= {p["component"] for p in v.get("penalties") or []}
        return sorted(leaves)

    def blockers(self, variant_id: str) -> list[dict[str, Any]]:
        """Components of a variant that are not defined well enough to compute."""
        out = []
        for name in self.leaf_components(variant_id):
            comp = self.components.get(name, {})
            if comp.get("status") != "standard_metric":
                out.append({"component": name, "status": comp.get("status", "unknown"),
                            "reference": comp.get("reference")})
        v = self.variant(variant_id)
        if (v.get("kl_penalty") or {}).get("placement") == "unspecified":
            out.append({"component": "kl_penalty", "status": "underspecified",
                        "reference": f"coefficient {v['kl_penalty']['coefficient']} p.{v['kl_penalty']['page']}; placement unstated"})
        for name, node in v["formula"].items():
            if node.get("normalised") is False:
                out.append({"component": name, "status": "range_unnormalised",
                            "reference": "printed as an unweighted sum; its range exceeds [0,1]"})
        return out

    # ------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        problems: list[str] = []
        comps = self.components
        for name, c in comps.items():
            if c.get("status") not in COMPONENT_STATUSES:
                problems.append(f"component {name!r}: status {c.get('status')!r} not in {COMPONENT_STATUSES}")
            if not c.get("reference"):
                problems.append(f"component {name!r}: missing reference")
        sel = self.selected_variant
        if sel is not None and sel not in self.variants:
            problems.append(f"selected_variant {sel!r} is not a variant")
        sel_ppo = self.data.get("selected_ppo_variant")
        if sel_ppo is not None and sel_ppo not in self.ppo_variants:
            problems.append(f"selected_ppo_variant {sel_ppo!r} is not a PPO variant")
        for vid, v in self.variants.items():
            kind = v.get("kind")
            if kind not in VARIANT_KINDS:
                problems.append(f"{vid}: kind {kind!r} not in {VARIANT_KINDS}")
            ref = v.get("reference") or {}
            if not ref.get("section") or not ref.get("pages") or not all(isinstance(p, int) for p in ref["pages"]):
                problems.append(f"{vid}: reference needs a section and integer pages")
            formula = v.get("formula") or {}
            if not formula:
                problems.append(f"{vid}: empty formula")
                continue
            if kind == "full" and v.get("output") not in formula:
                problems.append(f"{vid}: output {v.get('output')!r} is not a formula node")
            if kind == "fragment" and set(v.get("defines") or []) != set(formula):
                problems.append(f"{vid}: 'defines' must list exactly the formula nodes")
            for node_name, node in formula.items():
                try:
                    terms = _terms(node)
                except (KeyError, TypeError, ValueError) as exc:
                    problems.append(f"{vid}.{node_name}: malformed term ({exc})")
                    continue
                if not terms:
                    problems.append(f"{vid}.{node_name}: no terms")
                for t in terms:
                    if t.name not in formula and t.name not in comps:
                        problems.append(f"{vid}.{node_name}: term {t.name!r} is neither a node nor a component")
                    if not (t.weight > 0):
                        problems.append(f"{vid}.{node_name}: weight of {t.name!r} must be positive")
                    if t.transform not in TRANSFORMS:
                        problems.append(f"{vid}.{node_name}: unknown transform {t.transform!r}")
                if node.get("normalised", True) and terms and not math.isclose(sum(t.weight for t in terms), 1.0, abs_tol=1e-9):
                    problems.append(f"{vid}.{node_name}: weights sum to {sum(t.weight for t in terms):g}, not 1")
            if self._has_cycle(formula):
                problems.append(f"{vid}: formula nodes form a cycle")
            for p in v.get("penalties") or []:
                if p.get("applies_to") not in formula:
                    problems.append(f"{vid}: penalty applies to unknown node {p.get('applies_to')!r}")
                if p.get("component") not in comps:
                    problems.append(f"{vid}: penalty component {p.get('component')!r} not declared")
        for ex in self.data.get("worked_examples") or []:
            if ex.get("variant") not in self.variants:
                problems.append(f"worked example {ex.get('id')}: unknown variant {ex.get('variant')!r}")
                continue
            missing = [c for c in self.leaf_components(ex["variant"]) if c not in (ex.get("inputs") or {})]
            if missing:
                problems.append(f"worked example {ex.get('id')}: inputs missing {missing}")
        for pid, p in self.ppo_variants.items():
            ref = p.get("reference") or {}
            if not ref.get("section") or not ref.get("pages"):
                problems.append(f"ppo variant {pid}: reference needs a section and pages")
        return problems

    @staticmethod
    def _has_cycle(formula: Mapping[str, Any]) -> bool:
        graph = {n: [t.name for t in _terms(node) if t.name in formula] for n, node in formula.items()}
        state: dict[str, int] = {}

        def visit(n: str) -> bool:
            if state.get(n) == 1:
                return True
            if state.get(n) == 2:
                return False
            state[n] = 1
            if any(visit(m) for m in graph[n]):
                return True
            state[n] = 2
            return False

        return any(visit(n) for n in graph)

    # ------------------------------------------------------------ arithmetic
    def evaluate(self, variant_id: str, values: Mapping[str, float], *, allow_undefined: bool = False,
                 node: str | None = None) -> dict[str, Any]:
        """Evaluate a variant's formula for caller-supplied component values.

        Raises :class:`UndefinedComponentError` if the variant needs components
        the thesis does not define, unless ``allow_undefined`` is set (then the
        result lists them under ``caller_supplied_undefined``)."""
        v = self.variant(variant_id)
        blockers = [b for b in self.blockers(variant_id) if b["status"] not in ("range_unnormalised",)]
        if blockers and not allow_undefined:
            raise UndefinedComponentError(
                f"variant {variant_id!r} needs components without an operational definition: "
                + ", ".join(f"{b['component']} ({b['status']})" for b in blockers))
        missing = [c for c in self.leaf_components(variant_id) if c not in values]
        if missing:
            raise KeyError(f"missing component values: {missing}")
        nodes = self.nodes(variant_id)
        penalties: dict[str, list[dict[str, Any]]] = {}
        for p in v.get("penalties") or []:
            penalties.setdefault(p["applies_to"], []).append(p)
        cache: dict[str, float] = {}

        def value(name: str) -> float:
            if name not in nodes:
                return float(values[name])
            if name not in cache:
                total = sum(t.weight * TRANSFORMS[t.transform](value(t.name)) for t in nodes[name])
                for p in penalties.get(name, []):
                    total -= float(p["coefficient"]) * float(values[p["component"]])
                cache[name] = total
            return cache[name]

        target = node or v.get("output") or next(iter(nodes))
        result = value(target)
        for n in nodes:
            value(n)
        return {"variant": variant_id, "node": target, "value": result, "nodes": dict(cache),
                "caller_supplied_undefined": [b["component"] for b in blockers],
                "reproduces_thesis_result": False}

    def worked_examples(self) -> list[dict[str, Any]]:
        """Recompute each printed worked example from its printed inputs."""
        out = []
        for ex in self.data.get("worked_examples") or []:
            res = self.evaluate(ex["variant"], ex["inputs"], allow_undefined=True)
            rows = {}
            for name, printed in (ex.get("printed") or {}).items():
                recomputed = res["nodes"][name]
                rows[name] = {"printed": printed, "recomputed": round(recomputed, 6),
                              "difference": round(recomputed - printed, 6),
                              "consistent_to_printed_precision": abs(recomputed - printed) <= 0.0005 + 1e-12}
            if ex.get("printed_total_uses_printed_partials"):
                v = self.variant(ex["variant"])
                out_node = v["output"]
                partial_inputs = {**ex["inputs"], **{k: p for k, p in ex["printed"].items() if k != out_node}}
                top = sum(t.weight * partial_inputs[t.name] for t in self.nodes(ex["variant"])[out_node])
                rows[f"{out_node}_from_printed_partials"] = {
                    "printed": ex["printed"][out_node], "recomputed": round(top, 6),
                    "difference": round(top - ex["printed"][out_node], 6),
                    "consistent_to_printed_precision": abs(top - ex["printed"][out_node]) <= 0.0005 + 1e-12}
            out.append({"id": ex["id"], "variant": ex["variant"], "reference": ex.get("reference"), "rows": rows})
        return out

    def same_weights_different_terms(self) -> list[dict[str, Any]]:
        """Weight vectors printed for different term sets (C-REWARD-2 pattern)."""
        seen: dict[tuple[float, ...], list[tuple[str, str, tuple[str, ...]]]] = {}
        for vid, v in self.variants.items():
            for node_name, node in v["formula"].items():
                terms = _terms(node)
                if len(terms) < 3:
                    continue
                key = tuple(sorted((round(t.weight, 6) for t in terms), reverse=True))
                names = tuple(t.name for t in sorted(terms, key=lambda t: -t.weight))
                seen.setdefault(key, []).append((vid, node_name, names))
        out = []
        for key, uses in seen.items():
            if len({u[2] for u in uses}) > 1:
                out.append({"weights": list(key), "uses": [{"variant": u[0], "node": u[1], "terms_by_weight": list(u[2])} for u in uses]})
        return out

    def ppo_update_kwargs(self, ppo_variant_id: str) -> dict[str, Any]:
        """Arguments for ``ansg.training.ppo_update`` from ONE explicitly named PPO
        variant, plus the settings that function does not apply (learning rate,
        KL, GAE, batch sizes: the caller's training loop must handle them)."""
        return _ppo_update_kwargs(self, ppo_variant_id)

    # --------------------------------------------------- code vs thesis
    def code_alignment(self, training_py: Path = TRAINING_PY) -> dict[str, Any]:
        """Compare defaults in ``ansg.training`` with the PPO/reward variants (read with ast)."""
        defaults = _function_defaults(training_py)
        reward_w = defaults.get("cultural_reward", {}).get("weights")
        top_level = {vid: [t.weight for t in self.nodes(vid)[v["output"]]]
                     for vid, v in self.variants.items() if v.get("kind") == "full"}
        reward_rows = {vid: (list(reward_w) == w if reward_w is not None else None) for vid, w in top_level.items()}
        # equal weight vectors only; the terms they weight differ between variants
        ppo = defaults.get("ppo_update", {})
        code_map = {"clip_range": ppo.get("clip_eps"), "value_coef": ppo.get("value_coef"),
                    "max_grad_norm": ppo.get("max_grad_norm"), "entropy_coef": ppo.get("entropy_coef")}
        ppo_rows = {}
        for pid, p in self.ppo_variants.items():
            rows = {}
            for key, code_value in code_map.items():
                if key in p:
                    rows[key] = {"thesis": p[key], "code_default": code_value,
                                 "status": "absent_in_code" if code_value is None else
                                 ("match" if math.isclose(float(p[key]), float(code_value)) else "code_default_differs_selectable_by_argument")}
            for key in ("kl_coef", "gae_lambda", "learning_rate", "ppo_epochs", "minibatch", "rollout_batch", "steps"):
                if key in p:
                    rows[key] = {"thesis": p[key], "code_default": None, "status": "caller_or_not_implemented"}
            ppo_rows[pid] = rows
        return {"cultural_reward_default_weights": list(reward_w) if reward_w is not None else None,
                "cultural_reward_default_weights_equal_top_level_weights_of": reward_rows,
                "ppo_update_defaults": ppo, "ppo_variants": ppo_rows}


PPO_KWARGS = {"clip_range": "clip_eps", "value_coef": "value_coef", "max_grad_norm": "max_grad_norm",
              "entropy_coef": "entropy_coef"}


def _ppo_update_kwargs(spec: "RewardSpec", ppo_variant_id: str) -> dict[str, Any]:
    if ppo_variant_id not in spec.ppo_variants:
        raise KeyError(f"unknown PPO variant {ppo_variant_id!r}; known: {sorted(spec.ppo_variants)}")
    p = spec.ppo_variants[ppo_variant_id]
    kwargs = {arg: p[key] for key, arg in PPO_KWARGS.items() if key in p}
    not_mapped = {k: v for k, v in p.items() if k not in PPO_KWARGS and k != "reference"}
    return {"ppo_variant": ppo_variant_id, "kwargs": kwargs, "not_applied_by_ppo_update": not_mapped}


def _function_defaults(path: Path) -> dict[str, dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            args = fn.args.args
            pairs = zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults)
            vals = {}
            for a, d in list(pairs) + list(zip(fn.args.kwonlyargs, fn.args.kw_defaults)):
                if d is None:
                    continue
                try:
                    vals[a.arg] = ast.literal_eval(d)
                except ValueError:
                    continue
            out[fn.name] = vals
    return out


def load_reward_spec(path: str | Path | None = None) -> RewardSpec:
    p = Path(path) if path else DEFAULT_SPEC
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or data.get("schema") != "ansg.reward_rl_variants/1":
        raise RewardSpecError(f"{p}: not an ansg.reward_rl_variants/1 file")
    return RewardSpec(data, p)
