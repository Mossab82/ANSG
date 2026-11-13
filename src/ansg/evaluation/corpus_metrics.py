"""Cultural Reference Density (CRD) — dissertation Eq. (3.1), Ch. 3 §3.6.1.

    CRD = ( Σ_i 1[token_i ∈ M_cultural] / Total tokens ) × 1000

``M_cultural`` is the 2,847-entry cultural-marker lexicon (religious
invocations, proverbial expressions, archetypal references, moral virtue
terms). **The lexicon is not supplied with the dissertation**; callers must
provide it.

Under-specification (SPECIFICATION_CONFLICTS C-METRIC-CRD):
* Eq. (3.1) tests *token* membership, yet many listed markers are multi-word
  (e.g. "in shāʾ Allāh", proverbs). How multi-word markers are counted is not
  stated. This implementation therefore requires an explicit ``policy``:
  - ``"token_membership"`` — the literal Eq. (3.1): a token counts if it is,
    by itself, a lexicon entry; multi-word entries never match.
  - ``"marker_occurrences"`` — each non-overlapping occurrence of any entry
    (longest match first) counts once.
  - ``"tokens_covered"`` — every token inside a matched entry counts.
* The tokenizer is unspecified ("standard Arabic tokenizer", Table 3.1); it is
  an argument. Matching uses the same tokenizer on lexicon entries.
"""

from __future__ import annotations

from typing import Callable, Iterable, Literal

from ..data.normalization import simple_tokenize

Policy = Literal["token_membership", "marker_occurrences", "tokens_covered"]


def _default_tokenizer(text: str) -> list[str]:
    return [t for _, _, t in simple_tokenize(text)]


def cultural_reference_density(
    text: str,
    lexicon: Iterable[str],
    policy: Policy,
    tokenizer: Callable[[str], list[str]] = _default_tokenizer,
) -> float:
    tokens = tokenizer(text)
    if not tokens:
        raise ValueError("text has no tokens")
    entries = {tuple(tokenizer(e)) for e in lexicon}
    entries.discard(())
    if policy == "token_membership":
        singles = {e[0] for e in entries if len(e) == 1}
        hits = sum(t in singles for t in tokens)
    elif policy in ("marker_occurrences", "tokens_covered"):
        max_len = max((len(e) for e in entries), default=1)
        hits, i = 0, 0
        while i < len(tokens):
            for n in range(min(max_len, len(tokens) - i), 0, -1):
                if tuple(tokens[i:i + n]) in entries:
                    hits += 1 if policy == "marker_occurrences" else n
                    i += n
                    break
            else:
                i += 1
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return hits / len(tokens) * 1000.0
