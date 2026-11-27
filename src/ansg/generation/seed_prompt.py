"""Build story prompts conditioned on the regional cultural seed files.

The seed files in ``data/cultural_seeds`` list values, proverbs, dialect
markers, stock characters, settings and traditional openings and closings for
each region. This module samples a few of them into an instruction for an
instruct model, and returns the sampled items with the prompt, so that every
generated story can be traced back to the cultural elements it was asked to use.

Pure Python, no model dependency. Only proverbs with ``high`` or ``medium``
confidence are used.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEED_DIR = Path(__file__).resolve().parents[3] / "data" / "cultural_seeds"
REGIONS = ("msa", "egyptian", "levantine", "gulf", "maghrebi")
USABLE_CONFIDENCE = {"high", "medium"}

DIALOGUE_INSTRUCTION = {
    "msa": "اكتب السرد والحوار بالعربية الفصحى، بأسلوب الحكاية التراثية.",
    "egyptian": "اكتب السرد بالفصحى، والحوار باللهجة المصرية القاهرية.",
    "levantine": "اكتب السرد بالفصحى، والحوار باللهجة الشامية (الأردنية أو الفلسطينية)، ولا تخلطها بلهجة أخرى.",
    "gulf": "اكتب السرد بالفصحى، والحوار باللهجة الخليجية النجدية، ولا تخلطها بلهجة أخرى.",
    "maghrebi": "اكتب السرد بالفصحى، والحوار بالدارجة المغربية، ولا تخلطها بلهجة أخرى.",
}

SYSTEM_PROMPT = (
    "أنت راوٍ عربي ماهر. اكتب القصة وحدها بالعربية، بلا عنوان ولا مقدمة ولا شرح، "
    "ولا تستخدم أي لغة أخرى."
)


class SeedError(ValueError):
    """Raised when a seed file is missing or a region is unknown."""


def load_seed(region: str, seed_dir: Path | None = None) -> dict[str, Any]:
    if region not in REGIONS:
        raise SeedError(f"unknown region {region!r}; expected one of {REGIONS}")
    path = (seed_dir or SEED_DIR) / f"{region}.json"
    if not path.exists():
        raise SeedError(f"seed file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class StoryPrompt:
    region: str
    system: str
    user: str
    elements: dict[str, Any] = field(default_factory=dict)

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.user}]


def build_story_prompt(
    region: str,
    *,
    premise: str | None = None,
    words: int = 200,
    rng_seed: int = 0,
    n_markers: int = 8,
    seed_dir: Path | None = None,
) -> StoryPrompt:
    """Sample cultural elements for ``region`` and compose the instruction.

    ``premise`` is an optional one-line story idea. Without it, the prompt asks
    for a story built around the sampled value, character and setting.
    """
    seed = load_seed(region, seed_dir)
    rng = random.Random(rng_seed)

    value = rng.choice(seed["values"])
    character = rng.choice(seed["characters"])
    setting = rng.choice(seed["settings"])
    proverbs = [p for p in seed["proverbs"] if p.get("confidence") in USABLE_CONFIDENCE]
    proverb = rng.choice(proverbs) if proverbs else None
    openings = [o["text"] for o in seed["openings_closings"] if o["type"] == "opening"]
    closings = [o["text"] for o in seed["openings_closings"] if o["type"] == "closing"]
    opening = rng.choice(openings) if openings else None
    closing = rng.choice(closings) if closings else None
    markers = rng.sample(seed["dialect_markers"], k=min(n_markers, len(seed["dialect_markers"])))

    lines = [f"اكتب حكاية عربية قصيرة أصلية في نحو {words} كلمة."]
    if premise:
        lines.append(f"فكرة الحكاية: {premise}")
    lines.append(
        f"القيمة التي تدور حولها الحكاية: {value['concept_ar']}؛ {value['story_role']}"
    )
    lines.append(f"من شخصياتها: {character}. ومكانها: {setting}.")
    if proverb:
        lines.append(
            f"اجعل إحدى الشخصيات تقول المثل «{proverb['text']}» في موقف يناسب معناه: {proverb['meaning_msa']}"
        )
    if opening:
        lines.append(f"ابدأ الحكاية بصيغة الافتتاح التقليدية: «{opening}».")
    if closing:
        lines.append(f"واختمها بصيغة الختام التقليدية: «{closing}».")
    lines.append(DIALOGUE_INSTRUCTION[region])
    if region == "msa":
        lines.append("استعمل عبارات سردية تراثية مثل: " + "، ".join(f"«{m['msa']}»" for m in markers[:4]) + ".")
    else:
        lines.append(
            "أمثلة على عبارات الحوار في هذه اللهجة: " + "، ".join(f"«{m['dialect']}»" for m in markers) + "."
        )
    lines.append("اجعل في الحكاية حوارًا بين شخصيتين على الأقل، وعقدة واضحة وحلًّا.")

    elements = {
        "value": value["concept_ar"],
        "character": character,
        "setting": setting,
        "proverb": proverb["text"] if proverb else None,
        "opening": opening,
        "closing": closing,
        "markers": [m["dialect"] for m in markers],
        "seed_review_status": seed.get("provenance", {}).get("review_status"),
        "rng_seed": rng_seed,
    }
    return StoryPrompt(region=region, system=SYSTEM_PROMPT, user="\n".join(lines), elements=elements)
