#!/usr/bin/env python3
"""Generate data/sources/source_registry.json (single source of truth; edit THIS file, then rerun).

Every resource records what the thesis claims (section/page), where it officially lives,
version/hash, licence/terms, permission status, allowed uses and whether its files may be
redistributed. Nothing here grants a permission: ``permission_status`` changes only after
a documented owner review (see docs/reproduction/RIGHTS_GATE.md).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATEGORIES = {
    "thesis_original": "Resource the dissertation states it used for ANSG-Original (data, model, evaluation).",
    "supplementary_experiment": "Not claimed by the thesis; used only in new, separately reported experiments.",
    "evaluation_only": "May be used for evaluation/benchmarking under its terms, never for training.",
    "human_reference_only": "May only be read by people (e.g. raters); never copied, trained on or redistributed.",
}
DIALECT_IDS = {"NADI", "QADI", "DART", "PADIC", "Habibi", "Masader"}
PERMISSION = ["verified", "pending_owner_review", "requires_license_agreement", "not_granted", "unavailable",
              "not_applicable"]
USES = ["model_training", "evaluation", "human_reference", "metadata_only"]


def r(id, name, category, thesis_reference, official_source, *, version=None, sha256=None, license=None,
      permission_status, allowed_uses, redistribution_allowed, in_repository=False, availability, notes="",
      dialect_registry_id=None):
    assert category in CATEGORIES and permission_status in PERMISSION and set(allowed_uses) <= set(USES)
    return dict(id=id, name=name, category=category, thesis_reference=thesis_reference,
                dialect_registry_id=dialect_registry_id or (id if id in DIALECT_IDS else None),
                official_source=official_source, version=version, sha256=sha256, license=license,
                permission_status=permission_status, allowed_uses=allowed_uses,
                redistribution_allowed=redistribution_allowed, in_repository=in_repository,
                availability=availability, notes=notes)


ARTEFACT_NOTE = ("github.com/mossab-ansg returned HTTP 404 on 2026-11-14 (WebFetch); Zenodo could not be reached "
                 "from the audit sandbox; see evidence/external/external_evidence.json for the CI check")
RESOURCES = [
    # ---------------------------------------------------------------- thesis-original data
    r("anhc", "Arabic Narrative Heritage Corpus (ANHC)", "thesis_original",
      "§3 (p. 49–75); Table 3.2 (p. 56); App. A.20 (p. 288–290); Table 9.6 (p. 254)",
      "Zenodo DOI 10.5281/zenodo.8847291 (thesis); access via anhc-access@ucm.es (thesis)",
      license="CC BY-NC-SA 4.0 (thesis A.20.2)", permission_status="unavailable", allowed_uses=[],
      redistribution_allowed=None,
      availability="not obtained; thesis gives contradictory access statements (open Zenodo vs controlled "
                   "access vs 'DOI to be assigned')", notes=ARTEFACT_NOTE),
    r("anhc_test_subset", "ANHC test subset", "thesis_original", "App. A.20.2 (p. 290)",
      "https://github.com/mossab-ansg/anhc-test-set (thesis)", permission_status="unavailable",
      allowed_uses=[], redistribution_allowed=None, availability="URL not reachable (404)", notes=ARTEFACT_NOTE),
    r("alf_layla_mahdi_1984", "Alf Layla wa-Layla, Mahdi critical edition (1984)", "thesis_original",
      "§3.2.2 (p. 57); §3.2.4 (p. 58)", "Brill, Leiden (print)", version="Mahdi 1984",
      license="in copyright (modern critical edition)", permission_status="unavailable", allowed_uses=[],
      redistribution_allowed=False, availability="no digital copy available to this project",
      notes="thesis attributes 8.2M tokens to this edition; see DISCREPANCY_REPORT D-DATA-3"),
    r("alf_layla_bulaq_1835", "Alf Layla wa-Layla, Būlāq edition (1835)", "thesis_original", "§3.2.4 (p. 58)",
      "print edition; digital witness not specified by the thesis", version="Būlāq 1835",
      license="public domain (age); digitisation terms depend on the digital source",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None,
      availability="thesis names no digital source"),
    r("alf_layla_calcutta_ii", "Alf Layla wa-Layla, Calcutta II (1839–42)", "thesis_original", "§3.2.4 (p. 58)",
      "print edition; digital witness not specified by the thesis", license="public domain (age)",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None,
      availability="thesis names no digital source"),
    r("kalila_wa_dimna_thesis", "Kalīla wa-Dimna (Ibn al-Muqaffaʿ) as used in ANHC", "thesis_original",
      "§3.2.2 (p. 57)", "edition not specified by the thesis", license="text public domain (age); edition unknown",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None,
      availability="edition unknown", notes="thesis attributes 3.1M tokens; see D-DATA-4"),
    r("classical_adab", "Maqāmāt al-Ḥarīrī, Kitāb al-Aghānī, al-Jāḥiẓ", "thesis_original", "§3.2.2 (p. 57)",
      "editions not specified", license="texts public domain (age); editions unknown",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None,
      availability="editions unknown", notes="13.3M tokens claimed"),
    r("modern_canonical", "Mahfouz, Taha Hussein, canonical adab (1800–2000)", "thesis_original",
      "Table 3.2 (p. 56)", "published books", license="in copyright", permission_status="unavailable",
      allowed_uses=[], redistribution_allowed=False, availability="not licensed to this project"),
    r("contemporary_sources", "Blogs, transcribed oral tales, regional collections (post-2000)", "thesis_original",
      "Table 3.2 (p. 56); §3.2.3 (p. 57)", "not specified", license="unknown", permission_status="unavailable",
      allowed_uses=[], redistribution_allowed=None, availability="sources not identified"),
    r("cst_onto_ckg", "CST-Onto / Cultural Knowledge Graph (1,847 patterns)", "thesis_original",
      "§4.3.2 (p. 98); App. A.19 (p. 286); Table 9.6 (p. 254)",
      "https://github.com/mossab-ansg/ckg-arabic; DOI 10.5281/zenodo.8847378 (thesis)",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None,
      availability="not obtained", notes=ARTEFACT_NOTE),
    r("cultural_marker_lexicon", "Cultural marker lexicon (2,847 entries, CRD)", "thesis_original",
      "§3.6.1, Eq. 3.1 (p. 69–70)", "not published", permission_status="unavailable", allowed_uses=[],
      redistribution_allowed=None, availability="not supplied"),
    r("motif_codebook", "MOTIF codebook (87 archetypes)", "thesis_original", "§3.5.2 (p. 65); App. A.7 (p. 275)",
      "App. A.7 cited; codebook not reproduced in the thesis", permission_status="unavailable", allowed_uses=[],
      redistribution_allowed=None, availability="not supplied"),
    r("ansg_weights", "Trained ANSG checkpoints (phases 1–4)", "thesis_original", "Table 9.6 (p. 254); §5.2.1 (p. 135)",
      "https://github.com/mossab-ansg/ansg-weights; DOI 10.5281/zenodo.8847342 (thesis)",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None, availability="not obtained",
      notes=ARTEFACT_NOTE),
    r("craft_eval", "CRAFT evaluation implementation (CMPI, ALDA, CulturalScore)", "thesis_original",
      "Table 9.6 (p. 254); Ch. 6", "https://github.com/mossab-ansg/craft-eval; DOI 10.5281/zenodo.8847356 (thesis)",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None, availability="not obtained",
      notes=ARTEFACT_NOTE),
    r("ansg_train", "Training pipeline scripts", "thesis_original", "Table 9.6 (p. 254); §5.2.1 (p. 135)",
      "https://github.com/mossab-ansg/ansg-train (DOI ...8847367); ansg-framework commit a7b3c9d (thesis)",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None, availability="not obtained",
      notes=ARTEFACT_NOTE),
    r("human_evaluation_data", "Human evaluation ratings (5 experts × 200 narratives/system)", "thesis_original",
      "§6.1.3 (p. 165); Table 6.7 (p. 176)", "available upon request (thesis §3.7.1)",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None, availability="not obtained"),
    r("madar", "MADAR Arabic Dialect Corpus (parallel)", "thesis_original",
      "Table §3.1.5 (p. 55, as '2.1M, 26 cities'); §4.2.4 (p. 94) and §5.4.3 (p. 140): DIA CNN classifier, 94.3% on MADAR",
      "https://camel.abudhabi.nyu.edu/madar-parallel-corpus/ (CAMeL Lab, NYU Abu Dhabi / CMU)",
      version="as distributed by CAMeL Lab (local copy required)",
      license="MADAR licence (CMU & NYUAD): non-commercial research; no redistribution; no modification",
      permission_status="requires_license_agreement", allowed_uses=["evaluation"], redistribution_allowed=False,
      availability="must be obtained locally by an authorised user; never committed", dialect_registry_id="MADAR",
      notes="official description: 25 city dialects + MSA (Corpus-25/26: 2,000 sentences each; Corpus-5/6: "
            "additional 10,000 for five cities). Thesis '26 cities' and '2.1M' are not verified (D-RES-1)."),
    # ---------------------------------------------------------------- supplementary experiment
    r("wikisource_classical_arabic_pages", "Arabic Wikisource pages: Alf Layla wa-Layla (4 parts + Sinbad), "
      "15 Kalīla wa-Dimna chapters", "supplementary_experiment", None,
      "https://ar.wikisource.org (pinned revision ids in data/corpora/classical_arabic/index.json)",
      version="20 pinned revisions (frozen at commit da54c488)", sha256="per page in index.json",
      license="underlying texts pre-modern (public-domain status to be confirmed per page/edition); "
              "Wikisource site content licence recorded in evidence/external/external_evidence.json",
      permission_status="pending_owner_review", allowed_uses=["metadata_only", "human_reference"],
      redistribution_allowed=None, in_repository=True,
      availability="committed by the owner (PR #24); 195,197 words",
      notes="NOT the ANHC corpus (docs/reproduction/CORPUS_EQUIVALENCE.md). Training is blocked by the rights "
            "gate until the owner records a licence review."),
    r("qwen2_5_1_5b_instruct", "Qwen/Qwen2.5-1.5B-Instruct", "supplementary_experiment", None,
      "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct", license="per model card (verify before use)",
      permission_status="pending_owner_review", allowed_uses=["metadata_only"], redistribution_allowed=None,
      availability="downloaded at run time on Kaggle", notes="not a thesis backbone (C-MODEL-1)"),
    # ---------------------------------------------------------------- dialect resources (data/sources/dialect_registry.json)
    r("NADI", "NADI shared-task data", "evaluation_only", None, "https://nadi.dlnlp.ai/",
      license="edition-specific terms", permission_status="requires_license_agreement", allowed_uses=["evaluation"],
      redistribution_allowed=False, availability="registration may be required"),
    r("QADI", "QADI dialect identification tweets", "evaluation_only", None,
      "https://huggingface.co/datasets/Abdelrahman-Rezk/Arabic_Dialect_Identification",
      license="verify dataset card", permission_status="pending_owner_review", allowed_uses=["evaluation"],
      redistribution_allowed=None, availability="public dataset card"),
    r("DART", "DART dialectal Arabic tweets", "evaluation_only", None,
      "https://www.lrec-conf.org/proceedings/lrec2018/summaries/1013.html",
      license="verify original distribution terms", permission_status="pending_owner_review",
      allowed_uses=["evaluation"], redistribution_allowed=None, availability="see source"),
    r("PADIC", "PADIC parallel dialect corpus", "evaluation_only", None, "https://sites.aub.edu.lb/arabicnlp/tools/",
      license="unverified", permission_status="pending_owner_review", allowed_uses=["evaluation"],
      redistribution_allowed=None, availability="access unverified"),
    r("Habibi", "Habibi Arabic song-lyrics corpus", "human_reference_only", None,
      "https://aclanthology.org/2020.lrec-1.165/", license="CC BY-NC (per author page); lyrics remain copyrighted",
      permission_status="pending_owner_review", allowed_uses=["human_reference"], redistribution_allowed=False,
      availability="public download", notes="song lyrics, not folk narratives"),
    r("Masader", "Masader catalogue of Arabic datasets", "human_reference_only", None,
      "https://arbml.github.io/masader/", license="catalogue; each dataset has its own licence",
      permission_status="not_applicable", allowed_uses=["metadata_only", "human_reference"],
      redistribution_allowed=None, availability="public catalogue"),
    r("modern_novels_wattpad", "Modern dialect novels, Wattpad stories, subtitles", "human_reference_only", None,
      "various", license="in copyright / platform terms", permission_status="not_granted",
      allowed_uses=["human_reference"], redistribution_allowed=False, availability="not cleared",
      dialect_registry_id="Wattpad_and_modern_novels"),
    r("shami_corpus_claimed", "'Shami Corpus' of long-form Levantine stories (as described in a pasted list)",
      "human_reference_only", None, "no persistent official source found", license="unknown",
      permission_status="unavailable", allowed_uses=[], redistribution_allowed=None,
      availability="existence of a licensed long-form narrative release not verified",
      dialect_registry_id="Shami_Corpus_specific"),
]


def main() -> None:
    reg = {"schema": "ansg.source_registry/1", "generated_by": "data/sources/build_source_registry.py",
           "reviewed_on": "2026-11-14", "categories": CATEGORIES, "permission_statuses": PERMISSION,
           "uses": USES, "resources": RESOURCES}
    (HERE / "source_registry.json").write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
