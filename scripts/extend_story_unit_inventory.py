#!/usr/bin/env python3
"""Add whole-book narrative candidates to ANHC's provisional story-unit inventory.

Whole books remain unsegmented source containers. No automatic story boundaries,
annotations, or train/validation/test assignments are created here.
"""
from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc


def norm(value: str | None) -> str:
    value = unicodedata.normalize('NFKC', value or '').casefold()
    value = re.sub(r'[\u064b-\u065f\u0670\u0640]', '', value)
    value = re.sub(r'[أإآٱ]', 'ا', value)
    value = re.sub(r'[^\w]+', ' ', value, flags=re.UNICODE)
    return ' '.join(value.split())


def lock_for(record: dict) -> tuple[str, str]:
    src = record.get('source', {})
    titles = ' '.join(str(src.get(k) or '') for k in ('title', 'original_title'))
    if 'ألف ليلة' in titles or 'alf layla' in titles.casefold() or 'one thousand and one nights' in titles.casefold():
        return 'alf_layla_work_cycle', 'all editions/volumes stay in one split group'
    if 'كليلة ودمنة' in titles or 'kalila' in titles.casefold() or 'bidpai' in titles.casefold():
        return 'kalila_1816_frame_cycle', 'all editions/chapters stay in one split group'
    author = norm(str(src.get('original_author') or src.get('author') or ''))
    title = norm(str(src.get('original_title') or src.get('title') or ''))
    identity = '\x1f'.join((author, title)) if author and title else str(record.get('record_id'))
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f'work_{key}', 'same normalized title/author grouped; split remains unassigned'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate-manifest', type=Path, required=True)
    ap.add_argument('--existing-units', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    existing = list(read_jsonl(args.existing_units))
    existing_ids = {u['unit_id'] for u in existing}
    out = list(existing)
    rows = list(read_jsonl(args.candidate_manifest))
    added = 0
    scope_counts: dict[str, int] = {}
    word_counts: dict[str, int] = {}
    for r in rows:
        rid = r['record_id']
        uid = f'{rid}-whole-book'
        if uid in existing_ids:
            continue
        src, nar = r.get('source', {}), r.get('narrative', {})
        lock, lock_basis = lock_for(r)
        scope = nar.get('candidate_scope', 'unspecified')
        words = int(r.get('counts', {}).get('observed_whitespace_words', 0))
        unit = {
            'schema': 'ansg.story_unit_candidate/1',
            'unit_id': uid,
            'source_record_id': rid,
            'source_cycle_id': lock,
            'source_group_split_lock': lock,
            'split_lock_basis': lock_basis,
            'unit_type': 'whole_book_source_container',
            'title_candidate': src.get('title'),
            'author_candidate': src.get('author'),
            'source_title_original': src.get('original_title'),
            'source_author_original': src.get('original_author'),
            'candidate_scope': scope,
            'source_text_sha256': r.get('source_text_sha256'),
            'observed_whitespace_words': words,
            'status': 'provisional_unsegmented_source_container',
            'parent_story_id': None,
            'annotations': {'DIA': None, 'MOTIF': None, 'MORAL': None},
            'split': 'unassigned',
            'review_required': [
                'review_narrative_relevance_and_exclusions',
                'identify_story_boundaries_or_retain_as_context_only',
                'deduplicate_translations_and_editions_before_split',
                'assign_group_level_split_after_review',
            ],
        }
        out.append(unit)
        added += 1
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        word_counts[scope] = word_counts.get(scope, 0) + words
    out.sort(key=lambda u: (u.get('source_record_id', ''), u.get('unit_type', ''), u.get('unit_id', '')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='\n') as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
    report = {
        'schema': 'ansg.story_unit_review_manifest/2',
        'status': 'provisional_inventory_not_gold_segmented',
        'boundary_policy': 'Whole-book source containers remain unsegmented; heading spans and chapter containers remain provisional and never become gold boundaries automatically.',
        'split_policy': 'All units from the same work/cycle are locked together; no train/validation/test assignment is made until deduplication, boundary review, and grouping are complete.',
        'counts': {
            'story_span_and_chapter_container_rows': sum(1 for u in out if u.get('unit_type') != 'whole_book_source_container'),
            'whole_book_source_container_rows': sum(1 for u in out if u.get('unit_type') == 'whole_book_source_container'),
            'whole_book_candidates_by_scope': {
                scope: sum(1 for u in out if u.get('unit_type') == 'whole_book_source_container' and u.get('candidate_scope') == scope)
                for scope in sorted({u.get('candidate_scope', 'unspecified') for u in out if u.get('unit_type') == 'whole_book_source_container'})
            },
            'whole_book_observed_whitespace_words_by_scope': {
                scope: sum(int(u.get('observed_whitespace_words', 0)) for u in out if u.get('unit_type') == 'whole_book_source_container' and u.get('candidate_scope') == scope)
                for scope in sorted({u.get('candidate_scope', 'unspecified') for u in out if u.get('unit_type') == 'whole_book_source_container'})
            },
            'total_inventory_rows': len(out),
            'new_whole_book_containers_added_this_run': added,
            'unassigned_split_rows': sum(1 for u in out if u.get('split') == 'unassigned'),
            'non_null_annotation_rows': sum(1 for u in out if isinstance(u.get('annotations'), dict) and any(v is not None for v in u['annotations'].values())),
        },
        'limitations': [
            'Candidate-scope labels are intake metadata, not validated story labels.',
            'Word counts use the source manifest whitespace-token convention, not the thesis tokenizer.',
            'The resulting inventory does not establish the thesis ANHC total or its historical split/annotation counts.',
        ],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['counts'], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
