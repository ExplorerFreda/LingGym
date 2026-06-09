r"""Extract an IGT + knowledge-point HuggingFace dataset from the CSV grammars.

Each row of ``CSV-format/<Language>/min_knowledge_points_*.csv`` holds one
knowledge point plus up to ten ``(Label N, Content N)`` example pairs. The
``Content N`` cell already contains the full Interlinear Glossed Text packed as
a pipe-delimited LaTeX string (``\\gsrc | \\gll | \\gls | \\glt``). We parse
those cells directly -- the separate ``IGT-format/`` files are not needed
(joining them on label is lossy and adds nothing).

The output is a flat ``datasets.Dataset`` saved as Arrow (``save_to_disk``) plus
Parquet and JSONL exports.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, Features, Value


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger('extract_igt_kp')

# Canonical language directories, mirroring the set in
# qwen2.5_7B_source+gloss+kp+trans.py:14-18.
LANGUAGES = {
    'Fwe',
    'Gyeli',
    'Ik',
    'Japhug',
    'Kagayanen',
    'Kalamang',
    'Komnzo',
    'Mauwake',
    'Mehweb',
    'Moloko',
    'Palula',
    'Papuan_Malay',
    'Pichi',
    'Rapa_Nui',
    'Tuatschin',
    'Ulwa',
    'Vamale',
    'Yauyos_Quecha',
}

# IGT marker tokens. Order matters only for readability; matching is
# boundary-aware so "\gll" never swallows a "\glll" vernacular line.
MARKERS = ('gsrc', 'glt', 'gls', 'gll')

FEATURES = Features(
    {
        'igt-label': Value('string'),
        'igt-orthography': Value('string'),
        'igt-segmentation': Value('string'),
        'igt-gloss': Value('string'),
        'igt-translation': Value('string'),
        'knowledge point': Value('string'),
        'language': Value('string'),
        'source_file': Value('string'),
        'chapter': Value('string'),
        'section': Value('string'),
        'group_id': Value('string'),
    }
)

# Formatting wrappers whose brace content should be kept (the markup unwrapped).
# "textstyle..." commands have variable suffixes and are handled separately.
_WRAPPERS = (
    'textbf',
    'textsc',
    'textit',
    'textitbf',
    'textup',
    'textsubscript',
    'textsuperscript',
    'emph',
    'stem',
    'forme',
    'isi',
    'ipa',
    'mbox',
    'text',
)
_WHITESPACE_RE = re.compile(r'\s+')
_TEXTSTYLE_RE = re.compile(r'\\textstyle[A-Za-z]*\{((?:[^{}]|\{[^{}]*\})*)\}')
# Heading provenance: drop \label{...} (and dangling, unclosed fragments).
_LABEL_RE = re.compile(r'\\label\{[^}]*\}?')

# Example-grouping anchors. A KP range is two adjacent \ref/\cref/\autoref
# anchors joined by one or more hyphen/en-dash/em-dash characters (an optional
# empty "{}" may sit between the first anchor and the dash).
_REF_RANGE_RE = re.compile(
    r'\\(?:c?ref|autoref)\{([^}]+)\}\s*(?:\{\s*\})?\s*[-–—]+\s*'
    r'\\(?:c?ref|autoref)\{([^}]+)\}',
    re.IGNORECASE,
)
# Split a label into (scope, trailing integer): "ex:1:x656" -> ("ex:1:x", 656).
_TRAILING_INT_RE = re.compile(r'(.*?)(\d+)$')
# Only "ex:"/"exe:" labels carry sequential example indices worth expanding;
# bookmark ids ("bkm:Ref...") and section/figure refs must not split a row.
_EXAMPLE_LABEL_PREFIX_RE = re.compile(r'exe?:', re.IGNORECASE)


def unwrap_command(text: str, cmd: str) -> str:
    r"""Replace ``\\cmd{X}`` with ``X`` repeatedly (handles one nesting level).

    Args:
        text: The string to rewrite.
        cmd: The LaTeX command name, without its leading backslash.

    Returns:
        ``text`` with every ``\\cmd{...}`` wrapper replaced by its brace content.
    """
    pattern = re.compile(r'\\' + cmd + r'\{((?:[^{}]|\{[^{}]*\})*)\}')
    while pattern.search(text):
        text = pattern.sub(r'\1', text)
    return text


def _unwrap_all(text: str) -> str:
    r"""Unwrap every known formatting command, including ``\\textstyle*`` variants.

    Args:
        text: The string to rewrite.

    Returns:
        ``text`` with all ``_WRAPPERS`` commands and ``\\textstyle`` variants
        reduced to their brace content.
    """
    for cmd in _WRAPPERS:
        text = unwrap_command(text, cmd)
    while _TEXTSTYLE_RE.search(text):
        text = _TEXTSTYLE_RE.sub(r'\1', text)
    return text


def normalize_igt(text: str) -> str:
    """Light LaTeX cleanup for an IGT field, preserving linguistic content.

    Args:
        text: The raw IGT field as stored in the CSV cell.

    Returns:
        The field with formatting commands unwrapped, a few LaTeX escapes mapped
        to their Unicode equivalents, and runs of whitespace collapsed.
    """
    text = _unwrap_all(text)
    text = text.replace(r'\redp{}', '~').replace(r'\redp', '~')
    text = text.replace(r'\textasciitilde', '~')
    text = text.replace(r'\dots', '…').replace(r'\ldots', '…')
    text = text.replace(r'\tab', ' ').replace(r'\bluebold', '')
    return _WHITESPACE_RE.sub(' ', text).strip()


def normalize_kp(text: str) -> str:
    r"""Light cleanup for knowledge-point prose.

    Unwraps emphasis but keeps ``\\ref``/``\\label``/``\\cite`` anchors, which
    encode the example-label links.

    Args:
        text: The raw ``Knowledge Point`` cell, or ``None``.

    Returns:
        The cleaned prose, or an empty string when ``text`` is ``None``.
    """
    if text is None:
        return ''
    text = _unwrap_all(text)
    return _WHITESPACE_RE.sub(' ', text).strip()


def clean_heading(text: str | None) -> str:
    r"""Make a Chapter/Section provenance string human-readable.

    Args:
        text: The raw heading cell, or ``None``.

    Returns:
        The heading with ``\\label{...}`` markers and formatting stripped, or an
        empty string when ``text`` is ``None``.
    """
    if text is None:
        return ''
    text = _LABEL_RE.sub('', text)
    text = _unwrap_all(text)
    return _WHITESPACE_RE.sub(' ', text).strip()


def _match_marker(part: str) -> tuple[str | None, str]:
    r"""Classify a pipe segment as ``(marker, remainder)`` or ``(None, part)``.

    Args:
        part: One pipe-delimited segment of a ``Content N`` cell.

    Returns:
        A ``(marker, remainder)`` pair when ``part`` opens with a known IGT
        marker (the leading ``\\marker`` stripped from ``remainder``), otherwise
        ``(None, part)``.
    """
    for marker in MARKERS:
        tok = '\\' + marker
        if part == tok or part.startswith((tok + ' ', tok + '\t', tok + '{')):
            return marker, part[len(tok) :].strip()
    return None, part


def split_marker_parts(content: str) -> list[tuple[str | None, str]]:
    """Split a ``Content N`` cell on ``|`` and classify each non-empty segment.

    Args:
        content: The raw ``Content N`` cell text.

    Returns:
        The ``_match_marker`` classification of every non-empty, stripped pipe
        segment, in order.
    """
    parts = (p.strip() for p in content.split('|'))
    return [_match_marker(p) for p in parts if p]


def parse_content(content: str | None) -> dict[str, str] | None:
    r"""Parse a ``Content N`` cell into the four IGT fields, or ``None`` to skip.

    Skips empty / ``"Not found"`` cells and any cell lacking both a ``\\gll``
    (segmentation) and a ``\\gls`` (gloss); ``\\gsrc`` and ``\\glt`` are
    optional. The first occurrence of each marker wins.

    Args:
        content: The raw ``Content N`` cell text, or ``None``.

    Returns:
        A dict with ``igt-orthography``, ``igt-segmentation``, ``igt-gloss`` and
        ``igt-translation`` keys, or ``None`` when the cell is empty, marked
        ``"Not found"``, or missing the required segmentation/gloss markers.
    """
    if content is None:
        return None
    stripped = content.strip()
    if not stripped or stripped.lower() == 'not found':
        return None

    fields: dict[str, str | None] = {m: None for m in MARKERS}
    for marker, remainder in split_marker_parts(stripped):
        if marker is not None and fields[marker] is None:
            fields[marker] = remainder

    if fields['gll'] is None or fields['gls'] is None:
        return None

    return {
        'igt-orthography': normalize_igt(fields['gsrc']) if fields['gsrc'] else '',
        'igt-segmentation': normalize_igt(fields['gll']),
        'igt-gloss': normalize_igt(fields['gls']),
        'igt-translation': normalize_igt(fields['glt']) if fields['glt'] else '',
    }


def _split_trailing_int(label: str) -> tuple[str, int | None]:
    """Split a label into its scope and trailing integer.

    Args:
        label: An example label, e.g. ``ex:1:x656`` or ``exe:main``.

    Returns:
        A ``(scope, number)`` pair where ``scope`` is the substring before the
        trailing run of digits and ``number`` is that run as an ``int``
        (``ex:1:x656`` -> ``('ex:1:x', 656)``), or ``(label, None)`` when the
        label does not end in a digit.
    """
    match = _TRAILING_INT_RE.match(label)
    if match:
        return match.group(1), int(match.group(2))
    return label, None


def _sequential_ranges(kp_text: str) -> list[tuple[str, int, int]]:
    r"""Extract the expandable example ranges referenced by a knowledge point.

    A range counts as expandable only when both ``\\ref`` endpoints are
    example-style labels (``ex:``/``exe:``) sharing one scope with ascending
    trailing integers; bookmark-id or descending ranges are ignored so they do
    not split a row.

    Args:
        kp_text: The raw ``Knowledge Point`` cell text.

    Returns:
        A ``(scope, lo, hi)`` tuple per expandable range, in order of
        appearance.
    """
    ranges: list[tuple[str, int, int]] = []
    for start, end in _REF_RANGE_RE.findall(kp_text):
        if not (_EXAMPLE_LABEL_PREFIX_RE.match(start) and _EXAMPLE_LABEL_PREFIX_RE.match(end)):
            continue
        start_scope, start_num = _split_trailing_int(start)
        end_scope, end_num = _split_trailing_int(end)
        if start_num is None or end_num is None:
            continue
        if start_scope != end_scope or start_num > end_num:
            continue
        ranges.append((start_scope, start_num, end_num))
    return ranges


def _partition_row(labels: list[str], kp_text: str) -> list[int]:
    r"""Assign each example label in a row to a local group key.

    Every label defaults to group ``0`` (the row's catch-all group). Each
    expandable range from ``kp_text`` claims the labels whose scope and trailing
    integer fall inside it, minting a fresh group key (``1``, ``2`` ...) the
    first time it claims a label; a label matched by several ranges joins the
    first.

    Args:
        labels: The parseable example labels of one row, in column order.
        kp_text: The raw ``Knowledge Point`` cell text for that row.

    Returns:
        A list of local group keys parallel to ``labels``.
    """
    ranges = _sequential_ranges(kp_text)
    range_keys: dict[tuple[str, int, int], int] = {}
    keys: list[int] = []
    for label in labels:
        scope, num = _split_trailing_int(label)
        key = 0
        if num is not None:
            for rng in ranges:
                if scope == rng[0] and rng[1] <= num <= rng[2]:
                    key = range_keys.setdefault(rng, len(range_keys) + 1)
                    break
        keys.append(key)
    return keys


def discover_csv_files(csv_root: str, only_language: str | None = None) -> list[tuple[str, str]]:
    """Return ``(language, filepath)`` pairs for every grammar CSV.

    Scans the immediate language subdirectories of ``csv_root``, skipping any
    whose name is not in ``LANGUAGES``, and collects every ``.csv`` inside.

    Args:
        csv_root: Directory holding one subdirectory per language.
        only_language: If given, restrict the scan to this single language.

    Returns:
        ``(language, filepath)`` pairs for each grammar CSV, sorted by language
        and then by file name.
    """
    found: list[tuple[str, str]] = []
    for language in sorted(os.listdir(csv_root)):
        lang_dir = os.path.join(csv_root, language)
        if not os.path.isdir(lang_dir):
            continue
        if language not in LANGUAGES:
            logger.warning('Skipping unexpected directory: %s', language)
            continue
        if only_language and language != only_language:
            continue
        for name in sorted(os.listdir(lang_dir)):
            if name.endswith('.csv'):
                found.append((language, os.path.join(lang_dir, name)))
    return found


def iter_records(
    csv_root: str, only_language: str | None, stats: Counter
) -> Iterator[dict[str, str]]:
    """Yield one record dict per parseable ``(Label N, Content N)`` pair.

    Reads every CSV from ``discover_csv_files``, parses each of the up-to-ten
    example columns, and updates ``stats`` with per-stage counts (rows read,
    cells skipped, orthography/translation coverage, groups minted). The
    parseable examples of each row are partitioned into groups (see
    ``_partition_row``) and tagged with a ``group_id`` of the form
    ``<language>:<n>``, where ``n`` restarts per language. Unreadable files are
    logged and skipped.

    Args:
        csv_root: Directory holding one subdirectory per language.
        only_language: If given, restrict the scan to this single language.
        stats: Counter mutated in place with extraction tallies.

    Yields:
        One record dict per parseable example: the four IGT fields plus
        ``igt-label``, ``knowledge point``, ``language``, ``source_file``,
        ``chapter``, ``section`` and ``group_id``.
    """
    lang_counters: dict[str, int] = defaultdict(int)
    for language, filepath in discover_csv_files(csv_root, only_language):
        stats['files'] += 1
        source_file = os.path.basename(filepath)
        try:
            with open(filepath, newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    stats['rows'] += 1
                    raw_kp = row.get('Knowledge Point') or ''
                    kp = normalize_kp(row.get('Knowledge Point'))
                    chapter = clean_heading(row.get('Chapter'))
                    section = clean_heading(row.get('Section'))
                    examples: list[tuple[str, dict[str, str]]] = []
                    for i in range(1, 11):
                        content = row.get(f'Content {i}')
                        if content is None:
                            continue
                        if not content.strip():
                            stats['skip_empty'] += 1
                            continue
                        if content.strip().lower() == 'not found':
                            stats['skip_not_found'] += 1
                            continue
                        parsed = parse_content(content)
                        if parsed is None:
                            stats['skip_incomplete'] += 1
                            continue
                        label = (row.get(f'Label {i}') or '').strip()
                        examples.append((label, parsed))
                    if not examples:
                        continue
                    local_keys = _partition_row([label for label, _ in examples], raw_kp)
                    group_ids: dict[int, str] = {}
                    for key in local_keys:
                        if key not in group_ids:
                            group_ids[key] = f'{language}:{lang_counters[language]}'
                            lang_counters[language] += 1
                            stats['groups'] += 1
                    for (label, parsed), key in zip(examples, local_keys, strict=True):
                        stats[
                            'with_orthography'
                            if parsed['igt-orthography']
                            else 'without_orthography'
                        ] += 1
                        if parsed['igt-translation']:
                            stats['with_translation'] += 1
                        yield {
                            'igt-label': label,
                            **parsed,
                            'knowledge point': kp,
                            'language': language,
                            'source_file': source_file,
                            'chapter': chapter,
                            'section': section,
                            'group_id': group_ids[key],
                        }
        except OSError as exc:
            logger.error('Failed to read %s: %s', filepath, exc)


def dedup_records(records: Iterator[dict[str, str]], stats: Counter) -> list[dict]:
    """Keep one record per distinct content tuple, preserving first-seen order.

    Args:
        records: The record dicts to deduplicate.
        stats: Counter mutated in place with ``pre_dedup`` and ``post_dedup``
            totals.

    Returns:
        The deduplicated records, in first-seen order.
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for rec in records:
        stats['pre_dedup'] += 1
        key = (
            rec['igt-label'],
            rec['knowledge point'],
            rec['igt-segmentation'],
            rec['igt-gloss'],
            rec['igt-translation'],
            rec['igt-orthography'],
            rec['language'],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    stats['post_dedup'] = len(out)
    logger.info(
        'Deduplication dropped %d of %d records (%d kept)',
        stats['pre_dedup'] - len(out),
        stats['pre_dedup'],
        len(out),
    )
    return out


def split_train_test(
    records: list[dict], test_frac: float, seed: int, stats: Counter
) -> tuple[list[dict], list[dict]]:
    """Split records into train/test, reserving ``test_frac`` of each language.

    The split is taken at the ``group_id`` level so a group's examples never
    straddle the boundary. For each language its groups are shuffled with a
    seeded RNG and whole groups are moved to the test set until that language's
    held-out record count reaches ``round(test_frac * language_total)``. Because
    only whole groups move, the realised test fraction may overshoot the target
    by a few records. The split is deterministic for a given ``seed`` and set of
    ``group_id`` values, and each split keeps the input's first-seen order.

    Args:
        records: The deduplicated record dicts to split.
        test_frac: Fraction of each language's records to reserve for test.
        seed: Seed for the per-language group shuffle.
        stats: Counter mutated in place with ``train`` and ``test`` totals.

    Returns:
        A ``(train, test)`` pair of record lists.
    """
    group_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        group_counts[rec['language']][rec['group_id']] += 1

    rng = random.Random(seed)
    test_ids: set[str] = set()
    for language in sorted(group_counts):
        counts = group_counts[language]
        group_ids = sorted(counts)
        rng.shuffle(group_ids)
        target = round(test_frac * sum(counts.values()))
        held = 0
        for group_id in group_ids:
            if held >= target:
                break
            test_ids.add(group_id)
            held += counts[group_id]
        logger.info(
            '  %-16s train=%d test=%d (%.1f%%)',
            language,
            sum(counts.values()) - held,
            held,
            100 * held / sum(counts.values()),
        )

    train = [rec for rec in records if rec['group_id'] not in test_ids]
    test = [rec for rec in records if rec['group_id'] in test_ids]
    stats['train'] = len(train)
    stats['test'] = len(test)
    return train, test


def save_dataset(ds: Dataset | DatasetDict, out_dir: str) -> None:
    """Write ``ds`` to disk as an Arrow dataset plus Parquet and JSONL exports.

    For a ``DatasetDict`` the Arrow store at ``out_dir`` holds one subdirectory
    per split and the flat exports are written per split as
    ``<out_dir>.<split>.parquet`` / ``<out_dir>.<split>.jsonl``; for a plain
    ``Dataset`` they are ``<out_dir>.parquet`` / ``<out_dir>.jsonl``.

    Args:
        ds: The dataset (or split dict) to persist.
        out_dir: Output directory for the Arrow store; the Parquet and JSONL
            files are written alongside it.
    """
    out_dir = out_dir.rstrip('/')
    ds.save_to_disk(out_dir)
    splits = ds.items() if isinstance(ds, DatasetDict) else [(None, ds)]
    for split, part in splits:
        stem = out_dir if split is None else f'{out_dir}.{split}'
        part.to_parquet(stem + '.parquet')
        part.to_json(stem + '.jsonl', lines=True, force_ascii=False)
        logger.info(
            'Saved Arrow -> %s/ ; Parquet -> %s.parquet ; JSONL -> %s.jsonl',
            out_dir if split is None else f'{out_dir}/{split}',
            stem,
            stem,
        )


def _report(stats: Counter) -> None:
    """Log the per-stage extraction tallies in a fixed order.

    Args:
        stats: The counter populated by ``iter_records`` and ``dedup_records``.
    """
    logger.info('--- extraction report ---')
    for key in (
        'files',
        'rows',
        'skip_not_found',
        'skip_empty',
        'skip_incomplete',
        'with_orthography',
        'without_orthography',
        'with_translation',
        'groups',
        'pre_dedup',
        'post_dedup',
    ):
        logger.info('  %-20s %d', key, stats[key])


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the extraction, and save the resulting dataset.

    Args:
        argv: Command-line arguments to parse; defaults to ``sys.argv`` when
            ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Data dirs (CSV-format/, igt_kp_dataset/) live at the project root,
    # one level up from this src/ module.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--csv-root', default=os.path.join(here, 'CSV-format'))
    parser.add_argument('--out-dir', default=os.path.join(here, 'igt_kp_dataset'))
    parser.add_argument('--only-language', default=None)
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Cap on records (after dedup) for quick dry runs.',
    )
    parser.add_argument(
        '--test-frac',
        type=float,
        default=0.2,
        help='Per-language fraction reserved for the test split, taken at the '
        'group_id level; 0 produces a single unsplit dataset.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for the train/test group shuffle.',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    # Some Content/Knowledge-Point cells are large; lift the csv field cap.
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

    stats: Counter = Counter()
    records = dedup_records(iter_records(args.csv_root, args.only_language, stats), stats)
    if args.limit is not None:
        records = records[: args.limit]
        stats['post_dedup'] = len(records)

    _report(stats)
    if args.test_frac > 0:
        logger.info('--- train/test split (seed=%d) ---', args.seed)
        train, test = split_train_test(records, args.test_frac, args.seed, stats)
        ds: Dataset | DatasetDict = DatasetDict(
            {
                'train': Dataset.from_list(train, features=FEATURES),
                'test': Dataset.from_list(test, features=FEATURES),
            }
        )
        logger.info('Dataset: train=%d, test=%d rows', len(train), len(test))
    else:
        ds = Dataset.from_list(records, features=FEATURES)
        logger.info('Dataset: %d rows, columns=%s', ds.num_rows, ds.column_names)
    save_dataset(ds, args.out_dir)


if __name__ == '__main__':
    main()
