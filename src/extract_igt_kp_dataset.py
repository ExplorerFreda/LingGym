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
import re
import sys
from collections import Counter
from typing import TYPE_CHECKING

from datasets import Dataset, Features, Value


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
    cells skipped, orthography/translation coverage). Unreadable files are
    logged and skipped.

    Args:
        csv_root: Directory holding one subdirectory per language.
        only_language: If given, restrict the scan to this single language.
        stats: Counter mutated in place with extraction tallies.

    Yields:
        One record dict per parseable example: the four IGT fields plus
        ``igt-label``, ``knowledge point``, ``language``, ``source_file``,
        ``chapter`` and ``section``.
    """
    for language, filepath in discover_csv_files(csv_root, only_language):
        stats['files'] += 1
        source_file = os.path.basename(filepath)
        try:
            with open(filepath, newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    stats['rows'] += 1
                    kp = normalize_kp(row.get('Knowledge Point'))
                    chapter = clean_heading(row.get('Chapter'))
                    section = clean_heading(row.get('Section'))
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
    return out


def save_dataset(ds: Dataset, out_dir: str) -> None:
    """Write ``ds`` to disk as an Arrow dataset plus Parquet and JSONL exports.

    Args:
        ds: The dataset to persist.
        out_dir: Output directory for the Arrow dataset; the Parquet and JSONL
            files are written alongside it as ``<out_dir>.parquet`` and
            ``<out_dir>.jsonl``.
    """
    out_dir = out_dir.rstrip('/')
    ds.save_to_disk(out_dir)
    ds.to_parquet(out_dir + '.parquet')
    ds.to_json(out_dir + '.jsonl', lines=True, force_ascii=False)
    logger.info(
        'Saved Arrow -> %s/ ; Parquet -> %s.parquet ; JSONL -> %s.jsonl',
        out_dir,
        out_dir,
        out_dir,
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
    ds = Dataset.from_list(records, features=FEATURES)
    logger.info('Dataset: %d rows, columns=%s', ds.num_rows, ds.column_names)
    save_dataset(ds, args.out_dir)


if __name__ == '__main__':
    main()
