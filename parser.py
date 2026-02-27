"""CSV parsing and recommendation extraction utilities."""

from __future__ import annotations

import csv
import re
import unicodedata
from io import StringIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

REQUIRED_COLUMNS = {"display_name", "message"}
QUOTE_CHARS = '"\'“”‘’'

PRONOUN_LEADS = {
    "i",
    "i'm",
    "we",
    "my",
    "currently",
    "right now",
    "next up",
    "listening",
    "reading",
}

NON_TITLE_PHRASES = {
    "slow burn",
    "beach read",
    "currently reading",
    "next up",
    "highly recommend",
    "dnf",
    "audiobook",
}

TITLE_META_KEYWORDS = (
    "audiobook",
    "kindle",
    "paperback",
    "hardcover",
    "ebook",
    "edition",
    "unabridged",
    "book",
    "volume",
    "vol.",
    "series",
)

AUTHOR_PARTICLES = {
    "de",
    "del",
    "della",
    "di",
    "du",
    "la",
    "le",
    "van",
    "von",
    "da",
    "dos",
    "das",
    "bin",
    "ibn",
}

AUTHOR_BLOCKLIST = {
    "reading",
    "listening",
    "recommend",
    "recommended",
    "currently",
    "next",
    "up",
    "book",
    "series",
    "chapter",
    "audiobook",
}

BY_PATTERN = re.compile(
    r"(?P<title>[^.\n]{2,80}?)\s+by\s+"
    r"(?P<author>[A-Za-z][A-Za-z.'\-\s\(\)\[\]\{\}]{1,100}?)(?=$|[,;]|\s*-\s*)",
    flags=re.IGNORECASE,
)

DASH_PATTERN = re.compile(
    r"(?P<title>[^.\n]{2,80}?)\s*(?:-|—|–)\s*"
    r"(?P<author>[A-Za-z][A-Za-z.'\-\s\(\)\[\]\{\}]{1,100}?)(?=$|[,;])"
)

QUOTED_PATTERN = re.compile(
    r"(?<!\w)\"([^\"\n]{2,80})\"(?!\w)|(?<!\w)'([^'\n]{2,80})'(?!\w)"
)

EMPHASIZED_PATTERN = re.compile(
    r"(?<!\w)\*\*([^*\n]{2,80})\*\*(?!\w)|(?<!\w)\*([^*\n]{2,80})\*(?!\w)|(?<!\w)_([^_\n]{2,80})_(?!\w)"
)


def parse_comments_csv(file_obj: BinaryIO) -> list[dict[str, str]]:
    """Parse CSV bytes and return normalized comment records."""
    content = file_obj.read().decode("utf-8-sig")
    reader = csv.DictReader(StringIO(content))

    if reader.fieldnames is None:
        raise ValueError("CSV file is empty or missing headers.")

    headers = {h.strip() for h in reader.fieldnames if h}
    missing = REQUIRED_COLUMNS - headers
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    records: list[dict[str, str]] = []
    for row in reader:
        record = {
            "display_name": (row.get("display_name") or "").strip(),
            "message": (row.get("message") or "").strip(),
        }

        uuid_value = (row.get("uuid") or row.get("UUID") or "").strip()
        if "uuid" in headers or "UUID" in headers:
            record["uuid"] = uuid_value

        records.append(record)

    return records


def parse_comments_csv_path(path: str | Path) -> list[dict[str, str]]:
    """Parse CSV from a filesystem path."""
    with Path(path).open("rb") as f:
        return parse_comments_csv(f)


def _normalize_unicode(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u2014", "-").replace("\u2013", "-")
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    return value


def _normalize_text(value: str) -> str:
    value = _normalize_unicode(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _strip_wrappers(value: str) -> str:
    value = _normalize_text(value)
    value = value.strip(" \t\r\n-:;,.!?")

    emphasis_patterns = [r"^\*\*(.+)\*\*$", r"^\*(.+)\*$", r"^__(.+)__$", r"^_(.+)_$"]
    changed = True
    while changed:
        changed = False
        for pattern in emphasis_patterns:
            match = re.match(pattern, value)
            if match:
                value = match.group(1).strip()
                changed = True

    if len(value) >= 2 and value[0] in QUOTE_CHARS and value[-1] in QUOTE_CHARS:
        value = value[1:-1].strip()

    value = value.strip(" \t\r\n-:;,.!?")
    value = re.sub(r"\s+", " ", value)
    return value


def _clean_title(title: str) -> str:
    title = _strip_wrappers(title)
    title = re.sub(r"^[^A-Za-z0-9]+", "", title)
    title = re.sub(r"^(?:i[' ]?m\s+|im\s+)?", "", title, flags=re.IGNORECASE)
    title = re.sub(
        r"^(?:(?:currently|just|almost|finally|gonna)\s+)*(?:"
        r"currently\s+reading|reading\s+now|reading|"
        r"currently\s+listening\s+to|listening\s+to|"
        r"just\s+finished|finished|starting|started|"
        r"i\s+recommend|recommend|recommended|read|try"
        r")\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    if len(title) >= 2 and title[0] in QUOTE_CHARS and title[-1] in QUOTE_CHARS:
        title = title[1:-1]
    title = title.strip(QUOTE_CHARS + " ")
    title = re.sub(r"^(The)([A-Z])", r"\1 \2", title)

    meta_pattern = (
        r"\s*[\(\[\{]\s*[^)\]}]*(?:"
        + "|".join(re.escape(k) for k in TITLE_META_KEYWORDS)
        + r")[^)\]}]*[\)\]\}]\s*$"
    )
    title = re.sub(meta_pattern, "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s{2,}", " ", title)
    title = re.sub(r"[,.;:!?-]+$", "", title)
    return _strip_wrappers(title)


def _clean_author(author: str) -> str:
    author = _strip_wrappers(author)
    author = re.sub(r"^(?:(?:written\s+by|author|by)\s+)+", "", author, flags=re.IGNORECASE)
    author = re.sub(r"\s*[\(\[\{][^)\]}]*[\)\]\}]\s*$", "", author)
    author = re.sub(r"\s{2,}", " ", author)
    author = re.sub(r"[,.;:!?-]+$", "", author)
    return author.strip()


def _split_segments(message: str) -> list[str]:
    normalized = _normalize_unicode(message)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    raw_segments = re.split(r"\n+|;+|(?<!\b[A-Z]\.)(?<=[.!?])\s+", normalized)

    segments: list[str] = []
    for seg in raw_segments:
        seg = re.sub(r"^\s*[-•]\s*", "", seg)
        seg = seg.strip()
        if seg:
            segments.append(seg)

    return segments


def _count_words(value: str) -> int:
    return len(re.findall(r"\b\w+\b", value))


def _is_capitalized_name_token(token: str) -> bool:
    raw = token.strip()
    if not raw:
        return False
    if re.fullmatch(r"(?:[A-Z]\.){1,4}", raw):
        return True
    token = raw.strip(".")
    if not token:
        return False
    if re.fullmatch(r"[A-Z]", token):
        return True
    if re.fullmatch(r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)*", token):
        return True
    return False


def _validate_author(author: str) -> str:
    """Return one of: valid, borderline, invalid."""
    cleaned = _clean_author(author)
    if not cleaned:
        return "invalid"

    words = cleaned.split()
    if not 1 <= len(words) <= 5:
        return "invalid"

    non_particle_words = 0
    valid_tokens = 0

    for word in words:
        lowered = word.casefold().strip(".")
        if lowered in AUTHOR_BLOCKLIST:
            return "invalid"
        if lowered in AUTHOR_PARTICLES:
            continue

        non_particle_words += 1
        if _is_capitalized_name_token(word):
            valid_tokens += 1

    if non_particle_words == 0:
        return "invalid"
    if valid_tokens == non_particle_words:
        return "valid"
    if valid_tokens >= max(1, non_particle_words - 1):
        return "borderline"
    return "invalid"


def _title_penalty(title: str, quoted_or_emphasized: bool) -> tuple[int, bool]:
    penalty = 0
    reject = False

    norm = _clean_title(title)
    lowered = norm.casefold()

    for lead in PRONOUN_LEADS:
        if lowered.startswith(lead + " ") or lowered == lead:
            penalty -= 3
            break

    if any(phrase in lowered for phrase in NON_TITLE_PHRASES):
        penalty -= 3

    word_count = _count_words(norm)
    if len(norm) > 80 or word_count > 12:
        penalty -= 2
        if len(norm) > 140 or word_count > 18:
            reject = True

    if norm.islower() and not quoted_or_emphasized:
        penalty -= 2

    return penalty, reject


def _normalize_dedupe_value(value: str) -> str:
    value = _clean_title(value).casefold()
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_recommendations(
    df: pd.DataFrame,
    aggressive_mode: bool = True,
    include_missing_author: bool = True,
    min_confidence: int = 0,
    include_metadata: bool = False,
    drop_invalid_author: bool = False,
) -> pd.DataFrame:
    """Extract recommendations from comment rows with regex + scoring."""
    missing = {"display_name", "message"} - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    rows: list[dict[str, str | int]] = []

    has_uuid = "uuid" in df.columns
    input_cols = ["display_name", "message"] + (["uuid"] if has_uuid else [])

    for row_index, row in df[input_cols].fillna("").iterrows():
        display_name = _normalize_text(str(row["display_name"]))
        raw_message = str(row["message"])
        message = _normalize_text(raw_message)
        uuid = _normalize_text(str(row["uuid"])) if has_uuid else ""

        seen: set[tuple[str, str]] = set()
        segments = _split_segments(raw_message)

        for segment in segments:
            normalized_segment = _normalize_text(segment)

            candidates: list[dict[str, str | int | bool]] = []

            for match in BY_PATTERN.finditer(normalized_segment):
                candidates.append(
                    {
                        "title": match.group("title"),
                        "author": match.group("author"),
                        "pattern": "title_by_author",
                        "raw_match": match.group(0),
                        "confidence": 3,
                        "quoted_or_emphasized": False,
                    }
                )

            for match in DASH_PATTERN.finditer(normalized_segment):
                candidates.append(
                    {
                        "title": match.group("title"),
                        "author": match.group("author"),
                        "pattern": "title_dash_author",
                        "raw_match": match.group(0),
                        "confidence": 2,
                        "quoted_or_emphasized": False,
                    }
                )

            if aggressive_mode:
                for match in QUOTED_PATTERN.finditer(normalized_segment):
                    title = match.group(1) or match.group(2) or ""
                    candidates.append(
                        {
                            "title": title,
                            "author": "",
                            "pattern": "quoted_title",
                            "raw_match": match.group(0),
                            "confidence": 2,
                            "quoted_or_emphasized": True,
                        }
                    )

                for match in EMPHASIZED_PATTERN.finditer(normalized_segment):
                    title = next((g for g in match.groups() if g), "")
                    candidates.append(
                        {
                            "title": title,
                            "author": "",
                            "pattern": "emphasized_title",
                            "raw_match": match.group(0),
                            "confidence": 2,
                            "quoted_or_emphasized": True,
                        }
                    )

                seg_title = _clean_title(normalized_segment)
                if (
                    2 <= len(seg_title) <= 80
                    and _count_words(seg_title) <= 8
                    and " by " not in seg_title.casefold()
                    and not re.search(r"\s[-—–]\s", seg_title)
                ):
                    candidates.append(
                        {
                            "title": seg_title,
                            "author": "",
                            "pattern": "segment_fallback",
                            "raw_match": normalized_segment,
                            "confidence": 1,
                            "quoted_or_emphasized": False,
                        }
                    )

            for candidate in candidates:
                title_original = _normalize_text(str(candidate["title"]))
                author_original = _normalize_text(str(candidate["author"]))
                title = _clean_title(title_original)
                author = _clean_author(author_original)
                pattern = str(candidate["pattern"])
                confidence = int(candidate["confidence"])

                if not title:
                    continue
                if _count_words(title) > 12 or len(title) > 80:
                    # Keep candidate but score down heavily; very long leftovers are rejected.
                    confidence -= 2

                if pattern in {"title_by_author", "title_dash_author"}:
                    author_status = _validate_author(author)
                    if author_status != "valid":
                        if drop_invalid_author:
                            continue
                        author = ""
                        confidence -= 2

                title_penalty, should_reject = _title_penalty(
                    title, quoted_or_emphasized=bool(candidate["quoted_or_emphasized"])
                )
                confidence += title_penalty
                if should_reject:
                    continue

                if not include_missing_author and not author:
                    continue
                if confidence < min_confidence:
                    continue

                dedupe_key = (
                    _normalize_dedupe_value(title),
                    _normalize_dedupe_value(author),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                out_row: dict[str, str | int] = {
                    "Title": title,
                    "Author": author,
                    "display_name": display_name,
                    "message": message,
                }
                if has_uuid:
                    out_row["uuid"] = uuid
                if include_metadata:
                    out_row["Title_original"] = title_original
                    out_row["Author_original"] = author_original
                    out_row["raw_match"] = _normalize_text(str(candidate["raw_match"]))
                    out_row["pattern"] = pattern
                    out_row["confidence"] = confidence
                rows.append(out_row)

    base_cols = ["Title", "Author"] + (["uuid"] if has_uuid else []) + ["display_name", "message"]
    if include_metadata:
        cols = base_cols + ["Title_original", "Author_original", "raw_match", "pattern", "confidence"]
    else:
        cols = base_cols

    return pd.DataFrame(rows, columns=cols)
