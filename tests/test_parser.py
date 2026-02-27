from io import BytesIO

import pandas as pd
import pytest

from parser import extract_recommendations, parse_comments_csv


def test_parse_comments_csv_success():
    csv_bytes = b"display_name,message\nAlice,Hello\nBob,Hi there"
    records = parse_comments_csv(BytesIO(csv_bytes))

    assert records == [
        {"display_name": "Alice", "message": "Hello"},
        {"display_name": "Bob", "message": "Hi there"},
    ]


def test_parse_comments_csv_preserves_uuid_when_present():
    csv_bytes = b"uuid,display_name,message\nu-1,Alice,Hello\nu-2,Bob,Hi there"
    records = parse_comments_csv(BytesIO(csv_bytes))

    assert records == [
        {"display_name": "Alice", "message": "Hello", "uuid": "u-1"},
        {"display_name": "Bob", "message": "Hi there", "uuid": "u-2"},
    ]


def test_parse_comments_csv_missing_required_column():
    csv_bytes = b"display_name,text\nAlice,Hello"

    with pytest.raises(ValueError, match="missing required columns: message"):
        parse_comments_csv(BytesIO(csv_bytes))


def test_parse_comments_csv_empty_headers():
    csv_bytes = b""

    with pytest.raises(ValueError, match="empty or missing headers"):
        parse_comments_csv(BytesIO(csv_bytes))


def test_segmentation_prevents_cross_sentence_greedy_match():
    df = pd.DataFrame(
        [
            {
                "display_name": "Alice",
                "message": "I loved this one. Project Hail Mary by Andy Weir. Next up: maybe Dune.",
            }
        ]
    )

    out = extract_recommendations(df, aggressive_mode=False, min_confidence=2, include_metadata=True)

    pairs = {(row["Title"], row["Author"]) for _, row in out.iterrows()}
    assert ("Project Hail Mary", "Andy Weir") in pairs
    assert all("I loved this one" not in title for title, _ in pairs)


def test_reject_common_false_positives_by_confidence():
    df = pd.DataFrame(
        [
            {
                "display_name": "Bob",
                "message": "slow burn; beach read; currently reading something great",
            }
        ]
    )

    out = extract_recommendations(df, aggressive_mode=True, min_confidence=2)
    assert out.empty


def test_valid_formats_are_captured():
    df = pd.DataFrame(
        [
            {
                "display_name": "Cara",
                "message": 'Project Hail Mary by Andy Weir; Cassidy Blake - V.E. Schwab; "Piranesi"',
            }
        ]
    )

    out = extract_recommendations(df, aggressive_mode=True, min_confidence=2)
    tuples = {(row["Title"], row["Author"]) for _, row in out.iterrows()}

    assert ("Project Hail Mary", "Andy Weir") in tuples
    assert ("Cassidy Blake", "V.E. Schwab") in tuples
    assert ("Piranesi", "") in tuples


def test_dedupe_within_same_comment():
    df = pd.DataFrame(
        [
            {
                "display_name": "Dan",
                "message": 'Dune by Frank Herbert; dune by Frank Herbert; "Dune"',
            }
        ]
    )

    out = extract_recommendations(df, aggressive_mode=True, include_missing_author=True, min_confidence=0)
    pairs = [(row["Title"], row["Author"]) for _, row in out.iterrows()]

    assert pairs.count(("Dune", "Frank Herbert")) == 1


def test_cleanup_activity_prefix_and_title_metadata():
    df = pd.DataFrame(
        [
            {
                "display_name": "Eli",
                "message": "I'm currently reading TheGerman House (Audiobook) by Annette Hess",
            }
        ]
    )

    out = extract_recommendations(df, aggressive_mode=False, min_confidence=2)
    tuples = {(row["Title"], row["Author"]) for _, row in out.iterrows()}
    assert ("The German House", "Annette Hess") in tuples


def test_cleanup_author_role_words_and_bracket_suffix():
    df = pd.DataFrame(
        [
            {
                "display_name": "Fran",
                "message": "Piranesi by written by Susanna Clarke (Editor)",
            }
        ]
    )

    out = extract_recommendations(df, aggressive_mode=False, min_confidence=1)
    tuples = {(row["Title"], row["Author"]) for _, row in out.iterrows()}
    assert ("Piranesi", "Susanna Clarke") in tuples


def test_extract_recommendations_requires_columns():
    df = pd.DataFrame([{"name": "Alice", "text": "Dune"}])

    with pytest.raises(ValueError, match="missing required columns"):
        extract_recommendations(df)


def test_extract_recommendations_carries_uuid_column():
    df = pd.DataFrame(
        [
            {
                "uuid": "abc-123",
                "display_name": "Gio",
                "message": "Project Hail Mary by Andy Weir",
            }
        ]
    )

    out = extract_recommendations(df, aggressive_mode=False, min_confidence=2)
    assert list(out["uuid"]) == ["abc-123"]
