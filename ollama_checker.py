"""Ollama cloud API integration for cleaning and ranking book recommendations."""

from __future__ import annotations

import json
import re

import pandas as pd


def _aggregate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Group by normalized title+author and count occurrences."""
    df = df.copy()
    df["Title"] = df["Title"].fillna("").str.strip()
    df["Author"] = df["Author"].fillna("").str.strip()
    df["_key"] = df["Title"].str.lower() + " || " + df["Author"].str.lower()

    agg = (
        df.groupby("_key", sort=False)
        .agg(Title=("Title", "first"), Author=("Author", "first"), mentions=("_key", "count"))
        .reset_index(drop=True)
        .sort_values("mentions", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def enhance_with_ollama(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Use Ollama GLM-5.1 cloud to clean titles/authors and merge near-duplicates.

    Args:
        df: Raw extraction DataFrame with Title and Author columns.
        api_key: Ollama cloud API key from ollama.com.

    Returns:
        DataFrame with columns Title, Author, Mentions sorted by Mentions descending.
    """
    from openai import OpenAI  # lazy import; openai is optional until this step

    agg = _aggregate_raw(df)
    book_list = [
        {"title": row["Title"], "author": row["Author"], "count": int(row["mentions"])}
        for _, row in agg.iterrows()
    ]

    prompt = (
        "You are a book data cleaning assistant.\n"
        "I extracted book recommendations from user comments. "
        "Some entries are duplicates of the same book (different spelling, formatting, or partial title). "
        "Some titles or author names may be noisy or incomplete.\n\n"
        "Your tasks:\n"
        "1. Merge entries that refer to the same book, summing their counts.\n"
        "2. Standardize each book title (proper title-case, remove noise).\n"
        "3. Standardize each author name (full name, proper capitalization, e.g. 'Andy Weir').\n"
        "4. Return the result sorted by count descending.\n\n"
        f"Input books:\n{json.dumps(book_list, indent=2)}\n\n"
        'Return ONLY a valid JSON array. Each element must have exactly these keys: '
        '"title" (string), "author" (string), "mentions" (integer). '
        "No markdown fences, no explanation, no extra text."
    )

    client = OpenAI(base_url="https://api.ollama.com/v1", api_key=api_key)
    response = client.chat.completions.create(
        model="glm-5.1:cloud",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps output
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    cleaned = json.loads(raw)

    result = pd.DataFrame(cleaned).rename(
        columns={"title": "Title", "author": "Author", "mentions": "Mentions"}
    )
    return result.sort_values("Mentions", ascending=False).reset_index(drop=True)
