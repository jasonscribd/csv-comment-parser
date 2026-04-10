"""Generate a pastable SQL query from an enhanced book recommendations DataFrame."""

from __future__ import annotations

import pandas as pd

_STATIC_TAIL = """),
ranked AS (
  SELECT
    books.row_num,
    books.search_title,
    books.search_author,
    books.mentions,
    content_info.content_type,
    content_info.isbn,
    content_info.doc_id,
    CASE WHEN content_info.doc_id IS NOT NULL THEN
      CONCAT(
        'https://www.everand.com/audiobook/',
        content_info.doc_id,
        '/',
        REGEXP_REPLACE(
          REGEXP_REPLACE(content_info.doc_title, '[^a-zA-Z0-9 ]', '-'),
          '[ -]+', '-'
        )
      )
    END AS everand_link,
    ROW_NUMBER() OVER (PARTITION BY books.row_num ORDER BY content_info.num_ratings DESC NULLS LAST) AS rn
  FROM books
  LEFT JOIN prod.looker.content_info AS content_info
    ON (
      LOWER(content_info.doc_title) = LOWER(books.search_title)
      OR LOWER(content_info.doc_title) LIKE CONCAT(LOWER(books.search_title), ': %')
      OR LOWER(content_info.doc_title) LIKE CONCAT(LOWER(books.search_title), ' - %')
      OR REGEXP_REPLACE(LOWER(content_info.doc_title), '[^a-z0-9 ]', '') =
         REGEXP_REPLACE(LOWER(books.search_title), '[^a-z0-9 ]', '')
      OR REGEXP_REPLACE(LOWER(content_info.doc_title), '[^a-z0-9 ]', '') LIKE
         CONCAT(REGEXP_REPLACE(LOWER(books.search_title), '[^a-z0-9 ]', ''), ': %')
    )
    AND (books.search_author = '' OR LOCATE(LOWER(books.search_author), LOWER(content_info.authors)) > 0)
    AND content_info.is_published = true
    AND content_info.is_deleted = false
    AND content_info.is_duplicate = false
    AND content_info.content_type = 'audiobook'
    AND content_info.content_language = 'English'
)
SELECT
  row_num,
  search_title,
  search_author,
  mentions,
  content_type,
  isbn,
  doc_id,
  everand_link
FROM ranked
WHERE rn = 1
ORDER BY row_num"""


def _q(val: str) -> str:
    """Wrap value in single quotes, escaping any internal single quotes."""
    return "'" + str(val or "").replace("'", "''") + "'"


def generate_sql(df: pd.DataFrame) -> str:
    """Return a WITH books AS (...) SQL query for the given enhanced results.

    Args:
        df: DataFrame with columns Title, Author, Mentions (one row per book,
            already sorted by mentions descending).

    Returns:
        Complete SQL string ready to paste.
    """
    lines: list[str] = []
    for i, row in enumerate(df.itertuples(index=False)):
        n = i + 1
        title = _q(getattr(row, "Title", "") or "")
        author = _q(getattr(row, "Author", "") or "")
        mentions = int(getattr(row, "Mentions", 1) or 1)
        if i == 0:
            lines.append(
                f"  SELECT {n} AS row_num, {title} AS search_title, "
                f"{author} AS search_author, {mentions} AS mentions"
            )
        else:
            lines.append(f"  UNION ALL SELECT {n}, {title}, {author}, {mentions}")

    return "WITH books AS (\n" + "\n".join(lines) + "\n" + _STATIC_TAIL
