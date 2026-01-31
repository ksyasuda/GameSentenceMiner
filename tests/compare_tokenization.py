"""
Compare MeCab tokenization (with Yomitan-like merging) against Yomitan's native tokenize endpoint.

Usage:
    python -m tests.compare_tokenization [--lines N] [--verbose]

Requires Yomitan to be running with its local API server at http://127.0.0.1:19633
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from GameSentenceMiner.mecab.mecab_controller import MecabController
from GameSentenceMiner.util.db import gsm_db


YOMITAN_API_URL = "http://127.0.0.1:19633"
DEFAULT_LINE_COUNT = 20


@dataclass
class ComparisonResult:
    line_id: str
    line_text: str
    mecab_tokens: list[str]
    yomitan_tokens: Optional[list[str]]
    match: bool
    error: Optional[str] = None


def tokenize_with_yomitan(text: str, scan_length: int = 20) -> tuple[Optional[list[str]], Optional[str]]:
    """
    Call local Yomitan API to tokenize text.

    Returns:
        Tuple of (tokens, error_message). tokens is None if request failed.
    """
    try:
        url = f"{YOMITAN_API_URL}/tokenize"
        data = json.dumps({"text": text, "scanLength": scan_length}).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

        tokens = []
        if result and len(result) > 0:
            content = result[0].get("content", [])
            for segment in content:
                if isinstance(segment, dict):
                    token_text = segment.get("text", "").replace("\n", "").replace("\r", "").strip()
                    if token_text:
                        tokens.append(token_text)
                elif isinstance(segment, list):
                    token_text = ""
                    for item in segment:
                        if isinstance(item, dict):
                            token_text += item.get("text", "")
                    token_text = token_text.replace("\n", "").replace("\r", "").strip()
                    if token_text:
                        tokens.append(token_text)

        return tokens if tokens else [], None

    except urllib.error.URLError as e:
        return None, f"Connection error: {e.reason}"
    except Exception as e:
        return None, f"Error: {e}"


def tokenize_with_mecab(mecab: MecabController, text: str) -> list[str]:
    """Tokenize text using MeCab with Yomitan-like merging."""
    merged_tokens = mecab.translate(text, mimic_yomitan=True)
    return [t.surface for t in merged_tokens]


def fetch_lines_from_db(limit: int) -> list[tuple[str, str]]:
    """
    Fetch lines from the game_lines table.

    Returns:
        List of (id, line_text) tuples
    """
    query = f"""
        SELECT id, line_text
        FROM game_lines
        WHERE line_text IS NOT NULL AND line_text != ''
        ORDER BY RANDOM()
        LIMIT {limit}
    """

    results = gsm_db.fetchall(query)
    return [(row[0], row[1]) for row in results] if results else []


def compare_tokenizations(
    lines: list[tuple[str, str]],
    mecab: MecabController,
    verbose: bool = False
) -> list[ComparisonResult]:
    """Compare tokenization results between MeCab and Yomitan for each line."""
    results = []

    for line_id, line_text in lines:
        mecab_tokens = tokenize_with_mecab(mecab, line_text)
        yomitan_tokens, error = tokenize_with_yomitan(line_text)

        if yomitan_tokens is None:
            match = False
        else:
            match = mecab_tokens == yomitan_tokens

        result = ComparisonResult(
            line_id=line_id,
            line_text=line_text,
            mecab_tokens=mecab_tokens,
            yomitan_tokens=yomitan_tokens,
            match=match,
            error=error
        )
        results.append(result)

        if verbose:
            print_comparison(result)

    return results


def print_comparison(result: ComparisonResult) -> None:
    """Print a single comparison result."""
    status = "MATCH" if result.match else "DIFF"
    if result.error:
        status = "ERROR"

    print(f"\n[{status}] {result.line_text[:50]}{'...' if len(result.line_text) > 50 else ''}")
    print(f"  ID: {result.line_id}")
    print(f"  MeCab:   {' | '.join(result.mecab_tokens)}")

    if result.yomitan_tokens is not None:
        print(f"  Yomitan: {' | '.join(result.yomitan_tokens)}")
    elif result.error:
        print(f"  Yomitan: {result.error}")


def print_summary(results: list[ComparisonResult]) -> None:
    """Print summary statistics."""
    total = len(results)
    matches = sum(1 for r in results if r.match)
    errors = sum(1 for r in results if r.error)
    diffs = total - matches - errors

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total lines:  {total}")
    print(f"Matches:      {matches} ({100 * matches / total:.1f}%)" if total > 0 else "Matches: 0")
    print(f"Differences:  {diffs}")
    print(f"Errors:       {errors}")

    if diffs > 0:
        print("\nDifferences found:")
        for r in results:
            if not r.match and not r.error:
                print(f"\n  Text: {r.line_text}")
                print(f"  MeCab:   {' | '.join(r.mecab_tokens)}")
                print(f"  Yomitan: {' | '.join(r.yomitan_tokens or [])}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare MeCab and Yomitan tokenization results"
    )
    parser.add_argument(
        "--lines", "-n",
        type=int,
        default=DEFAULT_LINE_COUNT,
        help=f"Number of lines to compare (default: {DEFAULT_LINE_COUNT})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print each comparison result"
    )
    args = parser.parse_args()

    print(f"Fetching {args.lines} lines from database...")
    lines = fetch_lines_from_db(args.lines)

    if not lines:
        print("No lines found in database.")
        sys.exit(1)

    print(f"Found {len(lines)} lines.")

    print("Checking Yomitan API availability...")
    _, error = tokenize_with_yomitan("テスト")
    if error:
        print(f"Warning: Yomitan API not available - {error}")
        print("Make sure Yomitan is running with its local API server enabled.")
        sys.exit(1)
    print("Yomitan API is available.")

    print("Initializing MeCab controller...")
    mecab = MecabController(verbose=False, persistent=True)

    print(f"Comparing tokenization for {len(lines)} lines...")
    results = compare_tokenizations(lines, mecab, verbose=args.verbose)

    print_summary(results)


if __name__ == "__main__":
    main()
