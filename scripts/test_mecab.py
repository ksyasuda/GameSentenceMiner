#!/usr/bin/env python
"""
Test script for the MeCab parser bundled with GameSentenceMiner.

Usage:
    python test_mecab.py "日本語のテキスト"
    python test_mecab.py --raw "日本語のテキスト"
    python test_mecab.py --reading "日本語のテキスト"
    python test_mecab.py --hiragana "日本語のテキスト"
"""

import argparse
import io
import sys
from pathlib import Path

# Fix encoding for Windows console - only when running directly, not under pytest
if sys.platform == "win32" and "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the parent directory to the path so we can import GameSentenceMiner modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from GameSentenceMiner.mecab.mecab_controller import MecabController
from GameSentenceMiner.mecab.basic_types import MecabParsedToken, PartOfSpeech, Inflection


def format_token(token: MecabParsedToken, index: int) -> str:
    """Format a single token with all available information."""
    lines = [
        f"Token #{index + 1}: {token.word}",
        f"  Headword (dictionary form): {token.headword}",
        f"  Katakana reading: {token.katakana_reading or '(same as word)'}",
        f"  Part of speech: {token.part_of_speech.name} ({token.part_of_speech.value or 'N/A'})",
        f"  Inflection type: {token.inflection_type.name} ({token.inflection_type.value or 'N/A'})",
    ]
    return "\n".join(lines)


def print_tokens(tokens: list[MecabParsedToken]) -> None:
    """Print all tokens with their information."""
    print(f"\n{'=' * 60}")
    print(f"Total tokens: {len(tokens)}")
    print(f"{'=' * 60}\n")

    for i, token in enumerate(tokens):
        print(format_token(token, i))
        print()


def print_summary_table(tokens: list[MecabParsedToken]) -> None:
    """Print a compact summary table of all tokens."""
    print(f"\n{'=' * 80}")
    print(f"{'Word':<15} {'Headword':<15} {'Reading':<15} {'POS':<15} {'Inflection':<15}")
    print(f"{'=' * 80}")

    for token in tokens:
        word = token.word[:14]
        headword = token.headword[:14]
        reading = (token.katakana_reading or "-")[:14]
        pos = token.part_of_speech.name[:14]
        inflection = token.inflection_type.name[:14]
        print(f"{word:<15} {headword:<15} {reading:<15} {pos:<15} {inflection:<15}")

    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test the MeCab parser bundled with GameSentenceMiner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_mecab.py "昨日、林檎を2個買った。"
  python test_mecab.py --table "日本語を勉強しています"
  python test_mecab.py --reading "野獣の様な男"
  python test_mecab.py --hiragana "機械学習"
  python test_mecab.py --raw "テスト"
        """
    )

    parser.add_argument(
        "text",
        help="Japanese text to parse"
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw MeCab output (before parsing into tokens)"
    )

    parser.add_argument(
        "--reading",
        action="store_true",
        help="Show furigana-formatted reading (Anki format)"
    )

    parser.add_argument(
        "--hiragana",
        action="store_true",
        help="Convert text to hiragana"
    )

    parser.add_argument(
        "--table",
        action="store_true",
        help="Show compact table format instead of detailed output"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose MeCab output"
    )

    args = parser.parse_args()

    print(f"\nInput text: {args.text}")

    # Initialize MeCab controller
    mecab = MecabController(verbose=args.verbose, cache_max_size=1024)

    if args.raw:
        # Show raw MeCab output using the underlying controller
        print("\n--- Raw MeCab Output ---")
        raw_output = mecab.translate(args.text)
        for line in raw_output:
            print(line)
        print("--- End Raw Output ---\n")

    if args.reading:
        reading = mecab.reading(args.text)
        print(f"\nFurigana reading (Anki format): {reading}")

    if args.hiragana:
        hiragana = mecab.to_hiragana(args.text)
        print(f"\nHiragana: {hiragana}")

    # Always show parsed tokens unless only special modes requested
    if not (args.raw or args.reading or args.hiragana) or args.table:
        tokens = mecab.translate(args.text)

        if args.table:
            print_summary_table(tokens)
        else:
            print_tokens(tokens)
    elif not args.table:
        # If special modes were used, still show tokens by default
        tokens = mecab.translate(args.text)
        print_tokens(tokens)


if __name__ == "__main__":
    main()
