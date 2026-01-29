#!/usr/bin/env python3
"""
Test script to compare MeCab tokenization with and without Yomitan-style merging.
"""

import sys
import io
from pathlib import Path

# Fix Windows console Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from GameSentenceMiner.mecab.mecab_controller import MecabController


TEST_SENTENCES = [
    "食べている",
    "この世界の片隅に",
    "ぐらい上目遣いで言った方がやる気出るぜ？",
    "奇襲でもされたときに君が真っ先にやられると全滅確定",
    "でも、頑張って",
    "そっかそっか。ならま、いいんじゃねーかな",
    "昨日すき焼きを食べました",
    "放っておけない",
    "彼女は走っていった",
    "読んでいる本が面白い",
    "食べさせられる",
    "行かなければならない",
    "遊んでばかりいる",
]


def format_tokens(tokens, use_surface=True) -> str:
    """Format tokens as pipe-separated string."""
    if use_surface:
        # For merged tokens, use .surface attribute
        return "|".join(t.surface for t in tokens)
    else:
        # For raw MeCab tokens, use .word attribute
        return "|".join(t.word for t in tokens)


def main():
    mecab = MecabController()

    print("=" * 70)
    print("MeCab Token Merging Comparison")
    print("=" * 70)
    print()

    for sentence in TEST_SENTENCES:
        # Get raw MeCab tokens
        raw_tokens = mecab.translate(sentence, mimic_yomitan=False)
        raw_str = format_tokens(raw_tokens, use_surface=False)

        # Get merged (Yomitan-style) tokens
        merged_tokens = mecab.translate(sentence, mimic_yomitan=True)
        merged_str = format_tokens(merged_tokens, use_surface=True)

        # Calculate label width for alignment
        label_width = 10

        print(f"Sentence: {sentence}")
        print(f"{'raw':>{label_width}}: {raw_str}")
        print(f"{'merged':>{label_width}}: {merged_str}")

        # Show if any merging happened
        if len(merged_tokens) < len(raw_tokens):
            diff = len(raw_tokens) - len(merged_tokens) + 1
            print(f"{'':>{label_width}}  ({diff} merge{'d' if diff > 1 else ''})")

        print()

    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
