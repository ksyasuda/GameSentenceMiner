#!/usr/bin/env python3
"""
Test script to compare MeCab tokenization with and without Yomitan-style merging.
"""

import sys
import io
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

# fix windows console unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# enable windows ansi escape code support
if sys.platform == "win32":
    os.system("")
# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from GameSentenceMiner.mecab.mecab_controller import MecabController

YOMITAN_API_URL = "http://127.0.0.1:19633"
MERGED_COLOR = "\033[38;2;166;218;149m"  # #a6da95
SEPARATOR_COLOR = "\033[38;2;245;169;127m" #f5a97f
MECAB_COLOR = "\033[38;2;237;135;150m" # #ed8796
PASS_COLOR = "\033[38;2;166;218;149m"  # green
FAIL_COLOR = "\033[38;2;237;135;150m"  # red
RESET_COLOR = "\033[0m"

# Each entry can be either:
# - A string (input only, no expected output check)
# - A tuple (input, expected_merged_output) where expected is pipe-separated
TEST_SENTENCES = [
    # Bug fix verification: symbols should NOT merge with auxiliaries
    ("…た", "…|た"),  # ellipsis + ta should stay separate
    ("~」た", "~」|た"),  # symbols + ta should stay separate (MeCab groups ~」)
    # Bug fix verification: verbs/auxiliaries/adjectives SHOULD merge
    ("食べた", "食べた"),  # verb + ta should merge
    ("なかった", "なかった"),  # auxiliary + ta should merge
    ("高かった", "高かった"),  # i-adjective + ta should merge
    ("走れば", "走れば"),  # verb + ba should merge
    ("言ったって", "言ったって"),  # verb + tatte should merge

    "食べている",
    ("この世界の片隅に", "この|世界|の|片隅|に"),
    ("ぐらい上目遣いで言った方がやる気出るぜ？", "ぐらい|上目遣い|で|言った|方|が|やる気|出る|ぜ|？"),
    ("奇襲でもされたときに君が真っ先にやられると全滅確定", "奇襲|で|も|された|とき|に|君|が|真っ先|に|やられる|と|全滅|確定"),
    ("でも、頑張って", "でも|、|頑張って"),
    ("そっかそっか。ならま、いいんじゃねーかな", "そっ|か|そっ|か|。|なら|ま|、|いい|ん|じゃ|ねー|か|な"),
    "昨日すき焼きを食べました",
    "放っておけない",
    "彼女は走っていった",
    "読んでいる本が面白い",
    "食べさせられる",
    "行かなければならない",
    "遊んでばかりいる",
    ("立ち止まった少女に人混みをかき分けて歩み寄り", "立ち止まった|少女|に|人混み|を|かき分けて|歩み寄り"),
    ("Jonathanです", "Jonathan|です"),
    "老婆心ながら、ひとつ忠告させていただきましょうか",
    "るっせえ",
    "なんかシルヴィが、国に帰ると大切な人と別れなきゃいけないとかなんとか。",
    ("結局最後まで付き合わせてしまいましたね", "結局|最後|まで|付き合わせてしまいました|ね"),
]


def format_tokens(tokens, use_surface=True) -> str:
    """Format tokens as pipe-separated string."""
    if use_surface:
        return f"{SEPARATOR_COLOR}|{RESET_COLOR}".join(t.surface for t in tokens)
    else:
        return f"{SEPARATOR_COLOR}|{RESET_COLOR}".join(t.word for t in tokens)


def format_mecab_tokens_colored(raw_tokens, merged_tokens) -> str:
    """Format raw MeCab tokens with color highlighting for tokens that were merged."""
    merged_surfaces = [t.surface for t in merged_tokens]
    result_parts = []

    for token in raw_tokens:
        word = token.word
        if word not in merged_surfaces:
            result_parts.append(f"{MECAB_COLOR}{word}{RESET_COLOR}")
        else:
            result_parts.append(word)

    return f"{SEPARATOR_COLOR}|{RESET_COLOR}".join(result_parts)


def format_merged_tokens_colored(merged_tokens, raw_tokens) -> str:
    """Format merged tokens with color highlighting for tokens that were merged."""
    raw_surfaces = [t.word for t in raw_tokens]
    result_parts = []

    for token in merged_tokens:
        surface = token.surface
        if surface not in raw_surfaces:
            result_parts.append(f"{MERGED_COLOR}{surface}{RESET_COLOR}")
        else:
            result_parts.append(surface)

    return f"{SEPARATOR_COLOR}|{RESET_COLOR}".join(result_parts)


def tokenize_with_yomitan(text: str, scan_length: int = 20, debug: bool = False) -> list[str] | None:
    """Call local Yomitan API to tokenize text."""
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

        if debug:
            print(f"DEBUG: Yomitan response: {json.dumps(result, indent=2, ensure_ascii=False)}")

        tokens = []
        if result and len(result) > 0:
            content = result[0].get("content", [])
            for segment in content:
                if isinstance(segment, dict):
                    tokens.append(segment.get("text", ""))
                elif isinstance(segment, list):
                    token_text = ""
                    for item in segment:
                        if isinstance(item, dict):
                            token_text += item.get("text", "")
                    if token_text:
                        tokens.append(token_text)

        return tokens if tokens else None

    except urllib.error.URLError as e:
        if debug:
            print(f"DEBUG: URLError: {e}")
        return None
    except Exception as e:
        if debug:
            print(f"DEBUG: Exception: {e}")
        return None


def main():
    mecab = MecabController()

    print("=" * 80)
    print("MeCab Token Merging Comparison")
    print("=" * 80)
    print()

    yomitan_result = tokenize_with_yomitan("テスト")
    yomitan_available = yomitan_result is not None
    if not yomitan_available:
        print(f"Note: Yomitan API not available at {YOMITAN_API_URL}")
        print("      Make sure yomitan_api.py server is running")
        print()

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for entry in TEST_SENTENCES:
        if isinstance(entry, tuple):
            sentence, expected = entry
        else:
            sentence = entry
            expected = None

        raw_tokens = mecab.translate(sentence, mimic_yomitan=False)
        merged_tokens = mecab.translate(sentence, mimic_yomitan=True)

        raw_str_colored = format_mecab_tokens_colored(raw_tokens, merged_tokens)
        merged_str_colored = format_merged_tokens_colored(merged_tokens, raw_tokens)
        merged_str_plain = "|".join(t.surface for t in merged_tokens)

        label_width = 10

        print(f"Sentence: {sentence}")
        if yomitan_available:
            yomitan_tokens = tokenize_with_yomitan(sentence)
            if yomitan_tokens:
                yomitan_str = f"{SEPARATOR_COLOR}|{RESET_COLOR}".join(yomitan_tokens)
                print(f"{'yomitan':>{label_width}}: {yomitan_str}")
        print(f"{'mecab':>{label_width}}: {raw_str_colored}")
        print(f"{'merged':>{label_width}}: {merged_str_colored}", end=" ")

        merge_info = ""
        if len(merged_tokens) < len(raw_tokens):
            diff = len(raw_tokens) - len(merged_tokens)
            merge_info = f"({diff} merge{'s' if diff > 1 else ''})"
        else:
            merge_info = "(0 merges)"

        if expected is not None:
            if merged_str_plain == expected:
                print(f"{merge_info} {PASS_COLOR}✓ PASS{RESET_COLOR}")
                pass_count += 1
            else:
                print(f"{merge_info} {FAIL_COLOR}✗ FAIL{RESET_COLOR}")
                print(f"{'expected':>{label_width}}: {expected}")
                fail_count += 1
        else:
            print(merge_info)
            skip_count += 1
        if isinstance(entry, tuple):
            print(f" {'expected:':>{label_width}} " + f"{MERGED_COLOR}{SEPARATOR_COLOR}|{RESET_COLOR}".join(entry[1].split("|")) + RESET_COLOR)

        print()

    print("=" * 80)
    total_checked = pass_count + fail_count
    if total_checked > 0:
        print(f"Results: {PASS_COLOR}{pass_count} passed{RESET_COLOR}, {FAIL_COLOR}{fail_count} failed{RESET_COLOR}, {skip_count} skipped")
    print("=" * 80)


if __name__ == "__main__":
    main()
