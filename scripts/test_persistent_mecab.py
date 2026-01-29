#!/usr/bin/env python3
"""
Performance comparison between Persistent and Basic MeCab controllers.

Runs furigana translations on each controller and compares:
- Total time
- Average time per translation
- Speedup factor
"""

import argparse
import sys
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from GameSentenceMiner.mecab.mecab_controller import MecabController


# Sample Japanese sentences for testing
TEST_SENTENCES = [
    "カリン、自分でまいた種は自分で刈り取れ",
    "昨日、林檎を2個買った。",
    "真莉、大好きだよん",
    "彼二千三百六十円も使った。",
    "千葉",
    "昨日すき焼きを食べました",
    "二人の美人",
    "詳細はお気軽にお問い合わせ下さい。",
    "粗末な家に住んでいる",
    "向けていた目",
    "軽そうに見える",
    "相合い傘",
    "放っておけない",
    "放っておいて",
    "有り難う",
    "プールから出て",
    "一人暮らし",
    "今日は",
    "いい気分に当たって",
    "助からない。",
    "乗り込え",
    "ほほ笑む",
    "歩いた",
    "荒んだ",
    "温玉",
    "他人のアソコ弄ってる",
    "拗らせる,拗らせちゃった",
    "打付ける,打付けた",
    "遣る方無い",
    "私は日本語を勉強しています",
    "東京は大きい都市です",
    "明日は雨が降るでしょう",
    "この本はとても面白いです",
    "彼女は美しい花を買いました",
    "電車で学校に行きます",
    "友達と映画を見ました",
    "日本の食べ物は美味しいです",
    "来週、京都に旅行します",
    "猫が窓の外を見ています",
    "新しいコンピューターを買いたい",
]

DEFAULT_ITERATIONS = 10000


def run_benchmark(controller: MecabController, controller_name: str, num_iterations: int) -> dict:
    """Run the benchmark for a given controller."""
    print(f"\n{'='*60}")
    print(f"Running benchmark for: {controller_name}")
    print(f"{'='*60}")

    times = []
    num_sentences = len(TEST_SENTENCES)

    # Warm up (1 call to ensure process is started for persistent)
    _ = controller.reading(TEST_SENTENCES[0])

    start_total = time.perf_counter()
    progress_interval = max(1, num_iterations // 10)

    for i in range(num_iterations):
        sentence = TEST_SENTENCES[i % num_sentences]

        start = time.perf_counter()
        _ = controller.reading(sentence)
        end = time.perf_counter()

        times.append(end - start)

        # Progress indicator
        if (i + 1) % progress_interval == 0:
            print(f"  Completed {i + 1}/{num_iterations} iterations...")

    end_total = time.perf_counter()
    total_time = end_total - start_total
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    results = {
        "name": controller_name,
        "total_time": total_time,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "iterations": num_iterations,
    }

    print(f"\nResults for {controller_name}:")
    print(f"  Total time:   {total_time:.4f} seconds")
    print(f"  Average time: {avg_time*1000:.4f} ms per translation")
    print(f"  Min time:     {min_time*1000:.4f} ms")
    print(f"  Max time:     {max_time*1000:.4f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Performance comparison between Persistent and Basic MeCab controllers."
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of translations to run (default: {DEFAULT_ITERATIONS})"
    )
    args = parser.parse_args()

    num_iterations = args.iterations

    print("="*60)
    print("MeCab Controller Performance Comparison")
    print(f"Running {num_iterations} furigana translations on each controller")
    print("="*60)

    # Test Basic Controller (spawns new process each time)
    print("\nInitializing Basic MeCab Controller...")
    basic_controller = MecabController(persistent=False, cache_max_size=0)
    basic_results = run_benchmark(basic_controller, "Basic Controller", num_iterations)

    # Test Persistent Controller (keeps process alive)
    print("\nInitializing Persistent MeCab Controller...")
    persistent_controller = MecabController(persistent=True, cache_max_size=0)
    persistent_results = run_benchmark(persistent_controller, "Persistent Controller", num_iterations)

    # Cleanup
    persistent_controller.shutdown()

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    speedup = basic_results["total_time"] / persistent_results["total_time"]
    time_saved = basic_results["total_time"] - persistent_results["total_time"]
    avg_speedup = basic_results["avg_time"] / persistent_results["avg_time"]

    print(f"\n{'Metric':<25} {'Basic':>15} {'Persistent':>15} {'Speedup':>10}")
    print("-"*65)
    print(f"{'Total time (s)':<25} {basic_results['total_time']:>15.4f} {persistent_results['total_time']:>15.4f} {speedup:>9.2f}x")
    print(f"{'Avg time (ms)':<25} {basic_results['avg_time']*1000:>15.4f} {persistent_results['avg_time']*1000:>15.4f} {avg_speedup:>9.2f}x")
    print(f"{'Min time (ms)':<25} {basic_results['min_time']*1000:>15.4f} {persistent_results['min_time']*1000:>15.4f}")
    print(f"{'Max time (ms)':<25} {basic_results['max_time']*1000:>15.4f} {persistent_results['max_time']*1000:>15.4f}")

    print(f"\nTime saved: {time_saved:.4f} seconds ({time_saved/basic_results['total_time']*100:.1f}%)")
    print(f"Persistent controller is {speedup:.2f}x faster than Basic controller")

    if speedup > 1:
        print("\nConclusion: Persistent controller significantly outperforms Basic controller!")
    else:
        print("\nConclusion: Results are similar or Basic is faster (unexpected).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
