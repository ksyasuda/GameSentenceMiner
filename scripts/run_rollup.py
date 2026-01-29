"""
Simple script to force a daily stats rollup.

This script directly imports the daily_rollup module file, bypassing the cron package's
__init__.py to avoid circular import issues.
"""
import sys
import importlib.util
from pathlib import Path

# Get the path to the daily_rollup.py file
script_dir = Path(__file__).parent
daily_rollup_path = script_dir.parent / "GameSentenceMiner" / "util" / "cron" / "daily_rollup.py"

# Load the module directly from the file path
spec = importlib.util.spec_from_file_location("daily_rollup", daily_rollup_path)
daily_rollup = importlib.util.module_from_spec(spec)
sys.modules["daily_rollup"] = daily_rollup
spec.loader.exec_module(daily_rollup)

# Run the daily rollup
result = daily_rollup.run_daily_rollup()

# Print summary
print("\n" + "=" * 80)
print("DAILY ROLLUP SUMMARY")
print("=" * 80)
print(f"Success: {'Yes' if result['success'] else 'No'}")
if result['start_date']:
    print(f"Date range: {result['start_date']} to {result['end_date']}")
print(f"Total dates with data: {result['total_dates']}")
print(f"Successfully processed: {result['processed']}")
print(f"Overwritten: {result['overwritten']}")
print(f"Errors: {result['errors']}")
print(f"Time elapsed: {result['elapsed_time']:.2f} seconds")
if result['error_message']:
    print(f"Error: {result['error_message']}")
print("=" * 80)
