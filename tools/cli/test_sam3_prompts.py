#!/usr/bin/env python3
"""CLI for SAM3 prompt testing tool."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.test_sam3_prompts import print_results_table, save_results_json, test_prompts


def main():
    parser = argparse.ArgumentParser(description="Test SAM3 prompts for road marking detection")
    parser.add_argument("image", help="Path to test image")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./prompt_test_results",
        help="Directory to save mask images (default: ./prompt_test_results)",
    )
    parser.add_argument("--json", "-j", help="Save results to JSON file")
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY environment variable not set")
        sys.exit(1)

    # Run tests
    results = test_prompts(args.image, api_key, args.output_dir)

    # Print results table
    print_results_table(results)

    # Save JSON if requested
    if args.json:
        save_results_json(results, args.json)
    elif args.output_dir:
        json_path = os.path.join(args.output_dir, "results.json")
        save_results_json(results, json_path)


if __name__ == "__main__":
    main()
