#!/usr/bin/env python3
"""
SAM3 Prompt Testing Script

Tests alternative prompts for road marking detection and compares results.
Produces a comparison table as required by issue #105.

Usage:
    ROBOFLOW_API_KEY=<key> python3 tools/test_sam3_prompts.py ./Cartografia_valencia.png

Output:
    - Console table with prompt comparison results
    - Mask images saved to ./prompt_test_results/
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import requests

# Prompts to test as specified in issue #105
PROMPTS_TO_TEST = [
    "road markings",  # Current default
    "road lines",
    "lane markings",
    "white road paint",
    "pavement markings",
]


def call_sam3_api(image_base64: str, prompt: str, api_key: str) -> dict:
    """Call SAM3 API with given prompt and return parsed response."""
    api_url = f"https://serverless.roboflow.com/sam3/concept_segment?api_key={api_key}"

    request_body = {
        "format": "polygon",
        "image": {"type": "base64", "value": image_base64},
        "prompts": [{"type": "text", "text": prompt}],
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, json=request_body, headers=headers, timeout=120)

    if response.status_code != 200:
        return {"error": f"API error: {response.status_code}"}

    return response.json()


def create_mask_from_response(api_response: dict, frame_shape: tuple) -> tuple:
    """
    Create binary mask from SAM3 API response.

    Returns:
        (mask, total_polygons, confidence_scores)
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    prompt_results = api_response.get("prompt_results", [])
    total_polygons = 0
    confidence_scores = []

    for prompt_result in prompt_results:
        predictions = prompt_result.get("predictions", [])

        for prediction in predictions:
            confidence = prediction.get("confidence", 0)
            confidence_scores.append(confidence)
            masks = prediction.get("masks", [])

            for polygon in masks:
                if isinstance(polygon, list) and len(polygon) >= 3:
                    pts = np.array(
                        [[int(pt[0]), int(pt[1])] for pt in polygon if len(pt) >= 2], dtype=np.int32
                    )
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], 255)
                        total_polygons += 1

    return mask, total_polygons, confidence_scores


def calculate_mask_coverage(mask: np.ndarray) -> float:
    """Calculate percentage of image covered by mask."""
    total_pixels = mask.shape[0] * mask.shape[1]
    white_pixels = cv2.countNonZero(mask)
    return (white_pixels / total_pixels) * 100


def test_prompts(image_path: str, api_key: str, output_dir: str = None) -> list:
    """
    Test all prompts on the given image and return results.

    Returns:
        List of dicts with test results for each prompt
    """
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    print(f"Loaded image: {image_path}")
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")

    # Encode image
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print("Error: Failed to encode image")
        sys.exit(1)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []

    for i, prompt in enumerate(PROMPTS_TO_TEST, 1):
        print(f"\n[{i}/{len(PROMPTS_TO_TEST)}] Testing prompt: '{prompt}'")

        try:
            api_response = call_sam3_api(image_base64, prompt, api_key)

            if "error" in api_response:
                results.append(
                    {
                        "prompt": prompt,
                        "detections": 0,
                        "polygons": 0,
                        "coverage": 0.0,
                        "avg_confidence": 0.0,
                        "error": api_response["error"],
                    }
                )
                continue

            mask, total_polygons, confidence_scores = create_mask_from_response(
                api_response, frame.shape
            )
            coverage = calculate_mask_coverage(mask)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

            result = {
                "prompt": prompt,
                "detections": len(confidence_scores),
                "polygons": total_polygons,
                "coverage": coverage,
                "avg_confidence": avg_confidence,
                "min_confidence": min(confidence_scores) if confidence_scores else 0.0,
                "max_confidence": max(confidence_scores) if confidence_scores else 0.0,
                "error": None,
            }
            results.append(result)

            print(
                f"   Detections: {result['detections']}, Polygons: {result['polygons']}, Coverage: {result['coverage']:.2f}%"
            )

            # Save mask if output directory specified
            if output_dir:
                mask_filename = f"mask_{prompt.replace(' ', '_')}.png"
                mask_path = os.path.join(output_dir, mask_filename)
                cv2.imwrite(mask_path, mask)
                print(f"   Saved mask: {mask_path}")

                # Also save overlay
                overlay = frame.copy()
                overlay[mask > 0] = [0, 255, 0]  # Green overlay
                blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                overlay_filename = f"overlay_{prompt.replace(' ', '_')}.png"
                overlay_path = os.path.join(output_dir, overlay_filename)
                cv2.imwrite(overlay_path, blended)

        except Exception as e:
            results.append(
                {
                    "prompt": prompt,
                    "detections": 0,
                    "polygons": 0,
                    "coverage": 0.0,
                    "avg_confidence": 0.0,
                    "error": str(e),
                }
            )
            print(f"   Error: {e}")

    return results


def print_results_table(results: list):
    """Print results as a markdown table."""
    print("\n" + "=" * 80)
    print("SAM3 PROMPT COMPARISON RESULTS")
    print("=" * 80)

    # Markdown table header
    print("\n| Prompt | Detections | Polygons | Coverage % | Avg Confidence | Quality Assessment |")
    print("|--------|-----------|----------|------------|----------------|-------------------|")

    # Find best result for comparison
    best_coverage = max((r["coverage"] for r in results if not r.get("error")), default=0)
    best_detections = max((r["detections"] for r in results if not r.get("error")), default=0)

    for result in results:
        if result.get("error"):
            print(f"| {result['prompt']} | ERROR | - | - | - | {result['error'][:30]} |")
            continue

        # Quality assessment
        quality = []
        if result["coverage"] >= best_coverage * 0.9:
            quality.append("High coverage")
        if result["detections"] >= best_detections * 0.9:
            quality.append("Good detection count")
        if result["avg_confidence"] >= 0.7:
            quality.append("High confidence")
        elif result["avg_confidence"] >= 0.5:
            quality.append("Medium confidence")
        else:
            quality.append("Low confidence")

        quality_str = "; ".join(quality) if quality else "N/A"

        print(
            f"| {result['prompt']} | {result['detections']} | {result['polygons']} | {result['coverage']:.2f} | {result['avg_confidence']:.3f} | {quality_str} |"
        )

    print("\n" + "=" * 80)

    # Recommendation
    valid_results = [r for r in results if not r.get("error")]
    if valid_results:
        # Sort by a combined score (coverage * avg_confidence)
        best = max(valid_results, key=lambda r: r["coverage"] * r["avg_confidence"])
        print(f"\nRECOMMENDATION: Best performing prompt is '{best['prompt']}'")
        print(f"  - Coverage: {best['coverage']:.2f}%")
        print(f"  - Detections: {best['detections']}")
        print(f"  - Avg Confidence: {best['avg_confidence']:.3f}")

        if best["prompt"] != "road markings":
            print(f"\nSuggestion: Consider updating DEFAULT_SAM3_PROMPT to '{best['prompt']}'")
        else:
            print("\nThe current default 'road markings' appears to be optimal.")


def save_results_json(results: list, output_path: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "prompts_tested": PROMPTS_TO_TEST,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


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
