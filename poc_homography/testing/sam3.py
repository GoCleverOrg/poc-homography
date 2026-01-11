"""
SAM3 Prompt Testing Utilities.

Tests alternative prompts for road marking detection and compares results.
Produces a comparison table as required by issue #105.
"""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from poc_homography.types import Unitless

# Prompts to test as specified in issue #105
PROMPTS_TO_TEST = [
    "road markings",  # Current default
    "road lines",
    "lane markings",
    "white road paint",
    "pavement markings",
]


@dataclass
class PromptTestResult:
    """Result of testing a single prompt."""

    prompt: str
    detections: int
    polygons: int
    coverage: float
    avg_confidence: Unitless
    min_confidence: Unitless
    max_confidence: Unitless
    error: str | None = None


def call_sam3_api(image_base64: str, prompt: str, api_key: str) -> dict[str, Any]:
    """
    Call SAM3 API with given prompt and return parsed response.

    Args:
        image_base64: Base64-encoded image
        prompt: Text prompt for SAM3
        api_key: Roboflow API key

    Returns:
        API response as dictionary
    """
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


def create_mask_from_response(
    api_response: dict[str, Any], frame_shape: tuple[int, ...]
) -> tuple[np.ndarray, int, list[float]]:
    """
    Create binary mask from SAM3 API response.

    Args:
        api_response: Response from SAM3 API
        frame_shape: Shape of the frame (height, width, channels)

    Returns:
        Tuple of (mask, total_polygons, confidence_scores)
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    prompt_results = api_response.get("prompt_results", [])
    total_polygons = 0
    confidence_scores: list[float] = []

    for prompt_result in prompt_results:
        predictions = prompt_result.get("predictions", [])

        for prediction in predictions:
            confidence = prediction.get("confidence", 0)
            confidence_scores.append(confidence)
            masks = prediction.get("masks", [])

            for polygon in masks:
                if isinstance(polygon, list) and len(polygon) >= 3:
                    pts = np.array(
                        [[int(pt[0]), int(pt[1])] for pt in polygon if len(pt) >= 2],
                        dtype=np.int32,
                    )
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], (255,))
                        total_polygons += 1

    return mask, total_polygons, confidence_scores


def calculate_mask_coverage(mask: np.ndarray) -> float:
    """
    Calculate percentage of image covered by mask.

    Args:
        mask: Binary mask (numpy array)

    Returns:
        Coverage percentage (0-100)
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    white_pixels = cv2.countNonZero(mask)
    return (white_pixels / total_pixels) * 100


def test_prompts(
    image_path: Path,
    api_key: str,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> list[PromptTestResult]:
    """
    Test all prompts on the given image and return results.

    Args:
        image_path: Path to image file
        api_key: Roboflow API key
        output_dir: Optional directory to save mask/overlay images
        verbose: Whether to print progress messages

    Returns:
        List of PromptTestResult for each tested prompt

    Raises:
        RuntimeError: If image cannot be loaded or encoded
    """
    # Load image
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    if verbose:
        print(f"Loaded image: {image_path}")
        print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")

    # Encode image
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode image")
    image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: list[PromptTestResult] = []

    for i, prompt in enumerate(PROMPTS_TO_TEST, 1):
        if verbose:
            print(f"\n[{i}/{len(PROMPTS_TO_TEST)}] Testing prompt: '{prompt}'")

        try:
            api_response = call_sam3_api(image_base64, prompt, api_key)

            if "error" in api_response:
                results.append(
                    PromptTestResult(
                        prompt=prompt,
                        detections=0,
                        polygons=0,
                        coverage=0.0,
                        avg_confidence=Unitless(0.0),
                        min_confidence=Unitless(0.0),
                        max_confidence=Unitless(0.0),
                        error=api_response["error"],
                    )
                )
                continue

            mask, total_polygons, confidence_scores = create_mask_from_response(
                api_response, frame.shape
            )
            coverage = calculate_mask_coverage(mask)
            avg_confidence = Unitless(
                float(np.mean(confidence_scores)) if confidence_scores else 0.0
            )
            min_confidence = Unitless(min(confidence_scores) if confidence_scores else 0.0)
            max_confidence = Unitless(max(confidence_scores) if confidence_scores else 0.0)

            result = PromptTestResult(
                prompt=prompt,
                detections=len(confidence_scores),
                polygons=total_polygons,
                coverage=coverage,
                avg_confidence=avg_confidence,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                error=None,
            )
            results.append(result)

            if verbose:
                print(
                    f"   Detections: {result.detections}, "
                    f"Polygons: {result.polygons}, "
                    f"Coverage: {result.coverage:.2f}%"
                )

            # Save mask if output directory specified
            if output_dir:
                mask_filename = f"mask_{prompt.replace(' ', '_')}.png"
                mask_path = output_dir / mask_filename
                cv2.imwrite(str(mask_path), mask)
                if verbose:
                    print(f"   Saved mask: {mask_path}")

                # Also save overlay
                overlay = frame.copy()
                overlay[mask > 0] = [0, 255, 0]  # Green overlay
                blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                overlay_filename = f"overlay_{prompt.replace(' ', '_')}.png"
                overlay_path = output_dir / overlay_filename
                cv2.imwrite(str(overlay_path), blended)

        except Exception as e:
            results.append(
                PromptTestResult(
                    prompt=prompt,
                    detections=0,
                    polygons=0,
                    coverage=0.0,
                    avg_confidence=Unitless(0.0),
                    min_confidence=Unitless(0.0),
                    max_confidence=Unitless(0.0),
                    error=str(e),
                )
            )
            if verbose:
                print(f"   Error: {e}")

    return results
