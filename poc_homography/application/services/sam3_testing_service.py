"""SAM3 prompt testing service for comparing segmentation prompts."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime

import cv2
import numpy as np
import numpy.typing as npt

from poc_homography.domain.protocols import Sam3ApiError, Sam3Client
from poc_homography.domain.vo.mask import Mask
from poc_homography.types import Unitless

# Default prompts to test (from issue #105)
DEFAULT_PROMPTS = [
    "road markings",
    "road lines",
    "lane markings",
    "white road paint",
    "pavement markings",
]


@dataclass(frozen=True)
class Sam3PromptResult:
    """Result of testing a single prompt with SAM3.

    This DTO contains metrics for comparing prompt effectiveness.

    Attributes:
        prompt: The text prompt that was tested.
        detections: Number of detections returned by SAM3.
        polygons: Total number of polygons across all detections.
        coverage: Percentage of image covered by detections (0.0-100.0).
        avg_confidence: Average confidence score across detections.
        min_confidence: Minimum confidence score.
        max_confidence: Maximum confidence score.
        error: Error message if the test failed, None otherwise.
    """

    prompt: str
    detections: int
    polygons: int
    coverage: float
    avg_confidence: Unitless
    min_confidence: Unitless
    max_confidence: Unitless
    error: str | None = None


class Sam3PromptTestingService:
    """Service for testing SAM3 prompts and comparing results.

    This service orchestrates the testing of multiple prompts against
    an image using the SAM3 API, calculating metrics for comparison.

    Attributes:
        client: SAM3 API client for making segmentation requests.
    """

    def __init__(self, client: Sam3Client) -> None:
        """Initialize the testing service.

        Args:
            client: SAM3 API client for making segmentation requests.
        """
        self._client = client

    def test_prompts(
        self,
        image_path: Path,
        prompts: list[str] | None = None,
        output_dir: Path | None = None,
        verbose: bool = True,
    ) -> list[Sam3PromptResult]:
        """Test multiple prompts on an image and return comparison results.

        Args:
            image_path: Path to the image file to test.
            prompts: List of prompts to test. Defaults to DEFAULT_PROMPTS.
            output_dir: Optional directory to save mask/overlay images.
            verbose: Whether to print progress messages.

        Returns:
            List of Sam3PromptResult for each tested prompt.

        Raises:
            RuntimeError: If the image cannot be loaded or encoded.
        """
        if prompts is None:
            prompts = DEFAULT_PROMPTS

        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        if verbose:
            print(f"Loaded image: {image_path}")
            print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")

        # Encode image to base64
        image_base64 = self._encode_image(frame)

        # Create output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Test each prompt
        results: list[Sam3PromptResult] = []
        for i, prompt in enumerate(prompts, 1):
            if verbose:
                print(f"\n[{i}/{len(prompts)}] Testing prompt: '{prompt}'")

            result = self._test_single_prompt(
                prompt=prompt,
                image_base64=image_base64,
                frame=frame,
                output_dir=output_dir,
                verbose=verbose,
            )
            results.append(result)

        return results

    def _encode_image(self, frame: npt.NDArray) -> str:
        """Encode an image frame to base64.

        Args:
            frame: Image as numpy array (BGR format).

        Returns:
            Base64-encoded string of the JPEG image.

        Raises:
            RuntimeError: If encoding fails.
        """
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to encode image")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _test_single_prompt(
        self,
        prompt: str,
        image_base64: str,
        frame: npt.NDArray,
        output_dir: Path | None,
        verbose: bool,
    ) -> Sam3PromptResult:
        """Test a single prompt and return the result.

        Args:
            prompt: Text prompt to test.
            image_base64: Base64-encoded image.
            frame: Original image frame (for saving overlays).
            output_dir: Optional directory for saving outputs.
            verbose: Whether to print progress.

        Returns:
            Sam3PromptResult with metrics or error.
        """
        try:
            detections = self._client.segment(image_base64, prompt)

            if not detections:
                return Sam3PromptResult(
                    prompt=prompt,
                    detections=0,
                    polygons=0,
                    coverage=0.0,
                    avg_confidence=Unitless(0.0),
                    min_confidence=Unitless(0.0),
                    max_confidence=Unitless(0.0),
                    error=None,
                )

            # Collect all polygons and confidence scores
            all_polygons: list[list[tuple[int, int]]] = []
            confidence_scores: list[float] = []

            for detection in detections:
                confidence_scores.append(float(detection.confidence))
                for polygon in detection.polygons:
                    all_polygons.append(list(polygon))

            # Create mask and calculate coverage
            height, width = frame.shape[:2]
            mask = Mask.from_polygons(all_polygons, (height, width))

            # Calculate statistics
            avg_confidence = Unitless(
                float(np.mean(confidence_scores)) if confidence_scores else 0.0
            )
            min_confidence = Unitless(min(confidence_scores) if confidence_scores else 0.0)
            max_confidence = Unitless(max(confidence_scores) if confidence_scores else 0.0)

            result = Sam3PromptResult(
                prompt=prompt,
                detections=len(detections),
                polygons=len(all_polygons),
                coverage=mask.coverage,
                avg_confidence=avg_confidence,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                error=None,
            )

            if verbose:
                print(
                    f"   Detections: {result.detections}, "
                    f"Polygons: {result.polygons}, "
                    f"Coverage: {result.coverage:.2f}%"
                )

            # Save outputs if requested
            if output_dir:
                self._save_outputs(prompt, mask, frame, output_dir, verbose)

            return result

        except Sam3ApiError as e:
            error_msg = str(e)
            if verbose:
                print(f"   Error: {error_msg}")
            return Sam3PromptResult(
                prompt=prompt,
                detections=0,
                polygons=0,
                coverage=0.0,
                avg_confidence=Unitless(0.0),
                min_confidence=Unitless(0.0),
                max_confidence=Unitless(0.0),
                error=error_msg,
            )

        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"   Error: {error_msg}")
            return Sam3PromptResult(
                prompt=prompt,
                detections=0,
                polygons=0,
                coverage=0.0,
                avg_confidence=Unitless(0.0),
                min_confidence=Unitless(0.0),
                max_confidence=Unitless(0.0),
                error=error_msg,
            )

    def _save_outputs(
        self,
        prompt: str,
        mask: Mask,
        frame: npt.NDArray,
        output_dir: Path,
        verbose: bool,
    ) -> None:
        """Save mask and overlay images.

        Args:
            prompt: Prompt name (for filename).
            mask: Binary mask to save.
            frame: Original image frame.
            output_dir: Directory to save files.
            verbose: Whether to print progress.
        """
        safe_prompt = prompt.replace(" ", "_")

        # Save mask
        mask_path = output_dir / f"mask_{safe_prompt}.png"
        cv2.imwrite(str(mask_path), mask.data)
        if verbose:
            print(f"   Saved mask: {mask_path}")

        # Save overlay
        overlay = frame.copy()
        overlay[mask.data > 0] = [0, 255, 0]  # Green overlay
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        overlay_path = output_dir / f"overlay_{safe_prompt}.png"
        cv2.imwrite(str(overlay_path), blended)
