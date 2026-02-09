"""SAM3 API client implementation."""

from __future__ import annotations

from typing import Any

import requests

from poc_homography.domain.protocols import Sam3ApiError, Sam3Detection
from poc_homography.types import Unitless

SAM3_API_URL = "https://serverless.roboflow.com/sam3/concept_segment"
SAM3_REQUEST_TIMEOUT = 120  # seconds


class Sam3ApiClient:
    """Client for the Roboflow SAM3 concept segmentation API.

    This client implements the Sam3Client protocol, providing the actual
    HTTP communication with the Roboflow serverless API.

    Attributes:
        api_key: Roboflow API key for authentication.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize the SAM3 API client.

        Args:
            api_key: Roboflow API key for authentication.
        """
        self._api_key = api_key

    def segment(self, image_base64: str, prompt: str) -> list[Sam3Detection]:
        """Segment an image using the given text prompt.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt describing what to segment (e.g., "road markings").

        Returns:
            List of detections, each containing confidence and polygon coordinates.

        Raises:
            Sam3ApiError: If the API call fails.
        """
        response = self._call_api(image_base64, prompt)
        return self._parse_response(response)

    def _call_api(self, image_base64: str, prompt: str) -> dict[str, Any]:
        """Make the API request to SAM3.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt for segmentation.

        Returns:
            Raw API response as dictionary.

        Raises:
            Sam3ApiError: If the API call fails.
        """
        url = f"{SAM3_API_URL}?api_key={self._api_key}"

        request_body = {
            "format": "polygon",
            "image": {"type": "base64", "value": image_base64},
            "prompts": [{"type": "text", "text": prompt}],
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                url,
                json=request_body,
                headers=headers,
                timeout=SAM3_REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            raise Sam3ApiError(f"API request failed: {e}") from e

        if response.status_code != 200:
            raise Sam3ApiError(f"API error: {response.status_code}")

        return response.json()

    def _parse_response(self, api_response: dict[str, Any]) -> list[Sam3Detection]:
        """Parse the API response into Sam3Detection objects.

        Args:
            api_response: Raw API response dictionary.

        Returns:
            List of Sam3Detection objects.
        """
        detections: list[Sam3Detection] = []

        prompt_results = api_response.get("prompt_results", [])
        for prompt_result in prompt_results:
            predictions = prompt_result.get("predictions", [])

            for prediction in predictions:
                confidence = prediction.get("confidence", 0.0)
                masks = prediction.get("masks", [])

                # Convert masks (list of polygon coordinates) to tuples
                polygons: list[tuple[tuple[int, int], ...]] = []
                for polygon in masks:
                    if isinstance(polygon, list) and len(polygon) >= 3:
                        points = tuple(
                            (int(pt[0]), int(pt[1]))
                            for pt in polygon
                            if isinstance(pt, (list, tuple)) and len(pt) >= 2
                        )
                        if len(points) >= 3:
                            polygons.append(points)

                if polygons:
                    detections.append(
                        Sam3Detection(
                            confidence=Unitless(confidence),
                            polygons=tuple(polygons),
                        )
                    )

        return detections
