"""Survey automation for lens distortion calibration.

This module provides tools for automating camera image capture via the Camera Survey Tool,
either through MCP Playwright browser automation or direct API calls.

MCP Playwright Workflow:
========================

For Claude to automate the Camera Survey Tool via MCP Playwright, follow these steps:

1. Navigate to the Camera Survey Tool:
   mcp__playwright__browser_navigate(url="http://localhost:8000/camera-survey/")

2. Get page snapshot to identify UI elements:
   mcp__playwright__browser_snapshot()

3. Select tenant from dropdown (click to open, then click tenant):
   mcp__playwright__browser_click(element="tenant dropdown", ref="<ref_from_snapshot>")
   mcp__playwright__browser_click(element="target tenant option", ref="<ref_from_snapshot>")

4. Select camera from dropdown:
   mcp__playwright__browser_click(element="camera dropdown", ref="<ref_from_snapshot>")
   mcp__playwright__browser_click(element="target camera option", ref="<ref_from_snapshot>")

5. Configure survey parameters (axis, start, end, step):
   Use mcp__playwright__browser_click or mcp__playwright__browser_type for form fields

6. Start survey:
   mcp__playwright__browser_click(element="Start Survey button", ref="<ref_from_snapshot>")

7. Monitor progress by polling snapshot or waiting for completion:
   mcp__playwright__browser_wait_for(text="Survey Complete")
   OR poll mcp__playwright__browser_snapshot() until status shows completion

Direct API Usage:
=================

For programmatic access without browser automation, use the SurveyAutomation class
which calls the Camera Survey Tool's REST API directly:

    from poc_homography.calibration.lens_distortion.survey_automation import SurveyAutomation

    automation = SurveyAutomation(base_url="http://localhost:8000")

    # List available tenants
    tenants = automation.get_tenants()

    # List cameras for a tenant
    cameras = automation.get_cameras(tenant_id="tenant1")

    # Start a pan survey
    session_id = automation.start_survey(
        tenant_id="tenant1",
        camera_id="camera1",
        axis="pan",
        start=0,
        end=360,
        step=15,
        session_tags=["calibration", "parking_lines"]
    )

    # Wait for completion (polls status API)
    result = automation.wait_for_completion(session_id, timeout_seconds=600)

    # Get session details including image paths
    session = automation.get_session(session_id)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests


class SurveyAxis(str, Enum):
    """Survey axis options."""

    PAN = "pan"
    TILT = "tilt"
    ZOOM = "zoom"


class SurveyStatus(str, Enum):
    """Survey status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class SurveyProgress:
    """Current survey progress."""

    session_id: str
    status: SurveyStatus
    step_count: int
    total_steps: int
    current_ptz: dict[str, float] | None
    last_capture_path: str | None
    error_message: str | None = None


@dataclass
class SurveySession:
    """Completed survey session info."""

    session_id: str
    tenant_id: str
    camera_id: str
    axis: str
    start: float
    end: float
    step: float
    status: SurveyStatus
    captures: list[dict[str, Any]]
    session_path: str | None = None
    error_message: str | None = None


@dataclass
class TenantInfo:
    """Tenant information."""

    id: str
    name: str
    description: str = ""


@dataclass
class CameraInfo:
    """Camera information."""

    id: str
    name: str
    ip: str
    model: str | None = None


class SurveyAutomationError(Exception):
    """Error during survey automation."""

    pass


class SurveyAutomation:
    """Automate camera survey for lens calibration.

    This class provides a programmatic interface to the Camera Survey Tool's
    REST API, enabling automated image capture without browser automation.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize survey automation.

        Args:
            base_url: Base URL of the webapp server
        """
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _api_url(self, endpoint: str) -> str:
        """Build API URL."""
        return f"{self.base_url}/camera-survey{endpoint}"

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Make API request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional requests arguments

        Returns:
            Response data dict

        Raises:
            SurveyAutomationError: If request fails or returns error status
        """
        url = self._api_url(endpoint)
        try:
            response = self._session.request(method, url, **kwargs)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "error":
                raise SurveyAutomationError(data.get("message", "Unknown API error"))
            return data.get("data", data)
        except requests.RequestException as e:
            raise SurveyAutomationError(f"API request failed: {e}") from e

    def get_tenants(self) -> list[TenantInfo]:
        """Get list of available tenants.

        Returns:
            List of TenantInfo objects
        """
        data = self._request("GET", "/api/tenants/")
        return [
            TenantInfo(
                id=t["id"],
                name=t["name"],
                description=t.get("description", ""),
            )
            for t in data.get("tenants", [])
        ]

    def get_cameras(self, tenant_id: str) -> list[CameraInfo]:
        """Get list of cameras for a tenant.

        Args:
            tenant_id: Tenant ID to filter cameras

        Returns:
            List of CameraInfo objects
        """
        data = self._request("GET", f"/api/cameras/?tenant_id={tenant_id}")
        return [
            CameraInfo(
                id=c["id"],
                name=c["name"],
                ip=c["ip"],
                model=c.get("model"),
            )
            for c in data.get("cameras", [])
        ]

    def start_survey(
        self,
        tenant_id: str,
        camera_id: str,
        axis: str | SurveyAxis,
        start: float,
        end: float,
        step: float,
        restore_ptz: bool = True,
        retry_timeout: int = 60,
        session_tags: list[str] | None = None,
    ) -> str:
        """Start a new camera survey.

        Args:
            tenant_id: Tenant ID
            camera_id: Camera ID
            axis: Survey axis ("pan", "tilt", or "zoom")
            start: Start value for the axis
            end: End value for the axis
            step: Step size
            restore_ptz: Whether to restore PTZ after survey
            retry_timeout: Timeout for camera operations in seconds
            session_tags: Optional tags for the session

        Returns:
            Session ID string

        Raises:
            SurveyAutomationError: If survey fails to start
        """
        if isinstance(axis, SurveyAxis):
            axis = axis.value

        payload = {
            "tenant_id": tenant_id,
            "camera_id": camera_id,
            "axis": axis,
            "start": start,
            "end": end,
            "step": step,
            "restore_ptz": restore_ptz,
            "retry_timeout": retry_timeout,
            "session_tags": session_tags or [],
        }

        data = self._request("POST", "/api/survey/start/", json=payload)
        return data["session_id"]

    def get_survey_status(self, session_id: str) -> SurveyProgress:
        """Get current survey progress.

        Args:
            session_id: Survey session ID

        Returns:
            SurveyProgress object
        """
        data = self._request("GET", f"/api/survey/{session_id}/status/")
        return SurveyProgress(
            session_id=data.get("session_id", session_id),
            status=SurveyStatus(data.get("status", "pending")),
            step_count=data.get("step_count", 0),
            total_steps=data.get("total_steps", 0),
            current_ptz=data.get("current_ptz"),
            last_capture_path=data.get("last_capture_path"),
            error_message=data.get("error"),
        )

    def wait_for_completion(
        self,
        session_id: str,
        timeout_seconds: int = 600,
        poll_interval: float = 2.0,
        progress_callback: Any | None = None,
    ) -> SurveyProgress:
        """Wait for survey to complete.

        Args:
            session_id: Survey session ID
            timeout_seconds: Maximum time to wait
            poll_interval: Seconds between status checks
            progress_callback: Optional callback(progress: SurveyProgress) called on each poll

        Returns:
            Final SurveyProgress object

        Raises:
            SurveyAutomationError: If survey fails or times out
        """
        start_time = time.time()
        terminal_statuses = {SurveyStatus.COMPLETED, SurveyStatus.FAILED, SurveyStatus.ABORTED}

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise SurveyAutomationError(
                    f"Survey timed out after {timeout_seconds}s (session: {session_id})"
                )

            progress = self.get_survey_status(session_id)

            if progress_callback:
                progress_callback(progress)

            if progress.status in terminal_statuses:
                if progress.status == SurveyStatus.FAILED:
                    raise SurveyAutomationError(
                        f"Survey failed: {progress.error_message or 'Unknown error'}"
                    )
                if progress.status == SurveyStatus.ABORTED:
                    raise SurveyAutomationError("Survey was aborted")
                return progress

            time.sleep(poll_interval)

    def abort_survey(self, session_id: str) -> None:
        """Abort a running survey.

        Args:
            session_id: Survey session ID
        """
        self._request("POST", f"/api/survey/{session_id}/abort/")

    def get_session(self, session_id: str) -> SurveySession:
        """Get complete session details.

        Args:
            session_id: Survey session ID

        Returns:
            SurveySession object with all captures
        """
        data = self._request("GET", f"/api/survey/sessions/{session_id}/")
        return SurveySession(
            session_id=data.get("session_id", session_id),
            tenant_id=data.get("tenant_id", ""),
            camera_id=data.get("camera_id", ""),
            axis=data.get("axis", ""),
            start=data.get("start", 0),
            end=data.get("end", 0),
            step=data.get("step", 0),
            status=SurveyStatus(data.get("status", "pending")),
            captures=data.get("captures", []),
            session_path=data.get("session_path"),
            error_message=data.get("error"),
        )

    def list_sessions(self, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        """List survey sessions.

        Args:
            limit: Maximum number to return
            offset: Number to skip

        Returns:
            Tuple of (sessions list, total count)
        """
        data = self._request("GET", f"/api/survey/sessions/?limit={limit}&offset={offset}")
        return data.get("sessions", []), data.get("total", 0)

    def delete_session(self, session_id: str) -> None:
        """Delete a survey session.

        Args:
            session_id: Survey session ID
        """
        self._request("POST", f"/api/survey/sessions/{session_id}/delete/")


# MCP Playwright workflow reference
MCP_PLAYWRIGHT_WORKFLOW = """
# MCP Playwright Workflow for Camera Survey Automation

This document describes how Claude can automate the Camera Survey Tool using MCP Playwright.

## Prerequisites
- Webapp server running at http://localhost:8000
- MCP Playwright server connected
- Camera configured and accessible

## Step-by-Step Workflow

### Step 1: Navigate to Camera Survey Tool
```
mcp__playwright__browser_navigate(url="http://localhost:8000/camera-survey/")
```

### Step 2: Take Page Snapshot
```
mcp__playwright__browser_snapshot()
```
This returns the accessibility tree with element references (ref values) for interaction.

### Step 3: Select Tenant
From the snapshot, identify the tenant dropdown and tenant options:
```
mcp__playwright__browser_click(element="tenant dropdown", ref="<tenant_dropdown_ref>")
```
Then take another snapshot to see tenant options:
```
mcp__playwright__browser_snapshot()
```
Click the desired tenant:
```
mcp__playwright__browser_click(element="tenant name", ref="<tenant_option_ref>")
```

### Step 4: Select Camera
After tenant selection, cameras load automatically. Take snapshot:
```
mcp__playwright__browser_snapshot()
```
Click camera dropdown and select camera:
```
mcp__playwright__browser_click(element="camera dropdown", ref="<camera_dropdown_ref>")
mcp__playwright__browser_snapshot()
mcp__playwright__browser_click(element="camera name", ref="<camera_option_ref>")
```

### Step 5: Configure Survey Parameters
The survey form has these fields:
- Axis: Radio buttons for "pan", "tilt", "zoom"
- Start: Number input
- End: Number input
- Step: Number input

For a full pan survey at parking areas:
```
mcp__playwright__browser_click(element="pan radio button", ref="<pan_radio_ref>")
mcp__playwright__browser_type(element="start input", ref="<start_ref>", text="0")
mcp__playwright__browser_type(element="end input", ref="<end_ref>", text="360")
mcp__playwright__browser_type(element="step input", ref="<step_ref>", text="15")
```

### Step 6: Add Session Tags (Optional)
```
mcp__playwright__browser_type(element="tags input", ref="<tags_ref>", text="calibration,parking_lines")
```

### Step 7: Start Survey
```
mcp__playwright__browser_click(element="Start Survey button", ref="<start_button_ref>")
```

### Step 8: Monitor Progress
Option A - Wait for completion text:
```
mcp__playwright__browser_wait_for(text="Survey Complete", timeout=600)
```

Option B - Poll status via snapshots:
```
mcp__playwright__browser_snapshot()
# Check progress bar and status text in snapshot
# Repeat until status shows "completed"
```

### Step 9: Retrieve Results
After completion, the UI shows:
- Session ID
- List of captured images with PTZ values
- Download manifest button

Take a final snapshot to see session details:
```
mcp__playwright__browser_snapshot()
```

## Recommended Survey Presets for Lens Calibration

### Full Pan Coverage (parking lot overview)
- Axis: pan
- Start: 0
- End: 360
- Step: 15
- Purpose: Capture parking areas from all angles

### Tilt Variations (perspective coverage)
- Axis: tilt
- Start: -30
- End: 60
- Step: 10
- Purpose: Different viewing angles for line detection

### Zoom Calibration (distortion at different zooms)
- Axis: zoom
- Start: 1
- End: 10
- Step: 1
- Purpose: Characterize zoom-dependent distortion

## Error Handling

If survey fails:
1. Check snapshot for error messages
2. Verify camera is accessible
3. Check server logs

If UI elements not found:
1. Retake snapshot
2. Verify correct page loaded
3. Check for loading spinners before interacting
"""


def print_mcp_workflow() -> None:
    """Print the MCP Playwright workflow reference."""
    print(MCP_PLAYWRIGHT_WORKFLOW)


# Recommended survey configurations for lens calibration
CALIBRATION_SURVEY_PRESETS = [
    {
        "name": "Full Pan Coverage",
        "description": "Capture parking areas from all angles",
        "axis": "pan",
        "start": 0,
        "end": 360,
        "step": 15,
        "tags": ["calibration", "pan_sweep"],
    },
    {
        "name": "Tilt Variations",
        "description": "Different viewing angles for line detection",
        "axis": "tilt",
        "start": -30,
        "end": 60,
        "step": 10,
        "tags": ["calibration", "tilt_sweep"],
    },
    {
        "name": "Zoom Calibration",
        "description": "Characterize zoom-dependent distortion",
        "axis": "zoom",
        "start": 1,
        "end": 10,
        "step": 1,
        "tags": ["calibration", "zoom_sweep"],
    },
]
