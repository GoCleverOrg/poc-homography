"""Frame capture and management CLI commands."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from typing import TYPE_CHECKING

import typer

from poc_homography.application import ApplicationContext
from poc_homography.cli.main import frame_app
from poc_homography.domain.entities.annotation import Annotation
from poc_homography.domain.vo import PixelPoint, PTZState
from poc_homography.types import Degrees, PixelsFloat, Unitless

if TYPE_CHECKING:
    from poc_homography.calibration.interactive import CalibrationSession
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint

# Check for OpenCV availability without importing (avoids unused import warning)
CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


@frame_app.command("capture")
def capture_command(
    camera: str = typer.Argument(..., help="Camera name (e.g., 'Valte')"),
) -> None:
    """
    Capture a frame from a camera and store it in the repository.

    Captures a full-resolution frame from the specified camera via RTSP,
    records the current PTZ state, and saves both the image and metadata
    to the frames repository.

    The frame ID will be printed on success for use with other commands.

    Example:
        hom frame capture Valte
    """
    ctx = ApplicationContext.default()

    # Find camera config
    all_configs = ctx.repo_camera_config.get_all()
    cam_config = next((c for c in all_configs if c.name == camera), None)

    if not cam_config:
        available = [c.name for c in all_configs]
        typer.echo(
            f"Error: Camera '{camera}' not found. Available: {', '.join(available)}",
            err=True,
        )
        raise typer.Exit(1)

    if not cam_config.ip_address:
        typer.echo(f"Error: Camera '{camera}' has no IP address configured", err=True)
        raise typer.Exit(1)

    # Create controller and capture frame
    typer.echo(f"Connecting to camera '{camera}'...")

    try:
        controller = ctx.camera_controller(cam_config)
        typer.echo("Capturing frame...")

        result = ctx.frame_capture_service.capture_frame(
            camera_config=cam_config,
            controller=controller,
            timestamp=datetime.now(),
        )

        if not result.success:
            typer.echo(f"Error: {result.error}", err=True)
            raise typer.Exit(1)

        assert result.frame is not None
        assert result.image_path is not None

        typer.echo("Frame captured successfully!")
        typer.echo(f"  ID: {result.frame.id}")
        typer.echo(f"  Image: {result.image_path}")
        typer.echo(
            f"  PTZ: pan={result.frame.ptz_state.pan_raw:.1f}, "
            f"tilt={result.frame.ptz_state.tilt_deg:.1f}, "
            f"zoom={result.frame.ptz_state.zoom:.1f}"
        )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@frame_app.command("list")
def list_command(
    camera: str | None = typer.Option(None, "--camera", "-c", help="Filter by camera name"),
    map_id: str | None = typer.Option(None, "--map", "-m", help="Filter by map ID"),
) -> None:
    """
    List captured frames from the repository.

    Shows frame IDs, timestamps, and PTZ state for stored frames.
    Use --camera or --map to filter results.

    Example:
        hom frame list
        hom frame list --camera Valte
        hom frame list --map valte
    """
    ctx = ApplicationContext.default()

    # Get frames based on filters
    if camera and map_id:
        frames = ctx.repo_captured_frame.get_by_camera(map_id, camera)
    elif map_id:
        frames = ctx.repo_captured_frame.get_by_map(map_id)
    elif camera:
        # Need to find the map_id for this camera
        all_configs = ctx.repo_camera_config.get_all()
        cam_config = next((c for c in all_configs if c.name == camera), None)
        if cam_config:
            frames = ctx.repo_captured_frame.get_by_camera(cam_config.map_id, camera)
        else:
            typer.echo(f"Error: Camera '{camera}' not found", err=True)
            raise typer.Exit(1)
    else:
        frames = ctx.repo_captured_frame.get_all()

    if not frames:
        typer.echo("No frames found.")
        raise typer.Exit(0)

    # Sort by timestamp (newest first)
    frames.sort(key=lambda f: f.timestamp, reverse=True)

    typer.echo(f"Found {len(frames)} frame(s):\n")

    for frame in frames:
        typer.echo(f"  {frame.id}")
        typer.echo(f"    Timestamp: {frame.timestamp.isoformat()}")
        typer.echo(
            f"    PTZ: pan={frame.ptz_state.pan_raw:.1f}, "
            f"tilt={frame.ptz_state.tilt_deg:.1f}, "
            f"zoom={frame.ptz_state.zoom:.1f}"
        )
        typer.echo(f"    Image: {frame.image_path}")
        typer.echo()


@frame_app.command("show")
def show_command(
    frame_id: str = typer.Argument(..., help="Frame ID (e.g., 'valte/Valte/2024-01-15T10-30-45')"),
) -> None:
    """
    Show details of a specific captured frame.

    Example:
        hom frame show valte/Valte/2024-01-15T10-30-45
    """
    ctx = ApplicationContext.default()

    frame = ctx.repo_captured_frame.get(frame_id)

    if not frame:
        typer.echo(f"Error: Frame '{frame_id}' not found", err=True)
        raise typer.Exit(1)

    image_path = ctx.repo_captured_frame.get_image_path(frame)

    typer.echo(f"Frame: {frame.id}")
    typer.echo(f"  Map ID: {frame.map_id}")
    typer.echo(f"  Camera: {frame.camera_name}")
    typer.echo(f"  Timestamp: {frame.timestamp.isoformat()}")
    typer.echo("  PTZ State:")
    typer.echo(f"    Pan (raw): {frame.ptz_state.pan_raw:.1f}")
    typer.echo(f"    Tilt: {frame.ptz_state.tilt_deg:.1f}")
    typer.echo(f"    Zoom: {frame.ptz_state.zoom:.1f}")
    typer.echo(f"  Image Path: {image_path}")
    typer.echo(f"  Image Exists: {image_path.exists()}")


@frame_app.command("delete")
def delete_command(
    frame_id: str = typer.Argument(..., help="Frame ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """
    Delete a captured frame and its image.

    Example:
        hom frame delete valte/Valte/2024-01-15T10-30-45
        hom frame delete valte/Valte/2024-01-15T10-30-45 --force
    """
    ctx = ApplicationContext.default()

    frame = ctx.repo_captured_frame.get(frame_id)

    if not frame:
        typer.echo(f"Error: Frame '{frame_id}' not found", err=True)
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete frame '{frame_id}' and its image?")
        if not confirm:
            typer.echo("Cancelled.")
            raise typer.Exit(0)

    if ctx.repo_captured_frame.delete(frame_id):
        typer.echo(f"Deleted frame: {frame_id}")
    else:
        typer.echo("Error: Failed to delete frame", err=True)
        raise typer.Exit(1)


@frame_app.command("annotations")
def annotations_command(
    frame_id: str = typer.Argument(..., help="Frame ID (e.g., 'valte/Valte/2024-01-15T10-30-45')"),
) -> None:
    """
    List annotations for a captured frame.

    Shows GCP observations (annotations) stored for a frame, including
    GCP ID, pixel coordinates, and PTZ state when observed.

    Example:
        hom frame annotations valte/Valte/2024-01-15T10-30-45
    """
    ctx = ApplicationContext.default()

    frame = ctx.repo_captured_frame.get(frame_id)
    if not frame:
        typer.echo(f"Error: Frame '{frame_id}' not found", err=True)
        raise typer.Exit(1)

    annotations = ctx.repo_captured_frame.get_annotations(frame_id)

    if not annotations:
        typer.echo(f"No annotations found for frame: {frame_id}")
        raise typer.Exit(0)

    typer.echo(f"Annotations for frame: {frame_id}")
    typer.echo(f"Found {len(annotations)} annotation(s):\n")

    for i, ann in enumerate(annotations, 1):
        typer.echo(f"  [{i}] GCP: {ann.gcp_id}")
        typer.echo(f"      Pixel: ({ann.pixel.x:.1f}, {ann.pixel.y:.1f})")
        typer.echo(
            f"      PTZ: pan={ann.camera_pose.pan_raw:.1f}, "
            f"tilt={ann.camera_pose.tilt_deg:.1f}, "
            f"zoom={ann.camera_pose.zoom:.1f}"
        )
        typer.echo()


@frame_app.command("annotate")
def annotate_command(
    frame_id: str = typer.Argument(..., help="Frame ID (e.g., 'valte/Valte/2024-01-15T10-30-45')"),
    height: float = typer.Option(10.0, help="Initial camera height estimate in meters"),
    pan_offset: float = typer.Option(0.0, help="Initial pan offset estimate in degrees"),
) -> None:
    """
    Interactively annotate GCPs in a captured frame.

    Opens an interactive GUI where you can click on GCP locations in the frame
    and enter their IDs. Annotations are saved to the repository when you're done.

    Controls:
      - Click: Mark a GCP location
      - Enter GCP ID when prompted (e.g., Z1, P5)
      - C: Run calibration with current points
      - A: Save annotations to repository
      - S: Save calibration results to JSON
      - R: Reset all points
      - Q/ESC: Quit

    Example:
        hom frame annotate valte/Valte/2024-01-15T10-30-45
    """
    if not CV2_AVAILABLE:
        typer.echo(
            "Error: OpenCV is required for interactive mode but is not installed",
            err=True,
        )
        typer.echo("Install with: uv pip install opencv-python", err=True)
        raise typer.Exit(1)

    # Import here to avoid issues when CV2 is not available
    import cv2 as cv2_lib
    import numpy as np

    from poc_homography.calibration.interactive import (
        CalibrationSession,
    )
    from poc_homography.types import Degrees, Meters, Millimeters, Unitless

    ctx = ApplicationContext.default()

    # Load from CapturedFrame repository
    captured_frame = ctx.repo_captured_frame.get(frame_id)
    if captured_frame is None:
        typer.echo(f"Error: Frame '{frame_id}' not found in repository", err=True)
        raise typer.Exit(1)

    # Get image path and load
    image_path = ctx.repo_captured_frame.get_image_path(captured_frame)
    if not image_path.exists():
        typer.echo(f"Error: Image file not found: {image_path}", err=True)
        raise typer.Exit(1)

    frame_data = cv2_lib.imread(str(image_path))
    if frame_data is None:
        typer.echo(f"Error: Failed to load frame image: {image_path}", err=True)
        raise typer.Exit(1)

    # Use values from captured frame
    camera_name = captured_frame.camera_name
    map_id = captured_frame.map_id
    pan_raw_value = Degrees(captured_frame.ptz_state.pan_raw)
    tilt_value = Degrees(captured_frame.ptz_state.tilt_deg)
    zoom_value = Unitless(captured_frame.ptz_state.zoom)

    typer.echo(f"Loaded frame: {frame_id}")
    typer.echo(f"  Camera: {camera_name}, Map: {map_id}")
    typer.echo(f"  PTZ: pan={pan_raw_value:.1f}, tilt={tilt_value:.1f}, zoom={zoom_value:.1f}")

    # Ensure the frame is uint8
    frame = np.asarray(frame_data, dtype=np.uint8)

    # Load GCPs from repository based on map_id
    map_registry: dict[str, GroundControlPoint] = ctx.repo_gcp.get_by_map(map_id)  # type: ignore[assignment]

    if not map_registry:
        typer.echo(
            f"Error: No GCPs found in repository for map '{map_id}'",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(map_registry)} GCPs from repository for map '{map_id}'")

    # Load existing annotations if any
    existing_annotations = ctx.repo_captured_frame.get_annotations(frame_id)
    if existing_annotations:
        typer.echo(f"Found {len(existing_annotations)} existing annotations")

    # Create calibration session
    session = CalibrationSession(
        camera_name=camera_name,
        frame=frame,
        registry=map_registry,
        height_m=Meters(height),
        pan_offset_deg=Degrees(pan_offset),
        pan_raw=pan_raw_value,
        tilt_deg=tilt_value,
        zoom=zoom_value,
        sensor_width_mm=Millimeters(7.18),
    )

    # Add existing annotations as reference points
    for ann in existing_annotations:
        session.add_reference_point(
            int(ann.pixel.x),
            int(ann.pixel.y),
            ann.gcp_id,
        )

    # Run interactive annotation session
    try:
        _run_annotation_session(session, frame_id, ctx)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _run_annotation_session(
    session: CalibrationSession,
    frame_id: str,
    ctx: ApplicationContext,
) -> None:
    """Run the interactive annotation session with OpenCV GUI.

    This is similar to run_interactive_session but adds the 'A' key
    to save annotations to the repository.
    """
    import select
    import sys

    import cv2 as cv2_lib

    def mouse_callback(
        event: int, x: int, y: int, _flags: int, user_session: CalibrationSession | None
    ) -> None:
        """Handle mouse events for the annotation window."""
        if user_session is None:
            return
        if event == cv2_lib.EVENT_LBUTTONDOWN:
            user_session.pending_click = (x, y)
            print(f"\nClicked at pixel ({x}, {y})")
            print("Enter GCP ID (e.g., Z1, P5): ", end="", flush=True)

    window_name = f"Annotate - {session.camera_name}"
    cv2_lib.namedWindow(window_name, cv2_lib.WINDOW_NORMAL)
    cv2_lib.resizeWindow(
        window_name, min(1920, session.image_width), min(1080, session.image_height)
    )
    cv2_lib.setMouseCallback(window_name, mouse_callback, session)

    session._update_display()

    print("\n" + "=" * 60)
    print("INTERACTIVE ANNOTATION MODE")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Click on a GCP in the image")
    print("2. Enter the GCP ID when prompted (e.g., Z1, P5)")
    print("3. Repeat for all visible GCPs")
    print("4. Press 'A' to save annotations to repository")
    print("5. Press 'C' to run calibration")
    print("6. Press 'S' to save calibration results to JSON")
    print("7. Press 'R' to reset all points")
    print("8. Press 'Q' or ESC to quit")
    print("=" * 60 + "\n")

    while True:
        cv2_lib.imshow(window_name, session.display_frame)
        key = cv2_lib.waitKey(100) & 0xFF

        # Check for pending GCP ID input
        if session.pending_click is not None:
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    line = sys.stdin.readline().strip()
                    if line:
                        gcp_id = line.upper()
                        if gcp_id in session.registry:
                            session.add_reference_point(
                                session.pending_click[0], session.pending_click[1], gcp_id
                            )
                            map_point = session.registry[gcp_id].map_point
                            print(
                                f"Added annotation at ({session.pending_click[0]}, "
                                f"{session.pending_click[1]}) -> {gcp_id} "
                                f"(map coords: {float(map_point.pixel_point.x):.2f}, "
                                f"{float(map_point.pixel_point.y):.2f})"
                            )
                        else:
                            print(f"GCP ID '{gcp_id}' not found in registry")
                except (ValueError, IndexError):
                    print("Invalid format. Enter a GCP ID (e.g., Z1, P5)")
                session.pending_click = None

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord("a"):  # Save annotations
            if not session.reference_points:
                print("No annotations to save")
            else:
                # Convert reference points to Annotation entities
                annotations = []
                for pt in session.reference_points:
                    ann = Annotation(
                        gcp_id=pt.map_point_id,
                        frame_id=frame_id,
                        camera_pose=PTZState(
                            pan_raw=Degrees(float(session.pan_raw)),
                            tilt_deg=Degrees(float(session.tilt_deg)),
                            zoom=Unitless(float(session.zoom)),
                        ),
                        pixel=PixelPoint(
                            _x=PixelsFloat(pt.pixel_u),
                            _y=PixelsFloat(pt.pixel_v),
                        ),
                    )
                    annotations.append(ann)

                ctx.repo_captured_frame.save_annotations(frame_id, annotations)
                print(f"\nSaved {len(annotations)} annotations to repository")
        elif key == ord("c"):  # Calibrate
            session.calibrate()
        elif key == ord("s"):  # Save calibration JSON
            try:
                session.save_results()
            except ValueError as e:
                print(f"Error: {e}")
        elif key == ord("r"):  # Reset
            session.reference_points = []
            session.best_pan_offset = None
            session.best_height = None
            session.best_error = None
            session._update_display()
            print("Reset all annotations")

    cv2_lib.destroyAllWindows()
