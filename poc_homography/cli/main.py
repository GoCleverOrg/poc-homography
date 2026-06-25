"""Main Typer CLI application for homography tools."""

import typer

app = typer.Typer(
    help="Homography tools for camera calibration and coordinate transformations",
    no_args_is_help=True,
)

# Subcommand groups - will be populated as tools are migrated
calibrate_app = typer.Typer(help="Calibration commands")
camera_app = typer.Typer(help="Camera intrinsics and validation commands")
gcp_app = typer.Typer(help="GCP analysis commands")
survey_app = typer.Typer(help="Survey commands")
test_app = typer.Typer(help="Testing and data generation commands")

app.add_typer(calibrate_app, name="calibrate")
app.add_typer(camera_app, name="camera")
app.add_typer(gcp_app, name="gcp")
app.add_typer(survey_app, name="survey")
app.add_typer(test_app, name="test")


def _register_commands() -> None:
    """
    Import command modules to register commands with their respective apps.

    This function is called at module load time to register all commands.
    Commands use decorators like @calibrate_app.command() which register
    themselves when the module is imported.
    """
    from poc_homography.cli import (
        bootstrap,
        calibrate,
        calibrate_capture,
        camera,
        cleanplate,
        confidence_report,
        gcp,
        interactive,
        lens_auto_calibrate,
        line_picker,
        maps,
        point_picker,
        rollout,
        run_camera,
        self_calibrate,
        survey,
        test_cmds,
    )

    app.add_typer(cleanplate.cleanplate_app, name="cleanplate")
    app.add_typer(maps.maps_app, name="maps")

    # Avoid "imported but unused" warnings by explicitly using the module
    _ = bootstrap
    _ = calibrate
    _ = calibrate_capture
    _ = camera
    _ = cleanplate
    _ = confidence_report
    _ = gcp
    _ = interactive
    _ = lens_auto_calibrate
    _ = line_picker
    _ = maps
    _ = point_picker
    _ = rollout
    _ = run_camera
    _ = self_calibrate
    _ = survey
    _ = test_cmds


_register_commands()


if __name__ == "__main__":
    app()
