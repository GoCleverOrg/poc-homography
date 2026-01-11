"""Main Typer CLI application for homography tools."""

import typer

app = typer.Typer(
    help="Homography tools for camera calibration and coordinate transformations",
    no_args_is_help=True,
)

# Subcommand groups - will be populated as tools are migrated
calibrate_app = typer.Typer(help="Calibration commands")
camera_app = typer.Typer(help="Camera intrinsics and validation commands")
gcp_app = typer.Typer(help="Ground Control Point commands")
test_app = typer.Typer(help="Testing and data generation commands")

app.add_typer(calibrate_app, name="calibrate")
app.add_typer(camera_app, name="camera")
app.add_typer(gcp_app, name="gcp")
app.add_typer(test_app, name="test")


def _register_commands() -> None:
    """
    Import command modules to register commands with their respective apps.

    This function is called at module load time to register all commands.
    Commands use decorators like @calibrate_app.command() which register
    themselves when the module is imported.
    """
    from poc_homography.cli import calibrate, camera, gcp, interactive, test_cmds

    # Avoid "imported but unused" warnings by explicitly using the module
    _ = calibrate
    _ = camera
    _ = gcp
    _ = interactive
    _ = test_cmds


_register_commands()


if __name__ == "__main__":
    app()
