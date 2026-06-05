"""Device health and lens odometry value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poc_homography.domain.vo._xml import find_child, find_text

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


def _int_at(elem: Element, *path: str) -> int:
    """Return the integer text at ``path`` under ``elem``, defaulting to 0."""
    return int(find_text(elem, *path) or 0)


@dataclass(frozen=True)
class LensOdometry:
    """Cumulative lens and dome wear counters.

    Attributes:
        zoom_reverse_times: Zoom motor reversal count.
        zoom_total_steps: Total zoom steps travelled.
        focus_reverse_times: Focus motor reversal count.
        focus_total_steps: Total focus steps travelled.
        iris_shift_times: Iris shift count.
        iris_total_steps: Total iris steps travelled.
        icr_shift_times: IR-cut filter shift count.
        lens_intir_times: Lens internal IR count.
        camera_run_total_time: Total camera run time.
        pan_total_rounds: Total pan rotations of the dome.
        tilt_total_rounds: Total tilt rotations of the dome.
    """

    zoom_reverse_times: int
    zoom_total_steps: int
    focus_reverse_times: int
    focus_total_steps: int
    iris_shift_times: int
    iris_total_steps: int
    icr_shift_times: int
    lens_intir_times: int
    camera_run_total_time: int
    pan_total_rounds: int
    tilt_total_rounds: int

    @classmethod
    def from_element(cls, elem: Element) -> LensOdometry:
        """Build :class:`LensOdometry` from a ``DeviceStatus`` root element."""
        camera = find_child(elem, "CameraList", "Camera")
        dome = find_child(elem, "DomeInfoList", "DomeInfo")
        camera = camera if camera is not None else elem
        dome = dome if dome is not None else elem
        return cls(
            zoom_reverse_times=_int_at(camera, "zoomReverseTimes"),
            zoom_total_steps=_int_at(camera, "zoomTotalSteps"),
            focus_reverse_times=_int_at(camera, "focusReverseTimes"),
            focus_total_steps=_int_at(camera, "focusTotalSteps"),
            iris_shift_times=_int_at(camera, "irisShiftTimes"),
            iris_total_steps=_int_at(camera, "irisTotalSteps"),
            icr_shift_times=_int_at(camera, "icrShiftTimes"),
            lens_intir_times=_int_at(camera, "lensIntirTimes"),
            camera_run_total_time=_int_at(camera, "cameraRunTotalTime"),
            pan_total_rounds=_int_at(dome, "panTotalRounds"),
            tilt_total_rounds=_int_at(dome, "tiltTotalRounds"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "zoom_reverse_times": self.zoom_reverse_times,
            "zoom_total_steps": self.zoom_total_steps,
            "focus_reverse_times": self.focus_reverse_times,
            "focus_total_steps": self.focus_total_steps,
            "iris_shift_times": self.iris_shift_times,
            "iris_total_steps": self.iris_total_steps,
            "icr_shift_times": self.icr_shift_times,
            "lens_intir_times": self.lens_intir_times,
            "camera_run_total_time": self.camera_run_total_time,
            "pan_total_rounds": self.pan_total_rounds,
            "tilt_total_rounds": self.tilt_total_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LensOdometry:
        """Create :class:`LensOdometry` from a dictionary."""
        return cls(
            zoom_reverse_times=int(data["zoom_reverse_times"]),
            zoom_total_steps=int(data["zoom_total_steps"]),
            focus_reverse_times=int(data["focus_reverse_times"]),
            focus_total_steps=int(data["focus_total_steps"]),
            iris_shift_times=int(data["iris_shift_times"]),
            iris_total_steps=int(data["iris_total_steps"]),
            icr_shift_times=int(data["icr_shift_times"]),
            lens_intir_times=int(data["lens_intir_times"]),
            camera_run_total_time=int(data["camera_run_total_time"]),
            pan_total_rounds=int(data["pan_total_rounds"]),
            tilt_total_rounds=int(data["tilt_total_rounds"]),
        )


@dataclass(frozen=True)
class DeviceHealth:
    """Runtime health metrics for a camera device.

    Attributes:
        uptime_seconds: Device uptime in seconds.
        cpu_utilization: CPU utilization percentage.
        memory_usage: Memory usage percentage.
        fan_state: Dome fan state.
        heat_state: Dome heater state.
        total_reboot_count: Cumulative reboot count.
        odometry: Lens and dome wear counters.
    """

    uptime_seconds: int
    cpu_utilization: int
    memory_usage: int
    fan_state: int
    heat_state: int
    total_reboot_count: int
    odometry: LensOdometry

    @classmethod
    def from_element(cls, elem: Element) -> DeviceHealth:
        """Build :class:`DeviceHealth` from a ``DeviceStatus`` root element."""
        dome = find_child(elem, "DomeInfoList", "DomeInfo")
        dome = dome if dome is not None else elem
        return cls(
            uptime_seconds=_int_at(elem, "deviceUpTime"),
            cpu_utilization=_int_at(elem, "CPUList", "CPU", "cpuUtilization"),
            memory_usage=_int_at(elem, "MemoryList", "Memory", "memoryUsage"),
            fan_state=_int_at(dome, "fanState"),
            heat_state=_int_at(dome, "heatState"),
            total_reboot_count=_int_at(elem, "totalRebootCount"),
            odometry=LensOdometry.from_element(elem),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "cpu_utilization": self.cpu_utilization,
            "memory_usage": self.memory_usage,
            "fan_state": self.fan_state,
            "heat_state": self.heat_state,
            "total_reboot_count": self.total_reboot_count,
            "odometry": self.odometry.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceHealth:
        """Create :class:`DeviceHealth` from a dictionary."""
        return cls(
            uptime_seconds=int(data["uptime_seconds"]),
            cpu_utilization=int(data["cpu_utilization"]),
            memory_usage=int(data["memory_usage"]),
            fan_state=int(data["fan_state"]),
            heat_state=int(data["heat_state"]),
            total_reboot_count=int(data["total_reboot_count"]),
            odometry=LensOdometry.from_dict(data["odometry"]),
        )
