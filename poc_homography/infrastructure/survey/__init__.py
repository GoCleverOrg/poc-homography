"""Survey capture orchestration (the per-pose capture step, C2).

Programs against the domain :class:`~poc_homography.domain.protocols.camera_device.CameraDevice`
protocol and emits versioned C1 records
(:class:`~poc_homography.domain.entities.survey.frame_record.FrameRecord`,
:class:`~poc_homography.domain.entities.survey.video_burst_record.VideoBurstRecord`).
"""

from __future__ import annotations
