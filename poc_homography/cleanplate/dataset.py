"""Survey-dataset loader feeding the offline clean-plate pipeline.

This module bridges the persisted multi-visit *survey* layout
(``{frames_root}/{run_id}/{camera_id}/frames/*.yaml`` plus on-disk images and
optional floor masks) to the in-memory :class:`CleanPlateFrame` inputs consumed
by :func:`poc_homography.cleanplate.reconstruct_clean_plate`.

Pipeline:

    1. :meth:`CleanPlateDataset.from_survey_run` loads every
       :class:`~poc_homography.domain.entities.survey.frame_record.FrameRecord`
       for a run (optionally a single camera) via :class:`RepoYamlSurveyRun`.
    2. :meth:`CleanPlateDataset.groups` buckets frames by ``(camera_id,
       pose_id)`` (the natural clean-plate fusion unit: same camera, same
       physical pose, many visits).
    3. :meth:`CleanPlateDataset.frames_for` materialises one group: it reads
       each image from disk, loads or synthesizes a floor mask, and resolves the
       ground homography into a numpy 3x3.

Ground-homography orientation
------------------------------
:attr:`CleanPlateFrame.ground_homography` MUST map **world ground meters
``[x, y, 1]`` -> image pixels ``[u, v, 1]``** (``ortho.py`` inverts it). The
survey ``GroundHomography.h_matrix`` is stored in the SAME orientation: its
docstring describes it as "the ground homography H recomputable" from the
extrinsics, and the homography core that produces it
(:class:`poc_homography.homography.intrinsic_extrinsic.IntrinsicExtrinsicHomography`,
docstring: "The homography H maps world ground plane points to image pixels")
is world->image. Therefore the cached ``h_matrix`` is used DIRECTLY, without
inversion. The single conversion point is :func:`_survey_h_to_world_to_image`,
so the convention is easy to flip if survey ever changes direction.

On-disk path resolution
------------------------
``image_data.image_path`` and ``floor_mask_reference.mask_ref`` are stored
RELATIVE to the per-camera directory ``{frames_root}/{run_id}/{camera_id}/``
(e.g. ``frames/cap-0001.png`` resolves to
``{frames_root}/{run_id}/{camera_id}/frames/cap-0001.png``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from poc_homography.cleanplate.reconstruct import CleanPlateFrame
from poc_homography.infrastructure.repositories.repo_yaml_survey_run import (
    RepoYamlSurveyRun,
)

if TYPE_CHECKING:
    from pathlib import Path

    from poc_homography.domain.entities.survey.frame_record import FrameRecord
    from poc_homography.types import Unitless

logger = logging.getLogger(__name__)

GroupKey = tuple[str, str]
"""A ``(camera_id, pose_id)`` grouping key."""


def _survey_h_to_world_to_image(h_matrix: list[list[float]]) -> np.ndarray:
    """Convert a survey ``GroundHomography.h_matrix`` to a world->image 3x3.

    Survey stores ``h_matrix`` already in the world-meters -> image-pixels
    orientation (see the module docstring for the supporting evidence), so this
    is a direct ``np.asarray`` with NO inversion. Centralising the conversion
    here makes the convention trivial to flip should survey ever change.

    Args:
        h_matrix: Row-major nested list of floats (3x3).

    Returns:
        A ``(3, 3)`` float64 matrix mapping world meters to image pixels.
    """
    return np.asarray(h_matrix, dtype=float)


def _time_bucket_for(frame: FrameRecord) -> str | None:
    """Derive a coarse ``YYYY-MM-DD-HH`` time bucket from the capture timestamp.

    Returns ``None`` if no usable capture timestamp is present.
    """
    timestamp = frame.capture.timestamp_at_capture
    if timestamp is None:
        return None
    return timestamp.strftime("%Y-%m-%d-%H")


@dataclass(frozen=True)
class CleanPlateDataset:
    """A loaded survey run ready to be grouped into clean-plate inputs.

    Attributes:
        frames_root: The survey frames root (contains ``{run_id}/{camera_id}/``).
        run_id: The survey run identifier this dataset was loaded for.
        records: Every loaded :class:`FrameRecord` for the run/camera filter.
    """

    frames_root: Path
    run_id: str
    records: tuple[FrameRecord, ...]
    # Grouping computed ONCE at load time (single O(records) pass) and reused by
    # both ``groups`` and ``frames_for``; excluded from ``repr`` for brevity.
    _grouped: dict[GroupKey, list[FrameRecord]] = field(repr=False, compare=False)

    @staticmethod
    def _group_records(
        records: tuple[FrameRecord, ...],
    ) -> dict[GroupKey, list[FrameRecord]]:
        """Bucket records by ``(camera_id, pose_id)``, skipping pose-less frames.

        Frames whose ``survey_context.pose_id`` is ``None`` are SKIPPED (a
        clean-plate group is defined by a concrete physical pose) and logged at
        debug level.
        """
        grouped: dict[GroupKey, list[FrameRecord]] = {}
        for record in records:
            pose_id = record.survey_context.pose_id
            if pose_id is None:
                logger.debug("Skipping frame %s with no pose_id", record.id)
                continue
            key = (record.camera.camera_id, pose_id)
            grouped.setdefault(key, []).append(record)
        return grouped

    @classmethod
    def from_survey_run(
        cls,
        run_dir: Path,
        run_id: str,
        *,
        camera_id: str | None = None,
    ) -> CleanPlateDataset:
        """Load all frames for ``run_id`` from a survey frames root.

        ``run_dir`` is the survey *frames root* — the directory that directly
        contains ``{run_id}/{camera_id}/frames/*.yaml`` (i.e. the ``survey/``
        directory of the standard ``data/`` layout). This is passed straight to
        :class:`RepoYamlSurveyRun` as its ``frames_dir`` so frame discovery and
        on-disk path resolution share one base.

        Args:
            run_dir: Survey frames root containing ``{run_id}/{camera_id}/``.
            run_id: The run to load frames for.
            camera_id: If given, keep only frames from this camera.

        Returns:
            A :class:`CleanPlateDataset` holding the matching frame records.
        """
        # ``data_dir`` (run manifests) is unused for frame loading; point it at
        # ``run_dir`` and disable directory creation side effects.
        repo = RepoYamlSurveyRun(run_dir, frames_dir=run_dir, create_dir=False)
        records = repo.get_frames_by_run(run_id)
        if camera_id is not None:
            records = [r for r in records if r.camera.camera_id == camera_id]
        records_tuple = tuple(records)
        return cls(
            frames_root=run_dir,
            run_id=run_id,
            records=records_tuple,
            _grouped=cls._group_records(records_tuple),
        )

    def groups(self) -> dict[GroupKey, list[FrameRecord]]:
        """Group frames by ``(camera_id, pose_id)``.

        Returns a shallow copy of the grouping computed once at load time.
        Frames whose ``survey_context.pose_id`` is ``None`` were SKIPPED (a
        clean-plate group is defined by a concrete physical pose); they are
        logged at debug level and never appear in the returned mapping.

        Returns:
            Mapping from ``(camera_id, pose_id)`` to its list of frame records.
        """
        return {key: list(records) for key, records in self._grouped.items()}

    def frames_for(
        self,
        camera_id: str,
        pose_id: str,
        *,
        synthesize_missing_masks: bool = True,
    ) -> list[CleanPlateFrame]:
        """Materialise the ``(camera_id, pose_id)`` group as CleanPlateFrames.

        For each record in the group this:

            * reads the image from disk (``cv2.imread`` -> RGB ``(H, W, 3)``);
              records whose image file is missing are SKIPPED with a warning;
            * loads the floor mask from ``floor_mask_reference.mask_ref`` if it
              points to a readable file; otherwise the mask is left as ``None``,
              which ortho-rectification treats as all-floor (every in-image
              pixel is floor) via its cheaper no-mask footprint path. ``None``
              is the canonical "all-floor" representation here — equivalent to an
              all-True mask but cheaper — so ``synthesize_missing_masks`` no
              longer materialises an array; both flag values yield ``None`` when
              no readable mask exists (the flag is retained for caller intent);
            * resolves the ground homography from the cached
              ``ground_homography.h_matrix`` (world->image, used directly).
              Records with no cached ``h_matrix`` are SKIPPED with a warning,
              since recomputation from raw extrinsics is intentionally out of
              scope for this loader.

        Args:
            camera_id: Camera identifier of the group.
            pose_id: Pose identifier of the group.
            synthesize_missing_masks: Treat a missing mask reference as
                all-floor (default True). All-floor is represented by ``None``
                (handled by ortho's no-mask footprint path), so this no longer
                materialises an array; retained to signal caller intent.

        Returns:
            The group's loaded :class:`CleanPlateFrame` objects (possibly fewer
            than the records when images or homographies are missing).
        """
        records = self._grouped.get((camera_id, pose_id), [])
        camera_dir = self.frames_root / self.run_id / camera_id
        frames: list[CleanPlateFrame] = []
        for record in records:
            clean_frame = self._build_frame(
                record, camera_dir, synthesize_missing_masks=synthesize_missing_masks
            )
            if clean_frame is not None:
                frames.append(clean_frame)
        return frames

    def _build_frame(
        self,
        record: FrameRecord,
        camera_dir: Path,
        *,
        synthesize_missing_masks: bool,
    ) -> CleanPlateFrame | None:
        """Build one :class:`CleanPlateFrame` or return ``None`` to skip it."""
        homography = self._resolve_homography(record)
        if homography is None:
            return None

        image = self._load_image(camera_dir / record.image_data.image_path)
        if image is None:
            return None

        floor_mask = self._load_mask(record, camera_dir, synthesize=synthesize_missing_masks)
        gain = self._resolve_gain(record)

        return CleanPlateFrame(
            image=image,
            floor_mask=floor_mask,
            ground_homography=homography,
            gain=gain,
            time_bucket=_time_bucket_for(record),
        )

    @staticmethod
    def _resolve_homography(record: FrameRecord) -> np.ndarray | None:
        """Resolve the world->image 3x3, or ``None`` if not available.

        Only the cached ``h_matrix`` is supported. Recomputing H from raw
        extrinsics (camera height + pan/tilt/roll) is non-trivial — it requires
        the intrinsic matrix and the world/map-origin convention — so when no
        cached matrix is present the frame is skipped and the caller is told the
        cached ``h_matrix`` is required.
        """
        ground = record.ground_homography
        if ground is None or ground.h_matrix is None:
            logger.warning(
                "Frame %s has no cached ground_homography.h_matrix; skipping "
                "(recompute-from-extrinsics is not supported by this loader)",
                record.id,
            )
            return None
        return _survey_h_to_world_to_image(ground.h_matrix)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray | None:
        """Read an image from disk as RGB ``(H, W, 3)`` uint8, or ``None``."""
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("Image file missing or unreadable: %s; skipping frame", path)
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_mask(
        record: FrameRecord,
        camera_dir: Path,
        *,
        synthesize: bool,
    ) -> np.ndarray | None:
        """Load the floor mask, or return ``None`` to mean all-floor.

        When no readable mask reference is available, the mask is returned as
        ``None``. With ``synthesize`` True this ``None`` is the documented
        "all-floor" sentinel (every in-image pixel is treated as floor, which
        ortho-rectification handles via its cheaper no-mask footprint path —
        identical to an all-True mask). With ``synthesize`` False it likewise
        means "no mask supplied"; the two are behaviourally equivalent here, but
        the flag is retained so callers can distinguish the intent.
        """
        reference = record.floor_mask_reference
        if reference is not None and reference.mask_ref is not None:
            mask_path = camera_dir / reference.mask_ref
            raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                return raw > 0
            logger.warning(
                "Mask file missing or unreadable: %s; falling back to all-floor", mask_path
            )
        if synthesize:
            logger.debug("No floor mask for frame %s; treating as all-floor (None)", record.id)
        return None

    @staticmethod
    def _resolve_gain(record: FrameRecord) -> Unitless | None:
        """Pull the capture gain from ``full_optics``, if present."""
        optics = record.full_optics
        if optics is None or optics.gain is None:
            return None
        return optics.gain  # type: ignore[return-value]
