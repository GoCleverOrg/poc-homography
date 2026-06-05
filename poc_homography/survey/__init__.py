"""Survey orchestration package (planning, execution).

This package hosts the multi-phase survey planner (#260) and related
orchestration concerns. The domain entities/enums it builds on live under
``poc_homography.domain``; the planner consumes them and produces a C1
``SurveyRun`` header for persistence compatibility.
"""

from __future__ import annotations
