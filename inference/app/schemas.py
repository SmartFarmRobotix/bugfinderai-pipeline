# Copyright 2026 Smart Farm Robotix VCC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
the inference service's own upload/inspection API types. These are NOT the versioned wire
contract (see app/contract.py) — they are this service's local HTTP
surface for the caller that creates a session (robot / session-feeder) and
for debugging. Session-submission payloads pushed onward to the intake
service are built from app.contract.SessionSubmission instead.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    session_id: str
    best_image_id: str
    infestation_score: float = Field(
        ..., ge=0.0, le=1.0, description="Session-level aggregated score"
    )


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Optional client-provided session identifier"
    )
    parcel_id: Optional[str] = Field(default=None, description="Parcel identifier")
    time_start: Optional[str] = Field(default=None, description="ISO-8601 start time")
    time_end: Optional[str] = Field(default=None, description="ISO-8601 end time")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Arbitrary session metadata"
    )
