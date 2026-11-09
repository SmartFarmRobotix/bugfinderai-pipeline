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

"""Add session_record.origin and session_record.edge_report_id — set by the
cloud intake when a synced edge session is submitted for re-assessment
(origin "edge-sync", edge_report_id the synced on-device report's id); null
for every other session

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-09-25 00:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("session_record", sa.Column("origin", sa.String(), nullable=True))
    op.add_column("session_record", sa.Column("edge_report_id", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("session_record", "edge_report_id")
    op.drop_column("session_record", "origin")
