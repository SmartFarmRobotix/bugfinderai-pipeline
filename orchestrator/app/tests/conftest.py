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

from __future__ import annotations

import os
import sys

os.environ.setdefault("ORCH_GATEKEEPER_USERNAME", "test-orchestrator")
os.environ.setdefault("ORCH_GATEKEEPER_PASSWORD", "test-orchestrator-password")
os.environ.setdefault("ORCH_DATABASE_URL", "sqlite:///:memory:")

_APP_DIR = os.path.join(os.path.dirname(__file__), "..")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config as AlembicConfig  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

_SERVICE_ROOT = os.path.join(_APP_DIR, "..")
_ALEMBIC_INI = os.path.join(_SERVICE_ROOT, "alembic.ini")


@pytest.fixture()
def db(tmp_path) -> Session:
    """A fresh, fully-migrated SQLite database per test — `alembic upgrade
    head` run against a throwaway temp file, proving "migration
    applies from empty" for real on every test run, not just once."""
    db_path = tmp_path / "orchestrator-test.db"
    url = f"sqlite:///{db_path}"

    cfg = AlembicConfig(_ALEMBIC_INI)
    cfg.set_main_option("script_location", os.path.join(_SERVICE_ROOT, "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    engine = create_engine(url, connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def new_db_connection(db):
    """Returns a factory that opens a brand-new Session against the SAME
    on-disk database `db` is bound to — the persistence-layer equivalent of
    "the process restarted": no in-memory state survives, only what was
    committed to disk."""
    engine = db.get_bind()
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _open() -> Session:
        return session_factory()

    return _open
