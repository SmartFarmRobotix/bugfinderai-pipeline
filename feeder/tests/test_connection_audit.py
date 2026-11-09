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

"""An audit artifact is
produced per run, and it actually records socket/resolver attempts."""

import socket

from app import connection_audit


def test_install_is_noop_without_env_var(monkeypatch):
    monkeypatch.delenv("CONN_AUDIT_LOG_DIR", raising=False)
    connection_audit._reset_for_tests()
    try:
        assert connection_audit.install("feeder") is None
        assert connection_audit.current_log_path() is None
    finally:
        connection_audit._reset_for_tests()


def test_install_produces_an_empty_artifact_even_with_zero_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("CONN_AUDIT_LOG_DIR", str(tmp_path))
    connection_audit._reset_for_tests()
    try:
        path = connection_audit.install("feeder")
        assert path is not None
        assert path.exists()
        assert connection_audit.attempt_count() == 0
    finally:
        connection_audit._reset_for_tests()


def test_connect_attempt_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("CONN_AUDIT_LOG_DIR", str(tmp_path))
    connection_audit._reset_for_tests()
    try:
        connection_audit.install("feeder")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                sock.connect(("127.0.0.1", 1))
            except OSError:
                pass
        finally:
            sock.close()
        assert connection_audit.attempt_count() == 1
    finally:
        connection_audit._reset_for_tests()
