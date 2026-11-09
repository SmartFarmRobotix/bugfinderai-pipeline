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
Minimal structured logging with request/correlation-id propagation. Mirrors the PDM service's logging convention (
context-local `request_id_var`, populated from the inbound `X-Request-ID`
header when present, never minted here) and adds a second context-local
var, `session_id_var`, because every log line must also carry `session_id`, which PDM's
per-request-only convention has no equivalent for (PDM's own inbound
requests are already session-scoped 1:1; the orchestrator's are not: one
inbound HTTP request drives potentially many outbound calls and state
transitions for the SAME session, across which `session_id_var` stays set
while `request_id_var` (GateKeeper-propagated) stays constant for the
request that carried it in).

A full observability stack is explicitly out of scope — this is
exactly PDM's pattern plus one more contextvar, not a new logging
subsystem.
"""

from __future__ import annotations

import contextvars
import logging

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")


class CorrelationLogFilter(logging.Filter):
    """Injects the current context-local request id and session id into
    each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        return True


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s [request_id=%(request_id)s session_id=%(session_id)s] %(name)s: %(message)s",
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(CorrelationLogFilter())


def current_request_id() -> "str | None":
    """The request id to propagate on an outbound call's `X-Request-ID`
    header, or `None` when nothing is set for the current context. Callers
    must omit the header entirely on `None` (never send the literal "-") —
    GateKeeper mints its own id for a request that arrives without one, so
    omitting it here means the SAME honest behavior downstream, not a
    fabricated correlation id."""
    value = request_id_var.get()
    return value if value != "-" else None
