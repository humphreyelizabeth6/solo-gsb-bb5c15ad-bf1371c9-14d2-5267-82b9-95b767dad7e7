"""Read-only threshold-exposure accounting over one source's published readings.

``GET /sessions/{session_id}/exposure`` answers, for one declared source, how
much event time inside a half-open window ``[from, to)`` its **published**
readings held a value **strictly greater** than a numeric threshold.

The source's value is treated as a step function over event time:

* Every published reading's value holds from its own ``observed_at`` until
  that source's next published reading, or until the query endpoint ``to``.
* The window start ``from`` inherits the most recent published reading at or
  before it (``observed_at <= from``); when no such reading exists the time
  before the first in-window reading is **unknown**.
* Several readings sharing one ``observed_at`` take effect in ``seq`` order;
  the reading with the largest ``seq`` at that timestamp is the one that
  holds.
* A missing value, a non-finite number (NaN/Infinity), a boolean or a
  non-numeric payload (also a missing ``value_key``) starts an **unknown**
  interval: the previous *valid* value is not carried across it.

The response reports the above-threshold seconds, the seconds covered by a
valid sampled value (above or at/below the threshold) and the unknown
seconds; ``covered + unknown`` always equals the window length.  When there
is no valid coverage the above-threshold ratio is ``null``; otherwise it is
the above-threshold time divided by the covered time.

Only ``status = 'published'`` rows participate: buffered readings have not
been released yet and quarantined readings never belong to the published
timeline.  The whole computation -- session/source validation and the single
timeline read -- runs inside one deferred SQLite read transaction, i.e. one
consistent snapshot.  Nothing is ever written: watermarks, alerts,
checkpoints and seals are untouched.
"""
from __future__ import annotations

import calendar
import json
import math
import sqlite3
from typing import Any, Optional

from .alerts import extract_numeric
from .database import read_transaction
from .errors import BadRequestError
from .service import _load_session
from .time_utils import format_event_time, parse_event_time, store_event_time

__all__ = ["compute_exposure"]

# Step-function states swept over the window.
_STATE_UNKNOWN = "unknown"
_STATE_ABOVE = "above"
_STATE_BELOW = "below"


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------

def _required_time(value: Any, field: str) -> str:
    """Validate/normalise a required event-time query parameter."""
    if value is None:
        raise BadRequestError(f"{field} is required")
    if not isinstance(value, str) or not value.strip():
        raise BadRequestError(f"{field} must be a non-empty ISO 8601 string")
    # TimestampError is mapped to HTTP 400 by the application handler.
    return store_event_time(parse_event_time(value))


def _validate_threshold(value: Any) -> float:
    """Parse the required finite ``threshold`` query parameter (400 on error).

    Query parameters arrive as strings; anything missing, non-numeric or
    non-finite (NaN / Infinity) is a bad request rather than a Pydantic 422.
    """
    if value is None:
        raise BadRequestError("threshold is required")
    if not isinstance(value, str) or not value.strip():
        raise BadRequestError("threshold must be a finite number")
    try:
        threshold = float(value.strip())
    except ValueError as exc:
        raise BadRequestError(
            f"threshold must be a finite number, got {value!r}"
        ) from exc
    if not math.isfinite(threshold):
        raise BadRequestError("threshold must be a finite number")
    return threshold


def _stored_micros(store_ts: str) -> int:
    """Exact integer microseconds since the epoch for a stored UTC timestamp.

    Stored timestamps have microsecond resolution, so all interval lengths
    are accumulated as integer microseconds and only divided by 1e6 when the
    seconds values are returned.
    """
    dt = parse_event_time(store_ts)
    return calendar.timegm(dt.timetuple()) * 1_000_000 + dt.microsecond


# ---------------------------------------------------------------------------
# computation
# ---------------------------------------------------------------------------

def compute_exposure(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    source_id: Any,
    from_: Any,
    to: Any,
    threshold: Any,
    value_key: Any = None,
) -> dict[str, Any]:
    """Return above-threshold/covered/unknown seconds for one source window.

    Read-only: every value comes from one deferred read snapshot and no table
    is ever written.  Malformed parameters raise
    :class:`~app.errors.BadRequestError` / :class:`TimestampError` (400);
    an unknown session raises 404; an undeclared ``source_id`` raises 400.
    """
    if source_id is None or not isinstance(source_id, str) or not source_id.strip():
        raise BadRequestError("source_id is required and must be a non-empty string")
    source_id = source_id.strip()

    if value_key is not None:
        if not isinstance(value_key, str) or not value_key.strip():
            raise BadRequestError("value_key must be a non-empty string")
        value_key = value_key.strip()

    from_store = _required_time(from_, "from")
    to_store = _required_time(to, "to")
    # The window is half-open [from, to); an empty or backwards window is not
    # a legal interval.
    if from_store >= to_store:
        raise BadRequestError(
            f"'from' ({format_event_time(parse_event_time(from_store))}) must be"
            f" strictly earlier than 'to' ({format_event_time(parse_event_time(to_store))})"
        )
    threshold_value = _validate_threshold(threshold)

    from_micros = _stored_micros(from_store)
    to_micros = _stored_micros(to_store)

    # Single snapshot, single timeline scan: every published reading of the
    # source that could possibly matter (strictly before ``to``) is read in
    # deterministic (observed_at, seq) order.  Rows at or before ``from``
    # supply the inherited starting state; rows strictly inside the window
    # supply the in-window transitions.  A reading exactly at ``to`` starts at
    # the excluded endpoint and therefore cannot matter.
    sql = (
        "SELECT observed_at, seq, payload FROM readings"
        " WHERE session_id = ? AND source_id = ? AND status = 'published'"
        " AND observed_at < ?"
        " ORDER BY observed_at, seq"
    )

    with read_transaction(conn):
        session = _load_session(conn, session_id)
        allowed_sources = set(json.loads(session["source_ids"]))
        if source_id not in allowed_sources:
            raise BadRequestError(
                f"unknown source_id {source_id!r} for this session;"
                f" declared sources: {sorted(allowed_sources)}"
            )
        rows = conn.execute(sql, (session_id, source_id, to_store)).fetchall()

    # ``initial_row`` keeps the last reading (observed_at, seq) at or before
    # ``from`` -- the nearest prior reading the window start inherits.
    initial_row: Optional[sqlite3.Row] = None
    # In-window transitions collapsed to one effective reading per distinct
    # event time: (micros, extracted finite number or None).
    transitions: list[tuple[int, Optional[float]]] = []
    for row in rows:
        observed = row["observed_at"]
        if observed <= from_store:
            initial_row = row
            continue
        value = extract_numeric(row["payload"], value_key)
        ts_micros = _stored_micros(observed)
        # Rows arrive in (observed_at, seq) order, so overwriting the current
        # group leaves the largest-seq reading at each timestamp in effect.
        if transitions and transitions[-1][0] == ts_micros:
            transitions[-1] = (ts_micros, value)
        else:
            transitions.append((ts_micros, value))

    def classify(value: Optional[float]) -> str:
        if value is None:
            return _STATE_UNKNOWN
        # Strictly greater: equality stays covered but never above.
        return _STATE_ABOVE if value > threshold_value else _STATE_BELOW

    state = (
        classify(extract_numeric(initial_row["payload"], value_key))
        if initial_row is not None
        else _STATE_UNKNOWN
    )

    above_micros = 0
    covered_micros = 0
    unknown_micros = 0

    def account(state_value: str, duration_micros: int) -> None:
        nonlocal above_micros, covered_micros, unknown_micros
        if state_value == _STATE_UNKNOWN:
            unknown_micros += duration_micros
        else:
            covered_micros += duration_micros
            if state_value == _STATE_ABOVE:
                above_micros += duration_micros

    cursor = from_micros
    for ts_micros, value in transitions:
        account(state, ts_micros - cursor)
        state = classify(value)
        cursor = ts_micros
    # The last effective value (or unknown) holds until the query endpoint.
    account(state, to_micros - cursor)

    return {
        "session_id": session_id,
        "source_id": source_id,
        "from": format_event_time(parse_event_time(from_store)),
        "to": format_event_time(parse_event_time(to_store)),
        "threshold": threshold_value,
        "value_key": value_key,
        "above_seconds": above_micros / 1_000_000,
        "covered_seconds": covered_micros / 1_000_000,
        "unknown_seconds": unknown_micros / 1_000_000,
        # No valid sampled coverage -> the proportion is undefined (null);
        # otherwise above-threshold time as a fraction of covered time.
        "above_ratio": (
            above_micros / covered_micros if covered_micros > 0 else None
        ),
    }
