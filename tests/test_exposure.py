"""End-to-end tests for the read-only threshold-exposure endpoint."""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app.time_utils import parse_event_time, store_event_time


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        yield c, db_path


def create_session(client, sources=("a", "b"), lateness=10.0, session_id="s"):
    r = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "source_ids": list(sources),
            "allowed_lateness_seconds": lateness,
        },
    )
    assert r.status_code == 201, r.text
    return session_id


def exposure(client, sid, **params):
    return client.get(f"/sessions/{sid}/exposure", params=params)


def window(threshold=10, **over):
    params = {
        "from": "2026-01-01T10:00:00Z",
        "to": "2026-01-01T10:01:00Z",
        "threshold": str(threshold),
    }
    params.update(over)
    return params


def t(value: str) -> str:
    return store_event_time(parse_event_time(value))


def payload(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def insert_reading(
    db_path,
    sid,
    source,
    seq,
    observed_at,
    value,
    *,
    status="published",
    published_watermark="2026-01-01T11:00:00Z",
    quarantined_watermark=None,
):
    """Insert a reading row with an explicit status and canonical payload."""
    with sqlite3.connect(db_path) as raw:
        raw.execute(
            "INSERT INTO readings (session_id, source_id, seq, observed_at,"
            " payload, status, first_received_at, published_watermark,"
            " quarantined_watermark)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sid,
                source,
                seq,
                t(observed_at),
                None if value is None else payload(value),
                status,
                t("2026-01-01T09:00:00Z"),
                t(published_watermark) if published_watermark and status == "published" else None,
                t(quarantined_watermark) if quarantined_watermark else None,
            ),
        )
        raw.commit()


def publish_via_seal(client, sid, source, readings, final_at=None):
    """Report readings through the real API and tail-publish by sealing."""
    r = client.post(
        f"/sessions/{sid}/readings",
        json={"readings": [
            {"source_id": source, "seq": seq, "observed_at": ts, "value": val}
            for seq, ts, val in readings
        ]},
    )
    assert r.status_code == 200, r.text
    if final_at is None:
        final_at = max(ts for _, ts, _ in readings)
    r = client.post(
        f"/sessions/{sid}/sources/{source}/seal",
        json={"final_observed_at": final_at},
    )
    assert r.status_code == 200, r.text


class TestExposureComputation:
    def test_basic_step_function_above_and_below(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        # 5 (below) for 0-20s; 30 (above) for 20-60s.
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 5)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:20Z", 30)
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 40.0
        assert body["covered_seconds"] == 60.0
        assert body["unknown_seconds"] == 0.0
        assert body["above_ratio"] == pytest.approx(40 / 60)

    def test_strict_threshold_equality_not_above(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 10)
        body = exposure(c, sid, source_id="a", **window(threshold=10)).json()
        assert body["above_seconds"] == 0.0
        assert body["covered_seconds"] == 60.0
        assert body["above_ratio"] == 0.0

    def test_window_start_inherits_previous_reading(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        # prior reading at :50 of the previous minute (value 30 above)
        insert_reading(db, sid, "a", 1, "2026-01-01T09:59:50Z", 30)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:10Z", 5)
        # window [00:00, 00:20): inherits 30 for the first 10s, then 5.
        body = exposure(
            c, sid, source_id="a",
            **window(**{"from": "2026-01-01T10:00:00Z", "to": "2026-01-01T10:00:20Z"}),
        ).json()
        assert body["above_seconds"] == 10.0
        assert body["covered_seconds"] == 20.0
        assert body["unknown_seconds"] == 0.0
        assert body["above_ratio"] == 0.5

    def test_no_reading_before_start_is_unknown(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:10Z", 30)
        body = exposure(
            c, sid, source_id="a",
            **window(**{"from": "2026-01-01T10:00:00Z", "to": "2026-01-01T10:00:40Z"}),
        ).json()
        assert body["unknown_seconds"] == 10.0
        assert body["above_seconds"] == 30.0
        assert body["covered_seconds"] == 30.0
        assert body["above_ratio"] == 1.0

    def test_reading_exactly_at_from_is_inherited(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 30)
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["unknown_seconds"] == 0.0
        assert body["above_seconds"] == 60.0

    def test_half_open_to_reading_at_to_does_not_apply(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 30)
        # value 5 exactly at the exclusive endpoint must not change anything.
        insert_reading(db, sid, "a", 2, "2026-01-01T10:01:00Z", 5)
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 60.0
        assert body["covered_seconds"] == 60.0

    def test_value_holds_until_query_endpoint_after_last_reading(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 5)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", 30)
        body = exposure(
            c, sid, source_id="a",
            **window(**{"to": "2026-01-01T10:01:00Z"}),
        ).json()
        # 5 for 30s, 30 for 30s.
        assert body["covered_seconds"] == 60.0
        assert body["above_seconds"] == 30.0

    def test_none_value_makes_unknown_and_does_not_carry_previous(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 30)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:20Z", None)
        insert_reading(db, sid, "a", 3, "2026-01-01T10:00:40Z", 30)
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 40.0
        assert body["covered_seconds"] == 40.0
        assert body["unknown_seconds"] == 20.0
        assert body["above_ratio"] == 1.0

    def test_nan_and_infinity_values_are_unknown(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", float("nan"))
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:20Z", float("inf"))
        insert_reading(db, sid, "a", 3, "2026-01-01T10:00:40Z", float("-inf"))
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["covered_seconds"] == 0.0
        assert body["unknown_seconds"] == 60.0
        assert body["above_seconds"] == 0.0
        assert body["above_ratio"] is None

    def test_boolean_value_is_unknown(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        # true > 10 would hold in a naive numeric comparison; it must not.
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", True)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", False)
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["unknown_seconds"] == 60.0
        assert body["covered_seconds"] == 0.0
        assert body["above_ratio"] is None

    def test_non_numeric_string_value_is_unknown(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", "hot")
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", 30)
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["unknown_seconds"] == 30.0
        assert body["above_seconds"] == 30.0
        assert body["covered_seconds"] == 30.0

    def test_coverage_resumes_after_unknown_interval(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 30)  # above 10s
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:10Z", None)  # unknown 10s
        insert_reading(db, sid, "a", 3, "2026-01-01T10:00:20Z", 5)  # below 40s
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 10.0
        assert body["covered_seconds"] == 50.0
        assert body["unknown_seconds"] == 10.0
        assert body["above_ratio"] == pytest.approx(0.2)

    def test_same_timestamp_multiple_seqs_last_wins(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 5)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:00Z", 99)
        insert_reading(db, sid, "a", 3, "2026-01-01T10:00:00Z", 30)
        body = exposure(c, sid, source_id="a", **window()).json()
        # seq 3 (30) is the value in effect; seq 1/2 never hold any duration.
        assert body["above_seconds"] == 60.0
        assert body["unknown_seconds"] == 0.0

    def test_same_timestamp_transition_last_seq_classification(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        # Last reading at :30 is None: the unknown state wins over the above one.
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 30)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", 30)
        insert_reading(db, sid, "a", 3, "2026-01-01T10:00:30Z", None)
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 30.0
        assert body["unknown_seconds"] == 30.0

    def test_value_key_on_object_readings(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", {"temp": 30})
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", {"temp": 5})
        body = exposure(
            c, sid, source_id="a", value_key="temp", **window()
        ).json()
        assert body["value_key"] == "temp"
        assert body["above_seconds"] == 30.0
        assert body["covered_seconds"] == 60.0

    def test_value_key_missing_key_is_unknown(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", {"other": 99})
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", {"temp": 30})
        body = exposure(c, sid, source_id="a", value_key="temp", **window()).json()
        assert body["unknown_seconds"] == 30.0
        assert body["above_seconds"] == 30.0

    def test_value_key_on_scalar_reading_is_unknown(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 99)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", {"temp": 30})
        body = exposure(c, sid, source_id="a", value_key="temp", **window()).json()
        assert body["unknown_seconds"] == 30.0
        assert body["above_seconds"] == 30.0

    def test_value_key_blank_is_400(self, client):
        c, _ = client
        sid = create_session(c, sources=("a",))
        r = exposure(c, sid, source_id="a", value_key="  ", **window())
        assert r.status_code == 400

    def test_buffered_and_quarantined_readings_ignored(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 5)
        # A quarantined high reading at :10 must not enter the timeline.
        insert_reading(
            db, sid, "a", 2, "2026-01-01T10:00:10Z", 99,
            status="quarantined", published_watermark=None,
            quarantined_watermark="2026-01-01T10:00:05Z",
        )
        # A buffered high reading at :20 must not enter the timeline either.
        insert_reading(
            db, sid, "a", 3, "2026-01-01T10:00:20Z", 99,
            status="buffered", published_watermark=None,
        )
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 0.0
        # The value 5 of the only published reading holds the full window.
        assert body["covered_seconds"] == 60.0
        assert body["unknown_seconds"] == 0.0

    def test_microsecond_precision(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00.000000Z", 5)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:00.000500Z", 30)
        body = exposure(
            c, sid, source_id="a",
            **window(**{
                "from": "2026-01-01T10:00:00Z",
                "to": "2026-01-01T10:00:01Z",
            }),
        ).json()
        assert body["above_seconds"] == pytest.approx(0.9995)
        assert body["covered_seconds"] == 1.0
        assert body["unknown_seconds"] == 0.0

    def test_negative_threshold(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", -5)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", 5)
        body = exposure(c, sid, source_id="a", **window(threshold=0)).json()
        assert body["threshold"] == 0.0
        assert body["above_seconds"] == 30.0
        assert body["covered_seconds"] == 60.0
        assert body["above_ratio"] == 0.5

    def test_no_coverage_ratio_null_and_seconds_zero(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        # No readings of the source at all: the whole window is unknown.
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 0.0
        assert body["covered_seconds"] == 0.0
        assert body["unknown_seconds"] == 60.0
        assert body["above_ratio"] is None

    def test_covered_plus_unknown_equals_window_length_invariant(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:05Z", 30)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:17Z", None)
        insert_reading(db, sid, "a", 3, "2026-01-01T10:00:42Z", 5)
        body = exposure(
            c, sid, source_id="a",
            **window(**{"from": "2026-01-01T10:00:00Z", "to": "2026-01-01T10:00:50Z"}),
        ).json()
        assert body["covered_seconds"] + body["unknown_seconds"] == pytest.approx(50.0)
        assert (
            body["above_seconds"]
            + (body["covered_seconds"] - body["above_seconds"])
            + body["unknown_seconds"]
        ) == pytest.approx(50.0)

    def test_prior_unknown_reading_inherited_as_unknown(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        # Nearest reading before the window is a missing value.
        insert_reading(db, sid, "a", 1, "2026-01-01T09:59:59Z", None)
        insert_reading(db, sid, "a", 2, "2026-01-01T10:00:30Z", 30)
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["unknown_seconds"] == 30.0
        assert body["above_seconds"] == 30.0

    def test_works_through_real_publish_and_closed_session(self, client):
        c, _ = client
        sid = create_session(c, sources=("a",), lateness=0)
        publish_via_seal(
            c, sid, "a",
            [
                (1, "2026-01-01T10:00:00Z", 5),
                (2, "2026-01-01T10:00:20Z", 30),
                (3, "2026-01-01T10:00:40Z", 5),
            ],
            final_at="2026-01-01T10:01:00Z",
        )
        assert c.get(f"/sessions/{sid}").json()["closed"] is True
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 20.0
        assert body["covered_seconds"] == 60.0
        assert body["unknown_seconds"] == 0.0

    def test_echo_normalised_utc(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T12:00:00+02:00", 30)
        body = exposure(
            c, sid, source_id="a",
            **{
                "from": "2026-01-01T12:00:00+02:00",
                "to": "2026-01-01T12:01:00+02:00",
                "threshold": "10",
            },
        ).json()
        assert body["from"] == "2026-01-01T10:00:00Z"
        assert body["to"] == "2026-01-01T10:01:00Z"
        assert body["threshold"] == 10.0
        assert body["value_key"] is None
        assert body["above_seconds"] == 60.0


class TestExposureErrors:
    @pytest.mark.parametrize("dropped", ["source_id", "from", "to", "threshold"])
    def test_missing_required_param_400(self, client, dropped):
        c, _ = client
        sid = create_session(c)
        params = window(source_id="a")
        del params[dropped]
        assert exposure(c, sid, **params).status_code == 400

    def test_unknown_session_404(self, client):
        c, _ = client
        r = exposure(c, "missing", source_id="a", **window())
        assert r.status_code == 404

    def test_unknown_source_400(self, client):
        c, _ = client
        sid = create_session(c)
        assert exposure(c, sid, source_id="nope", **window()).status_code == 400

    @pytest.mark.parametrize("bad_source", ["", "   "])
    def test_blank_source_400(self, client, bad_source):
        c, _ = client
        sid = create_session(c)
        assert exposure(c, sid, source_id=bad_source, **window()).status_code == 400

    def test_bad_time_bounds_400(self, client):
        c, _ = client
        sid = create_session(c)
        assert exposure(
            c, sid, source_id="a", **window(**{"from": "not-a-time"})
        ).status_code == 400
        assert exposure(
            c, sid, source_id="a", **window(**{"to": "2026-13-99T00:00:00Z"})
        ).status_code == 400

    @pytest.mark.parametrize("from_v,to_v", [
        ("2026-01-01T10:00:10Z", "2026-01-01T10:00:10Z"),  # empty interval
        ("2026-01-01T10:00:20Z", "2026-01-01T10:00:10Z"),  # backwards
    ])
    def test_from_not_before_to_400(self, client, from_v, to_v):
        c, _ = client
        sid = create_session(c)
        r = exposure(
            c, sid, source_id="a",
            **window(**{"from": from_v, "to": to_v}),
        )
        assert r.status_code == 400

    @pytest.mark.parametrize("bad", ["", "   ", "junk", "nan", "inf", "-infinity", "1e999"])
    def test_bad_threshold_400(self, client, bad):
        c, _ = client
        sid = create_session(c)
        assert exposure(c, sid, source_id="a", **window(threshold=bad)).status_code == 400

    def test_scientific_and_decimal_thresholds_accepted(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 0.005)
        body = exposure(
            c, sid, source_id="a", **window(threshold="1e-3")
        ).json()
        assert body["above_seconds"] == 60.0


class TestExposureReadOnly:
    def test_call_changes_nothing(self, client):
        c, db = client
        sid = create_session(c, sources=("a", "b"))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 30)
        insert_reading(
            db, sid, "a", 2, "2026-01-01T10:00:30Z", 99,
            status="quarantined", published_watermark=None,
            quarantined_watermark="2026-01-01T10:00:20Z",
        )
        # An alert rule exists; the read-only call must not advance or touch it.
        r = c.post(
            f"/sessions/{sid}/alert-rules",
            json={
                "source_id": "a", "direction": "above",
                "enter_threshold": 20, "recover_threshold": 10,
                "enter_consecutive": 1, "recover_consecutive": 1,
            },
        )
        assert r.status_code == 201, r.text

        def snapshot():
            with sqlite3.connect(db) as raw:
                raw.row_factory = sqlite3.Row
                return {
                    name: [dict(r2) for r2 in raw.execute(f"SELECT * FROM {name}")]
                    for name in (
                        "sessions", "session_sources", "readings",
                        "alert_rules", "alert_events", "checkpoints",
                    )
                }

        before = snapshot()
        for _ in range(3):
            assert exposure(c, sid, source_id="a", **window()).status_code == 200
            assert exposure(
                c, sid, source_id="a", value_key="temp",
                **window(**{"from": "2026-01-01T09:00:00Z", "to": "2026-01-01T11:00:00Z"}),
            ).status_code == 200
        assert snapshot() == before

    def test_session_state_and_alerts_unchanged(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 30)
        c.post(
            f"/sessions/{sid}/alert-rules",
            json={
                "source_id": "a", "direction": "above",
                "enter_threshold": 20, "recover_threshold": 10,
                "enter_consecutive": 1, "recover_consecutive": 1,
            },
        )
        session_before = c.get(f"/sessions/{sid}").json()
        rules_before = c.get(f"/sessions/{sid}/alert-rules").json()
        current_before = c.get(f"/sessions/{sid}/alerts/current").json()
        exposure(c, sid, source_id="a", **window())
        assert c.get(f"/sessions/{sid}").json() == session_before
        assert c.get(f"/sessions/{sid}/alert-rules").json() == rules_before
        assert c.get(f"/sessions/{sid}/alerts/current").json() == current_before


class TestExposureSnapshot:
    def test_uncommitted_reading_not_visible(self, client):
        c, db = client
        sid = create_session(c, sources=("a",))
        insert_reading(db, sid, "a", 1, "2026-01-01T10:00:00Z", 5)

        opened = threading.Event()
        release = threading.Event()

        def hold_uncommitted_insert():
            writer = sqlite3.connect(db, timeout=30)
            try:
                writer.execute("BEGIN IMMEDIATE")
                writer.execute(
                    "INSERT INTO readings (session_id, source_id, seq,"
                    " observed_at, payload, status, first_received_at,"
                    " published_watermark, quarantined_watermark)"
                    " VALUES ('s', 'a', 2, ?, '30', 'published', ?, ?, '')",
                    (
                        t("2026-01-01T10:00:30Z"),
                        t("2026-01-01T09:00:00Z"),
                        t("2026-01-01T11:00:00Z"),
                    ),
                )
                opened.set()
                release.wait(2.0)
                writer.commit()
            finally:
                writer.close()

        thread = threading.Thread(target=hold_uncommitted_insert)
        thread.start()
        try:
            assert opened.wait(2.0)
            # The uncommitted above-threshold reading must not be visible:
            # the whole window stays covered by value 5 (below threshold).
            body = exposure(c, sid, source_id="a", **window()).json()
            assert body["above_seconds"] == 0.0
            assert body["covered_seconds"] == 60.0
        finally:
            release.set()
            thread.join()

        # After commit the second reading is visible (above for 30s).
        body = exposure(c, sid, source_id="a", **window()).json()
        assert body["above_seconds"] == 30.0
        assert body["covered_seconds"] == 60.0
