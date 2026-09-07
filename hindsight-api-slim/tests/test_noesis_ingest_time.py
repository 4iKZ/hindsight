"""Noesis event_time resolution tests (requirement 02 §9.2 / §17.3).

``resolve_event_time`` is a pure function over observed_at + modifier atoms +
an injected analyzer, so the whole matrix runs offline. A few integration
cases exercise the real ``DateparserQueryAnalyzer`` with Chinese time text.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from hindsight_api.engine.retain.noesis_ingest import resolve_event_time
from tests.noesis_fakes import OBSERVED_AT, FakeAnalyzer, constraint

SHANGHAI = ZoneInfo("Asia/Shanghai")


def atom(pos: int, text: str, role: str = "modifier", atom_type: str = "E", target_occ: int | None = 4):
    from types import SimpleNamespace

    return SimpleNamespace(pos=pos, text=text, type=atom_type, role=role, target_occ=target_occ, resolved=None)


def root_predicate(pos: int = 4, text: str = "买"):
    return atom(pos, text, role="predicate", atom_type="P", target_occ=None)


def yesterday_start_utc() -> datetime:
    """昨天 with observed_at = 2026-09-05T10:00+08:00 → 2026-09-04T00:00+08:00 → UTC."""
    return datetime(2026, 9, 4, 0, 0, tzinfo=SHANGHAI).astimezone(UTC)


def codes(warnings) -> list[str]:
    return [warning["alert_code"] for warning in warnings]


# ---------------------------------------------------------------------------
# Pure resolution matrix
# ---------------------------------------------------------------------------

def test_no_temporal_modifier_uses_observed_at():
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[root_predicate(), atom(5, "苹果", role="patient")],
        analyzer=FakeAnalyzer(),
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == OBSERVED_AT
    assert resolution.metadata["strategy"] == "observed_at"
    assert resolution.metadata["matched_atoms"] == []
    assert resolution.warnings == []


def test_non_temporal_modifier_does_not_change_event_time():
    analyzer = FakeAnalyzer(mapping={"在超市": None})  # parses to no constraint
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[root_predicate(), atom(3, "在超市")],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == OBSERVED_AT
    assert resolution.metadata["strategy"] == "observed_at"


def test_yesterday_resolved_with_observed_at_as_reference():
    analyzer = FakeAnalyzer(mapping={"昨天": constraint(datetime(2026, 9, 4, 0, 0), datetime(2026, 9, 4, 23, 59, 59, 999999))})
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "昨天"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    # reference_date is observed_at anchored in the business timezone
    # (naive, because DateparserQueryAnalyzer works on naive datetimes).
    assert analyzer.calls == [("昨天", datetime(2026, 9, 5, 10, 0))]
    assert resolution.event_time == yesterday_start_utc()
    assert resolution.metadata["strategy"] == "modifier_atom"
    assert resolution.metadata["matched_atoms"] == ["昨天"]
    assert resolution.metadata["start"] == "2026-09-03T16:00:00Z"
    assert resolution.metadata["end"] == "2026-09-04T15:59:59.999999Z"
    assert resolution.warnings == []


def test_absolute_date_used_as_start():
    analyzer = FakeAnalyzer(mapping={"2026年9月1日": constraint(datetime(2026, 9, 1, 0, 0), datetime(2026, 9, 1, 23, 59, 59, 999999))})
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "2026年9月1日"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == datetime(2026, 9, 1, 0, 0, tzinfo=SHANGHAI).astimezone(UTC)


def test_same_constraint_from_two_atoms_deduped():
    same = constraint(datetime(2026, 9, 4, 0, 0), datetime(2026, 9, 4, 23, 59, 59, 999999))
    analyzer = FakeAnalyzer(mapping={"昨天": same, "昨日": same})
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "昨天"), atom(3, "昨日"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == yesterday_start_utc()
    assert resolution.metadata["strategy"] == "modifier_atom"
    assert sorted(resolution.metadata["matched_atoms"]) == ["昨天", "昨日"]
    assert resolution.warnings == []


def test_conflicting_constraints_fallback_with_warning():
    analyzer = FakeAnalyzer(
        mapping={
            "昨天": constraint(datetime(2026, 9, 4, 0, 0)),
            "2026年9月1日": constraint(datetime(2026, 9, 1, 0, 0)),
        }
    )
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "昨天"), atom(3, "2026年9月1日"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == OBSERVED_AT
    assert resolution.metadata["strategy"] == "fallback"
    assert resolution.metadata["fallback_reason"] is not None
    assert codes(resolution.warnings) == ["event_time_conflict"]


def test_parser_exception_fallback_with_warning():
    analyzer = FakeAnalyzer(error=RuntimeError("dateparser exploded"))
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "昨天"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == OBSERVED_AT
    assert resolution.metadata["strategy"] == "fallback"
    assert codes(resolution.warnings) == ["event_time_parse_failed"]


def test_aware_result_converted_to_utc():
    aware_start = datetime(2026, 9, 4, 0, 0, tzinfo=SHANGHAI)
    analyzer = FakeAnalyzer(mapping={"昨天": constraint(aware_start, aware_start + timedelta(days=1))})
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "昨天"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == yesterday_start_utc()


def test_suspicious_future_time_fallback_with_warning():
    future_start = OBSERVED_AT + timedelta(hours=48)
    analyzer = FakeAnalyzer(mapping={"三天后": constraint(future_start, future_start + timedelta(days=1))})
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "三天后"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == OBSERVED_AT
    assert resolution.metadata["strategy"] == "fallback"
    assert codes(resolution.warnings) == ["event_time_out_of_range"]


def test_timezone_config_switch_parses_naive_in_configured_zone():
    """Naive parse results are interpreted in the configured business zone."""
    analyzer = FakeAnalyzer(mapping={"昨天": constraint(datetime(2026, 9, 4, 0, 0), datetime(2026, 9, 4, 23, 59, 59, 999999))})
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "昨天"), root_predicate()],
        analyzer=analyzer,
        timezone_name="UTC",
    )
    assert resolution.event_time == datetime(2026, 9, 4, 0, 0, tzinfo=UTC)


def test_invalid_timezone_name_falls_back_to_config_alert_free_behavior():
    """An unusable timezone name must not crash; observed_at fallback wins."""
    analyzer = FakeAnalyzer(mapping={"昨天": constraint(datetime(2026, 9, 4, 0, 0))})
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "昨天"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Not/AZone",
    )
    assert resolution.event_time == OBSERVED_AT
    assert codes(resolution.warnings) == ["event_time_parse_failed"]


def test_no_atoms_at_all_uses_observed_at():
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT, atoms=[], analyzer=FakeAnalyzer(), timezone_name="Asia/Shanghai"
    )
    assert resolution.event_time == OBSERVED_AT
    assert resolution.metadata["strategy"] == "observed_at"


# ---------------------------------------------------------------------------
# Real analyzer integration (Chinese time text)
# ---------------------------------------------------------------------------

def test_real_dateparser_yesterday_chinese():
    from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer

    analyzer = DateparserQueryAnalyzer()
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "昨天"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time.astimezone(SHANGHAI).date() == datetime(2026, 9, 4, tzinfo=SHANGHAI).date()
    assert resolution.event_time.tzinfo is not None
    assert resolution.metadata["strategy"] == "modifier_atom"


def test_real_dateparser_non_time_text_has_no_constraint():
    from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer

    analyzer = DateparserQueryAnalyzer()
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(3, "在超市"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.event_time == OBSERVED_AT
    assert resolution.metadata["strategy"] == "observed_at"


def test_real_dateparser_last_week_range():
    from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer

    analyzer = DateparserQueryAnalyzer()
    resolution = resolve_event_time(
        observed_at=OBSERVED_AT,
        atoms=[atom(1, "上周"), root_predicate()],
        analyzer=analyzer,
        timezone_name="Asia/Shanghai",
    )
    assert resolution.metadata["strategy"] == "modifier_atom"
    assert resolution.event_time < OBSERVED_AT
    # range metadata keeps both endpoints
    assert resolution.metadata["start"] is not None
    assert resolution.metadata["end"] is not None
