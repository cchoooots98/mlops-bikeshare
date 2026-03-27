from __future__ import annotations

from datetime import timedelta
from functools import lru_cache, partial

MAX_CRON_LOOKBACK_MINUTES = 366 * 24 * 60
CRON_PRESETS = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
}


def execution_date_fn_for_schedule(upstream_schedule: str):
    normalized_schedule = _normalize_schedule(upstream_schedule)
    return partial(resolve_latest_upstream_logical_date, upstream_schedule=normalized_schedule)


def resolve_latest_upstream_logical_date(current_logical_date, *, upstream_schedule: str, **context):
    """
    Resolve the upstream logical date for the latest run whose interval has ended.

    We intentionally align on the upstream schedule itself instead of a hand-maintained
    execution_delta so downstream sensors stay correct when cron expressions move.
    """
    current_end = context.get("data_interval_end")
    if current_end is None:
        raise ValueError("data_interval_end is required to resolve the upstream logical date")

    latest_upstream_end = _previous_matching_time(upstream_schedule, current_end, inclusive=True)
    return _previous_matching_time(upstream_schedule, latest_upstream_end, inclusive=False)


def _normalize_schedule(schedule: str) -> str:
    return CRON_PRESETS.get(schedule.strip().lower(), schedule.strip())


@lru_cache(maxsize=None)
def _parse_schedule(schedule: str) -> tuple[set[int], set[int], set[int], set[int], set[int], bool, bool]:
    minute, hour, day_of_month, month, day_of_week = _normalize_schedule(schedule).split()
    return (
        _parse_field(minute, 0, 59),
        _parse_field(hour, 0, 23),
        _parse_field(day_of_month, 1, 31),
        _parse_field(month, 1, 12),
        _parse_field(day_of_week, 0, 7, is_day_of_week=True),
        day_of_month == "*",
        day_of_week == "*",
    )


def _parse_field(field: str, minimum: int, maximum: int, *, is_day_of_week: bool = False) -> set[int]:
    values: set[int] = set()
    for part in field.split(","):
        values.update(_parse_part(part.strip(), minimum, maximum))

    if is_day_of_week:
        values = {0 if value == 7 else value for value in values}

    return values


def _parse_part(part: str, minimum: int, maximum: int) -> set[int]:
    if not part:
        raise ValueError("empty cron field part")

    step = 1
    base = part
    if "/" in part:
        base, step_str = part.split("/", 1)
        step = int(step_str)
        if step <= 0:
            raise ValueError(f"invalid cron step: {part}")

    if base == "*":
        start, end = minimum, maximum
    elif "-" in base:
        start_str, end_str = base.split("-", 1)
        start, end = int(start_str), int(end_str)
    else:
        start = end = int(base)

    if start < minimum or end > maximum or start > end:
        raise ValueError(f"invalid cron range: {part}")

    return set(range(start, end + 1, step))


def _matches(schedule: str, candidate) -> bool:
    minute_values, hour_values, day_of_month_values, month_values, day_of_week_values, dom_any, dow_any = (
        _parse_schedule(schedule)
    )
    cron_weekday = (candidate.weekday() + 1) % 7
    dom_match = candidate.day in day_of_month_values
    dow_match = cron_weekday in day_of_week_values

    if dom_any and dow_any:
        day_match = True
    elif dom_any:
        day_match = dow_match
    elif dow_any:
        day_match = dom_match
    else:
        day_match = dom_match or dow_match

    return (
        candidate.minute in minute_values
        and candidate.hour in hour_values
        and candidate.month in month_values
        and day_match
    )


def _previous_matching_time(schedule: str, reference, *, inclusive: bool):
    cursor = reference.replace(second=0, microsecond=0)
    if not inclusive:
        cursor -= timedelta(minutes=1)

    for _ in range(MAX_CRON_LOOKBACK_MINUTES):
        if _matches(schedule, cursor):
            return cursor
        cursor -= timedelta(minutes=1)

    raise ValueError(f"could not resolve a previous run for schedule={schedule!r}")
