"""Small, reproducible forecasting helpers for the public demo.

The functions in this module deliberately implement an interpretable
time-based baseline rather than a production forecasting model.  Every
reference observation is strictly earlier than the target day, so the
result can be used to demonstrate the forecasting workflow without claiming
real-market accuracy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


NORMAL_FORECAST_STATE = {
    "MFG01": "NORMAL_PRODUCTION",
    "CONT01": "STEADY",
    "COMM01": "NORMAL_BUSINESS",
}


def _is_blank(value: Any) -> bool:
    return value is None or pd.isna(value) or str(value).strip() in {"", "nan", "None"}


def scenario_category(calendar_row: pd.Series | None) -> str:
    """Classify a day using only the event information available in the calendar."""

    if calendar_row is None:
        return "正常日"
    event_type = calendar_row.get("event_type")
    if _is_blank(event_type):
        return "正常日"
    known = calendar_row.get("event_known_at_forecast_flag")
    if str(known).strip() in {"0", "0.0", "False"}:
        return "突发事件日"
    return "计划变化日"


def prepare_customer_frame(customer_data: pd.DataFrame) -> pd.DataFrame:
    """Add stable time keys used by the baseline forecast."""

    frame = customer_data.copy()
    frame["datetime_15m_start"] = pd.to_datetime(frame["datetime_15m_start"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["quarter_hour"] = (
        frame["datetime_15m_start"].dt.hour * 4
        + frame["datetime_15m_start"].dt.minute // 15
    )
    frame["weekday_num"] = frame["datetime_15m_start"].dt.weekday
    return frame.sort_values("datetime_15m_start").reset_index(drop=True)


def prepare_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    result = calendar.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    for column in ("weekend_flag", "holiday_flag"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    return result


def _target_calendar_row(calendar: pd.DataFrame, target_date: Any) -> pd.Series | None:
    target = pd.Timestamp(target_date).date()
    match = calendar.loc[calendar["date"] == target]
    return None if match.empty else match.iloc[0]


def _reference_dates(
    calendar: pd.DataFrame,
    target_date: Any,
    lookback: int,
) -> list[Any]:
    """Return prior comparable dates; never include the target or later dates."""

    target = pd.Timestamp(target_date).date()
    target_row = _target_calendar_row(calendar, target)
    prior = calendar.loc[calendar["date"] < target].copy()
    if target_row is None or prior.empty:
        return []

    target_is_special = bool(target_row.get("weekend_flag", 0)) or bool(target_row.get("holiday_flag", 0))
    if target_is_special:
        candidates = prior.loc[
            (prior["weekend_flag"] == int(target_row.get("weekend_flag", 0)))
            & (prior["holiday_flag"] == int(target_row.get("holiday_flag", 0)))
        ]
    else:
        prior_weekday = pd.to_datetime(prior["date"]).dt.weekday
        candidates = prior.loc[
            (prior["weekend_flag"] == 0)
            & (prior["holiday_flag"] == 0)
            & (prior_weekday == target.weekday())
        ]

    dates = candidates["date"].drop_duplicates().sort_values(ascending=False).head(lookback).tolist()
    if len(dates) >= 2:
        return dates

    # Early-year dates may not have enough same-type observations. Fall back
    # to the most recent prior dates, still strictly before the target.
    return prior["date"].drop_duplicates().sort_values(ascending=False).head(lookback).tolist()


def _state_factor(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    customer_id: str,
    target_date: Any,
) -> float:
    """Estimate a known-state multiplier from prior actual days only.

    The target state is taken from ``forecast_state_assumption`` rather than
    the realised ``day_state``. For an unplanned outage this remains the
    normal operating state, which prevents future information leakage.
    """

    target = pd.Timestamp(target_date).date()
    target_row = _target_calendar_row(calendar, target)
    if target_row is None:
        return 1.0
    target_state = target_row.get("forecast_state_assumption")
    if _is_blank(target_state):
        target_state = target_row.get("day_state")
    normal_state = NORMAL_FORECAST_STATE.get(customer_id)
    if _is_blank(target_state) or target_state == normal_state:
        return 1.0

    daily_energy = (
        frame.loc[frame["date"] < target]
        .groupby("date", as_index=False)["actual_power_mw"]
        .sum()
        .rename(columns={"actual_power_mw": "actual_energy_mwh"})
    )
    daily_energy["actual_energy_mwh"] *= 0.25
    prior_calendar = calendar.loc[calendar["date"] < target, ["date", "forecast_state_assumption"]]
    daily_energy = daily_energy.merge(prior_calendar, on="date", how="left")
    state_values = daily_energy.loc[
        daily_energy["forecast_state_assumption"].astype(str) == str(target_state),
        "actual_energy_mwh",
    ]
    normal_values = daily_energy.loc[
        daily_energy["forecast_state_assumption"].astype(str) == str(normal_state),
        "actual_energy_mwh",
    ]
    if len(state_values) < 3 or len(normal_values) < 3 or normal_values.median() == 0:
        return 1.0
    factor = float(state_values.median() / normal_values.median())
    return max(0.50, min(1.80, factor))


def _state_factors_by_date(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    customer_id: str,
) -> dict[Any, float]:
    """Precompute state multipliers once for a full chronological backtest."""

    daily_energy = (
        frame.groupby("date", as_index=False)["actual_power_mw"]
        .sum()
        .rename(columns={"actual_power_mw": "actual_energy_mwh"})
    )
    daily_energy["actual_energy_mwh"] *= 0.25
    state_frame = daily_energy.merge(
        calendar[["date", "forecast_state_assumption"]], on="date", how="left"
    ).sort_values("date")
    normal_state = NORMAL_FORECAST_STATE.get(customer_id)
    factors: dict[Any, float] = {}
    for target in state_frame["date"].dropna().tolist():
        target_state = state_frame.loc[state_frame["date"] == target, "forecast_state_assumption"].iloc[0]
        if _is_blank(target_state) or target_state == normal_state:
            factors[target] = 1.0
            continue
        prior = state_frame.loc[state_frame["date"] < target]
        state_values = prior.loc[
            prior["forecast_state_assumption"].astype(str) == str(target_state),
            "actual_energy_mwh",
        ]
        normal_values = prior.loc[
            prior["forecast_state_assumption"].astype(str) == str(normal_state),
            "actual_energy_mwh",
        ]
        if len(state_values) < 3 or len(normal_values) < 3 or normal_values.median() == 0:
            factors[target] = 1.0
        else:
            factor = float(state_values.median() / normal_values.median())
            factors[target] = max(0.50, min(1.80, factor))
    return factors


def _forecast_day_prepared(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    target_date: Any,
    lookback: int = 4,
    state_factor: float | None = None,
) -> pd.DataFrame:
    target = pd.Timestamp(target_date).date()
    day = frame.loc[frame["date"] == target].copy()
    if day.empty:
        return day

    references = _reference_dates(calendar, target, lookback)
    history = frame.loc[frame["date"].isin(references)]
    profile = history.groupby("quarter_hour")["actual_power_mw"].median()
    day["baseline_forecast_power_mw"] = day["quarter_hour"].map(profile)
    target_calendar = _target_calendar_row(calendar, target)
    customer_id = "" if target_calendar is None else str(target_calendar.get("customer_id", ""))
    factor = _state_factor(frame, calendar, customer_id, target) if state_factor is None else state_factor
    day["known_state_factor"] = factor
    day["state_adjusted_forecast_power_mw"] = day["baseline_forecast_power_mw"] * factor
    day["forecast_power_mw"] = day["state_adjusted_forecast_power_mw"]
    day["forecast_reference_days"] = len(references)
    day["forecast_method"] = "前4个同类日同一时刻中位数＋已知状态修正"
    return day


def forecast_day(
    customer_data: pd.DataFrame,
    calendar: pd.DataFrame,
    target_date: Any,
    lookback: int = 4,
) -> pd.DataFrame:
    """Forecast one day using prior comparable-day medians.

    Returned columns include the neutral baseline and a known-state-adjusted
    version. The latter uses only the forecast-state assumption available for
    the target day; it never uses the realised target-day state.
    """

    frame = prepare_customer_frame(customer_data)
    cal = prepare_calendar(calendar)
    return _forecast_day_prepared(frame, cal, target_date, lookback=lookback)


def _customer_id_from_frame(frame: pd.DataFrame) -> str:
    if "customer_id" not in frame or frame["customer_id"].dropna().empty:
        return ""
    return str(frame["customer_id"].dropna().iloc[0])


def backtest_customer(
    customer_data: pd.DataFrame,
    calendar: pd.DataFrame,
    minimum_reference_days: int = 2,
) -> pd.DataFrame:
    """Run a chronological, day-by-day backtest for one customer."""

    frame = prepare_customer_frame(customer_data)
    cal = prepare_calendar(calendar)
    customer_id = _customer_id_from_frame(frame)
    if not customer_id and "customer_id" in cal and not cal["customer_id"].dropna().empty:
        customer_id = str(cal["customer_id"].dropna().iloc[0])
    state_factors = _state_factors_by_date(frame, cal, customer_id)
    rows: list[dict[str, Any]] = []
    for target in sorted(frame["date"].dropna().unique()):
        forecast = _forecast_day_prepared(frame, cal, target, state_factor=state_factors.get(target, 1.0))
        if forecast.empty or forecast["baseline_forecast_power_mw"].isna().any():
            continue
        reference_days = int(forecast["forecast_reference_days"].iloc[0])
        if reference_days < minimum_reference_days:
            continue
        actual = forecast["actual_power_mw"]
        actual_energy = float(actual.sum() * 0.25)
        target_calendar = _target_calendar_row(cal, target)
        row = {
            "date": target,
            "customer_id": customer_id,
            "scenario": scenario_category(target_calendar),
            "day_state": None if target_calendar is None else target_calendar.get("day_state"),
            "event_type": None if target_calendar is None else target_calendar.get("event_type"),
            "forecast_state_assumption": None if target_calendar is None else target_calendar.get("forecast_state_assumption"),
            "reference_days": reference_days,
            "actual_energy_mwh": actual_energy,
        }
        for method, column in (
            ("基准预测", "baseline_forecast_power_mw"),
            ("已知状态修正", "state_adjusted_forecast_power_mw"),
        ):
            error = forecast[column] - actual
            absolute_error_mwh = float(error.abs().sum() * 0.25)
            row[f"{method}_energy_mwh"] = float(forecast[column].sum() * 0.25)
            row[f"{method}_absolute_error_mwh"] = absolute_error_mwh
            row[f"{method}_wape_ratio"] = absolute_error_mwh / actual_energy if actual_energy else pd.NA
            row[f"{method}_mae_mw"] = float(error.abs().mean())
            row[f"{method}_signed_error_mwh"] = float(error.sum() * 0.25)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_backtest(backtest: pd.DataFrame, method: str = "已知状态修正") -> pd.DataFrame:
    """Aggregate backtest results by customer and operating scenario."""

    if backtest.empty:
        return pd.DataFrame()
    groups = []
    for (customer_id, scenario), group in backtest.groupby(["customer_id", "scenario"], dropna=False):
        groups.append(
            {
                "customer_id": customer_id,
                "scenario": scenario,
                "days": len(group),
                "actual_energy_mwh": group["actual_energy_mwh"].sum(),
                "absolute_error_mwh": group[f"{method}_absolute_error_mwh"].sum(),
                "wape_ratio": group[f"{method}_absolute_error_mwh"].sum() / group["actual_energy_mwh"].sum(),
                "mae_mw": group[f"{method}_mae_mw"].mean(),
                "p90_daily_wape_ratio": group[f"{method}_wape_ratio"].quantile(0.90),
            }
        )
    return pd.DataFrame(groups)
