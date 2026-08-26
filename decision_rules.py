"""Business-facing scenario labels and event decision helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from forecasting import scenario_category


def _is_blank(value: Any) -> bool:
    return value is None or pd.isna(value) or str(value).strip() in {"", "nan", "None"}


def event_snapshot(day: pd.DataFrame, event_time: Any) -> dict[str, float | int | None]:
    """Summarise before/remaining intervals without inventing future facts."""

    if day.empty or _is_blank(event_time):
        return {
            "before_intervals": 0,
            "remaining_intervals": 0,
            "before_actual_mwh": None,
            "before_forecast_mwh": None,
            "remaining_forecast_mwh": None,
            "remaining_actual_mwh": None,
            "remaining_contract_mwh": None,
        }
    frame = day.copy()
    frame["datetime_15m_start"] = pd.to_datetime(frame["datetime_15m_start"], errors="coerce")
    event = pd.Timestamp(event_time)
    before = frame.loc[frame["datetime_15m_start"] < event]
    remaining = frame.loc[frame["datetime_15m_start"] >= event]

    def energy(part: pd.DataFrame, column: str) -> float:
        return float(part[column].sum() * 0.25)

    return {
        "before_intervals": len(before),
        "remaining_intervals": len(remaining),
        "before_actual_mwh": energy(before, "actual_power_mw"),
        "before_forecast_mwh": energy(before, "forecast_power_mw"),
        "remaining_forecast_mwh": energy(remaining, "forecast_power_mw"),
        "remaining_actual_mwh": energy(remaining, "actual_power_mw"),
        "remaining_contract_mwh": energy(remaining, "wholesale_contract_power_mw"),
    }


def decision_items(calendar_row: pd.Series | None, snapshot: dict[str, Any]) -> list[dict[str, str]]:
    """Return a concrete decision card, not a generic task checklist."""

    scenario = scenario_category(calendar_row)
    if scenario == "正常日":
        return [
            {
                "阶段": "常态判断",
                "触发条件": "误差和覆盖均在Demo阈值内",
                "需要判断": "维持常态跟踪，不因单个15分钟波动改变采购判断",
                "输出": "保留当日误差和覆盖结果，作为后续客户基线",
            }
        ]
    if scenario == "计划变化日":
        return [
            {
                "阶段": "信息确认",
                "触发条件": "客户计划、订单、检修或客流变化已在预测签发前获知",
                "需要判断": "变化影响哪些时段，是否会持续到下一个工作日",
                "输出": "更新已知状态假设，并重算剩余时段预测与覆盖",
            },
            {
                "阶段": "交易输入",
                "触发条件": "更新后的预测改变申报或采购匹配",
                "需要判断": "变化是否超过公司设定的调整阈值和可用交易窗口",
                "输出": "把变化量、影响时段和信息来源交给有权限人员复核",
            },
        ]
    return [
        {
            "阶段": "事件确认",
            "触发条件": "事件已获知，但恢复时间尚未确认",
            "需要判断": "停机范围、保底负荷、最早/最晚恢复时间和恢复方式",
            "输出": "建立不恢复与部分恢复两种剩余负荷情景",
        },
        {
            "阶段": "仓位重算",
            "触发条件": "连续两个15分钟点明显低于原预测，或客户确认停机",
            "需要判断": "剩余预测、客户级采购分摊和组合仓位是否出现方向性变化",
            "输出": "给出剩余影响电量、受影响时段和下一次更新时间",
        },
        {
            "阶段": "复核边界",
            "触发条件": "客户级结果显示采购可能偏多",
            "需要判断": "是否存在其他客户抵消、可用交易窗口和调整权限",
            "输出": "形成交易负责人复核输入，不从客户级结果直接下单",
        },
    ]
