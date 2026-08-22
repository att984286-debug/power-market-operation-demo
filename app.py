from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "售电运营数据分析原型"
DATA_FILENAME = "售电交易辅助分析系统——2026全年模拟数据 v1.0.xlsx"
# The deployed package is self-contained: app.py and data/ live together.
# Path(__file__) is resolved at runtime and contains no machine-specific user path.
DATA_PATH = Path(__file__).resolve().parent / "data" / DATA_FILENAME
CUSTOMERS = ["MFG01", "CONT01", "COMM01"]
CUSTOMER_LABELS = {
    "MFG01": "MFG01｜皖新精密制造（模拟）",
    "CONT01": "CONT01｜皖恒新材料（模拟）",
    "COMM01": "COMM01｜合创商务中心（模拟）",
}
STATE_LABELS = {
    "NORMAL_PRODUCTION": "正常生产",
    "HIGH_ORDER": "高订单/高负荷",
    "LOW_LOAD": "低负荷",
    "MAINTENANCE": "计划检修",
    "OUTAGE": "异常停机",
    "RECOVERY": "恢复",
    "STEADY": "稳态运行",
    "HIGH_LOAD": "高负荷",
    "DERATED": "降负荷",
    "NORMAL_BUSINESS": "正常营业",
    "HIGH_FOOTFALL": "高客流/高负荷",
    "LOW_FOOTFALL": "低客流",
    "MAINTENANCE_WINDOW": "夜间检修窗口",
}
CRITICAL_STATES = {"OUTAGE", "MAINTENANCE", "MAINTENANCE_WINDOW"}
WATCH_STATES = {
    "HIGH_ORDER",
    "LOW_LOAD",
    "RECOVERY",
    "HIGH_LOAD",
    "DERATED",
    "HIGH_FOOTFALL",
    "LOW_FOOTFALL",
}


def _read_sheet(path: Path, sheet_name: str, header: int = 2) -> pd.DataFrame:
    """Read a source sheet without writing back to the workbook."""
    return pd.read_excel(path, sheet_name=sheet_name, header=header)


@st.cache_data(show_spinner=False)
def load_data(path_str: str) -> dict[str, pd.DataFrame]:
    """Load the existing workbook read-only and normalize dates/numeric columns."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"未找到年度模拟数据工作簿：{path}")

    params = _read_sheet(path, "客户冻结参数")
    contracts = _read_sheet(path, "合同口径")
    state_calendar = _read_sheet(path, "年度状态日历")
    state_config = _read_sheet(path, "年度状态配置")
    daily = _read_sheet(path, "日汇总")
    monthly = _read_sheet(path, "月汇总")
    qa = _read_sheet(path, "质量检查")
    # This sheet has its header in the first row rather than the title/header layout
    # used by the summary sheets.
    intraday = _read_sheet(path, "组合15分钟数据", header=0)

    for frame in (state_calendar, daily):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    monthly["month"] = monthly["month"].astype(str)
    intraday["datetime_15m_start"] = pd.to_datetime(
        intraday["datetime_15m_start"], errors="coerce"
    )
    intraday["date"] = pd.to_datetime(intraday["date"], errors="coerce").dt.date
    for column in ("weather_available_time", "forecast_issue_time", "submitted_time"):
        intraday[column] = pd.to_datetime(intraday[column], errors="coerce")
    state_calendar["event_known_time"] = pd.to_datetime(
        state_calendar["event_known_time"], errors="coerce"
    )

    numeric_columns = [
        "daily_energy_mwh",
        "average_power_mw",
        "max_power_mw",
        "forecast_mape_ratio",
        "contract_coverage_ratio",
        "absolute_deviation_energy_mwh",
        "deviation_risk_exposure_yuan",
    ]
    for column in numeric_columns:
        if column in daily:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
    for column in ("energy_mwh", "average_power_mw", "max_power_mw", "forecast_mape_ratio", "contract_coverage_ratio", "absolute_deviation_energy_mwh", "risk_exposure_yuan"):
        if column in monthly:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    for customer_id in CUSTOMERS:
        for prefix in (
            f"{customer_id}_actual_power_mw",
            f"{customer_id}_actual_energy_mwh",
            f"{customer_id}_forecast_power_mw",
            f"{customer_id}_submitted_power_mw",
            f"{customer_id}_wholesale_contract_power_mw",
            f"{customer_id}_deviation_energy_mwh",
        ):
            intraday[prefix] = pd.to_numeric(intraday[prefix], errors="coerce")

    return {
        "params": params,
        "contracts": contracts,
        "state_calendar": state_calendar,
        "state_config": state_config,
        "daily": daily,
        "monthly": monthly,
        "qa": qa,
        "intraday": intraday,
    }


def customer_intraday(data: dict[str, pd.DataFrame], customer_id: str) -> pd.DataFrame:
    """Return a long, customer-specific view of the flattened intraday sheet."""
    source = data["intraday"]
    result = source[
        [
            "datetime_15m_start",
            "date",
            "month",
            "temperature_c",
            "weather_type",
            "weather_available_time",
            "forecast_issue_time",
            "submitted_time",
            "day_ahead_price_yuan_per_mwh",
            "real_time_price_yuan_per_mwh",
            f"{customer_id}_actual_power_mw",
            f"{customer_id}_actual_energy_mwh",
            f"{customer_id}_forecast_power_mw",
            f"{customer_id}_submitted_power_mw",
            f"{customer_id}_wholesale_contract_power_mw",
            f"{customer_id}_deviation_energy_mwh",
        ]
    ].copy()
    result.columns = [
        "datetime_15m_start",
        "date",
        "month",
        "temperature_c",
        "weather_type",
        "weather_available_time",
        "forecast_issue_time",
        "submitted_time",
        "day_ahead_price_yuan_per_mwh",
        "real_time_price_yuan_per_mwh",
        "actual_power_mw",
        "actual_energy_mwh",
        "forecast_power_mw",
        "submitted_power_mw",
        "wholesale_contract_power_mw",
        "deviation_energy_mwh",
    ]
    return result


def daily_customer_view(data: dict[str, pd.DataFrame], customer_id: str) -> pd.DataFrame:
    """Build explicit daily metrics, including actual coverage not stored in daily summary."""
    daily = data["daily"].loc[data["daily"]["customer_id"] == customer_id].copy()
    intraday = customer_intraday(data, customer_id)
    integrated = (
        intraday.assign(
            forecast_energy_mwh=intraday["forecast_power_mw"] * 0.25,
            submitted_energy_mwh=intraday["submitted_power_mw"] * 0.25,
            wholesale_contract_energy_mwh=intraday["wholesale_contract_power_mw"] * 0.25,
        )
        .groupby("date", as_index=False)[
            ["forecast_energy_mwh", "submitted_energy_mwh", "wholesale_contract_energy_mwh"]
        ]
        .sum()
    )
    result = daily.merge(integrated, on="date", how="left")
    result["actual_coverage_ratio"] = result["wholesale_contract_energy_mwh"] / result["daily_energy_mwh"]
    result["actual_minus_forecast_mwh"] = result["daily_energy_mwh"] - result["forecast_energy_mwh"]
    result["actual_deviation_ratio"] = result["actual_minus_forecast_mwh"].abs() / result["forecast_energy_mwh"].replace(0, pd.NA)
    return result


def get_calendar_row(data: dict[str, pd.DataFrame], customer_id: str, selected_date: Any) -> pd.Series | None:
    frame = data["state_calendar"]
    match = frame[(frame["customer_id"] == customer_id) & (frame["date"] == selected_date)]
    return match.iloc[0] if not match.empty else None


def risk_level(row: pd.Series, calendar_row: pd.Series | None) -> str:
    """Transparent demo screening label; not an official market/settlement rule."""
    state = str(row.get("day_state", ""))
    event_type = "" if calendar_row is None else str(calendar_row.get("event_type", ""))
    deviation_ratio = float(row.get("actual_deviation_ratio", 0) or 0)
    actual_coverage = float(row.get("actual_coverage_ratio", 1) or 1)
    if state in CRITICAL_STATES or event_type in {"EQUIPMENT_TRIP", "OUTAGE", "MAINTENANCE"}:
        return "高风险"
    if deviation_ratio >= 0.20 or actual_coverage < 0.80 or actual_coverage > 1.20:
        return "高风险"
    if state in WATCH_STATES or deviation_ratio >= 0.05 or actual_coverage < 0.90 or actual_coverage > 1.10:
        return "需关注"
    return "常态"


def risk_explanation(row: pd.Series, calendar_row: pd.Series | None) -> str:
    state = str(row.get("day_state", ""))
    if state == "OUTAGE":
        return "异常停机：实际负荷低于事前预测，需确认生产状态并重算客户级仓位。"
    if state in {"MAINTENANCE", "MAINTENANCE_WINDOW"}:
        return "检修状态：基荷或营业曲线发生变化，应核对检修窗口与剩余时段。"
    if state in {"HIGH_ORDER", "HIGH_LOAD", "HIGH_FOOTFALL"}:
        return "计划性增负荷：需确认订单、产量或客流信息是否已进入申报判断。"
    if state in {"LOW_LOAD", "LOW_FOOTFALL", "DERATED"}:
        return "低负荷/降负荷：需关注合同覆盖偏高和后续恢复节奏。"
    if state == "RECOVERY":
        return "恢复阶段：负荷爬坡可能带来新的预测误差，应跟踪复产曲线。"
    return "未见异常状态，按常态负荷与覆盖指标跟踪。"


def fmt_mwh(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):,.2f} MWh"


def fmt_mw(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):,.2f} MW"


def fmt_pct(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):.1%}"


def fmt_yuan(value: Any) -> str:
    return "—" if pd.isna(value) else f"¥{float(value):,.2f}"


def metric_row(data: dict[str, pd.DataFrame], customer_id: str, selected_date: Any) -> pd.Series:
    view = daily_customer_view(data, customer_id)
    selected = view.loc[view["date"] == selected_date]
    if selected.empty:
        raise KeyError(f"没有找到{customer_id}在{selected_date}的日汇总")
    return selected.iloc[0]


def figure_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def render_dashboard(data: dict[str, pd.DataFrame], selected_date: Any, customer_filter: str) -> None:
    st.header("所选分析日的客户风险总览")
    st.caption("页面中的‘风险等级’为Demo筛选标签，仅用于运营关注排序，不是安徽市场正式考核或结算规则。")

    customer_ids = CUSTOMERS if customer_filter == "全部客户" else [customer_filter]
    rows: list[dict[str, Any]] = []
    for customer_id in customer_ids:
        row = metric_row(data, customer_id, selected_date)
        calendar_row = get_calendar_row(data, customer_id, selected_date)
        risk = risk_level(row, calendar_row)
        rows.append(
            {
                "客户": CUSTOMER_LABELS[customer_id],
                "行业": data["params"].loc[data["params"]["customer_id"] == customer_id, "类型"].iloc[0],
                "当前状态": STATE_LABELS.get(str(row["day_state"]), str(row["day_state"])),
                "负荷风险等级": risk,
                "预测覆盖率": fmt_pct(row["contract_coverage_ratio"]),
                "实际覆盖率": fmt_pct(row["actual_coverage_ratio"]),
                "是否需要关注": "是" if risk != "常态" else "否",
            }
        )
    overview = pd.DataFrame(rows)

    total_energy = sum(float(metric_row(data, cid, selected_date)["daily_energy_mwh"]) for cid in customer_ids)
    high_risk_count = int((overview["负荷风险等级"] == "高风险").sum())
    events = data["state_calendar"]
    events = events[(events["date"] == selected_date) & (events["customer_id"].isin(customer_ids))]
    events = events[events["event_type"].notna() & (events["event_type"].astype(str).str.strip() != "")]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("客户数量", len(customer_ids))
    k2.metric("所选日期客户总电量", fmt_mwh(total_energy))
    k3.metric("异常事件数量", len(events))
    k4.metric("高风险客户数量", high_risk_count)

    st.subheader("客户风险总览")
    st.dataframe(overview, width="stretch", hide_index=True)

    st.subheader("所选日期重点事件")
    if events.empty:
        st.success("所选分析日未发现模拟事件台账记录。")
    else:
        for _, event in events.iterrows():
            row = metric_row(data, str(event["customer_id"]), selected_date)
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**客户**\n\n{CUSTOMER_LABELS.get(str(event['customer_id']), event['customer_id'])}")
                c2.markdown(f"**事件时间**\n\n{event['event_known_time']:%Y-%m-%d %H:%M}")
                c3.markdown(f"**实际/预测电量**\n\n{fmt_mwh(row['daily_energy_mwh'])} / {fmt_mwh(row['forecast_energy_mwh'])}")
                c4.markdown(f"**风险**\n\n{risk_level(row, event)}")
                st.caption(
                    f"{event.get('触发条件', '模拟事件')}；{risk_explanation(row, event)} "
                    f"情景风险暴露 {fmt_yuan(row['deviation_risk_exposure_yuan'])}。"
                )


def render_profile(data: dict[str, pd.DataFrame], customer_id: str) -> None:
    st.header("不同客户的运行方式决定不同风险管理方式")
    params = data["params"].loc[data["params"]["customer_id"] == customer_id].iloc[0]
    monthly = data["monthly"].loc[data["monthly"]["customer_id"] == customer_id].copy()
    calendar = data["state_calendar"].loc[data["state_calendar"]["customer_id"] == customer_id].copy()

    st.subheader(CUSTOMER_LABELS[customer_id])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年用电量", fmt_mwh(params["annual_energy_mwh"]))
    c2.metric("最大负荷", fmt_mw(params["max_power_mw"]))
    c3.metric("平均负荷", fmt_mw(params["average_power_mw"]))
    c4.metric("负荷率", fmt_pct(params["load_factor_ratio"]))

    st.markdown(
        f"**行业**：{params['类型']}　　**运行特点**：{params['运行/检修说明']}  "
        f"\n\n**基础负荷**：{fmt_mw(params['base_power_mw'])}　　**正常峰值**：{fmt_mw(params['normal_peak_power_mw'])}"
    )
    risk_text = {
        "MFG01": "订单、班次和设备状态变化会直接改变日内曲线，重点关注预测误差与合同覆盖反转。",
        "CONT01": "稳态基荷使正常日较易预测，但检修或跳停会带来较大的结构性偏差。",
        "COMM01": "营业时段、周末、高温和客流共同影响负荷，重点关注峰段申报与天气敏感性。",
    }[customer_id]
    st.info(f"主要交易运营风险：{risk_text}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["energy_mwh"], mode="lines+markers", name="月用电量"))
    fig.update_yaxes(title="月用电量（MWh）")
    fig.update_xaxes(title="月份")
    st.plotly_chart(figure_layout(fig, 340), width="stretch")

    state_counts = calendar["day_state"].map(lambda x: STATE_LABELS.get(str(x), str(x))).value_counts().reset_index()
    state_counts.columns = ["状态", "天数"]
    st.subheader("年度运行状态分布")
    st.dataframe(state_counts, width="stretch", hide_index=True)


def render_load_analysis(data: dict[str, pd.DataFrame], customer_id: str, selected_date: Any) -> None:
    st.header("96点负荷曲线帮助识别客户状态变化")
    customer_data = customer_intraday(data, customer_id)
    available_dates = sorted(customer_data["date"].dropna().unique())
    curve_date = st.date_input(
        "曲线日期（模拟）",
        value=selected_date,
        min_value=min(available_dates),
        max_value=max(available_dates),
        key="load_curve_date",
    )
    day = customer_data.loc[customer_data["date"] == curve_date].copy()
    calendar_row = get_calendar_row(data, customer_id, curve_date)
    if day.empty:
        st.warning("所选日期没有可展示的15分钟记录。")
        return

    state_text = "—" if calendar_row is None else STATE_LABELS.get(str(calendar_row["day_state"]), str(calendar_row["day_state"]))
    event_text = "无模拟事件" if calendar_row is None or pd.isna(calendar_row.get("event_type")) else str(calendar_row["event_type"])
    st.caption(f"客户：{CUSTOMER_LABELS[customer_id]}｜状态：{state_text}｜事件：{event_text}")

    fig = go.Figure()
    for column, label, color in (
        ("actual_power_mw", "实际负荷", "#0B6E99"),
        ("forecast_power_mw", "预测负荷", "#D97706"),
        ("submitted_power_mw", "日前申报", "#7C3AED"),
    ):
        fig.add_trace(
            go.Scatter(
                x=day["datetime_15m_start"].dt.strftime("%H:%M"),
                y=day[column],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            )
        )
    fig.update_yaxes(title="功率（MW）")
    fig.update_xaxes(title="时刻")
    st.plotly_chart(figure_layout(fig, 430), width="stretch")

    # A compact workday/weekend comparison, using the existing simulated calendar.
    # The flattened intraday sheet has one shared timestamp; join this customer's
    # simulated calendar explicitly for the workday/weekend comparison.
    all_days = customer_data.merge(
        data["state_calendar"].loc[data["state_calendar"]["customer_id"] == customer_id, ["date", "weekend_flag"]],
        on="date",
        how="left",
    )
    all_days["quarter_hour"] = all_days["datetime_15m_start"].dt.hour * 4 + all_days["datetime_15m_start"].dt.minute // 15
    profile = all_days.groupby(["weekend_flag", "quarter_hour"], as_index=False)["actual_power_mw"].mean()
    profile["类型"] = profile["weekend_flag"].map({0: "工作日", 1: "周末/节假日"})
    profile["时刻"] = profile["quarter_hour"].map(lambda n: f"{int(n)//4:02d}:{int(n)%4*15:02d}")
    fig_profile = go.Figure()
    for label, color in (("工作日", "#0B6E99"), ("周末/节假日", "#94A3B8")):
        part = profile.loc[profile["类型"] == label]
        fig_profile.add_trace(go.Scatter(x=part["时刻"], y=part["actual_power_mw"], mode="lines", name=label, line=dict(color=color)))
    fig_profile.update_yaxes(title="平均功率（MW）")
    fig_profile.update_xaxes(title="时刻")
    st.plotly_chart(figure_layout(fig_profile, 320), width="stretch")

    st.subheader("运营含义")
    daily_row = metric_row(data, customer_id, curve_date)
    st.info(risk_explanation(daily_row, calendar_row))

    events = data["state_calendar"].loc[
        (data["state_calendar"]["customer_id"] == customer_id)
        & data["state_calendar"]["event_type"].notna()
        & (data["state_calendar"]["event_type"].astype(str).str.strip() != "")
    ][["date", "day_state", "event_type", "触发条件", "event_known_time"]].copy()
    events["day_state"] = events["day_state"].map(lambda x: STATE_LABELS.get(str(x), str(x)))
    events.columns = ["日期", "状态", "事件类型", "触发条件", "事件获知时间"]
    st.subheader("典型异常事件")
    st.dataframe(events.sort_values("日期").tail(8), width="stretch", hide_index=True)


def render_coverage(data: dict[str, pd.DataFrame], selected_date: Any) -> None:
    st.header("合同覆盖需要同时看预测、申报和实际用电")
    st.warning("模拟客户级批发采购分摊分析，不代表真实合同、公司整体仓位或安徽正式结算。")
    customer_id = "MFG01"
    row = metric_row(data, customer_id, selected_date)
    intraday = customer_intraday(data, customer_id).loc[lambda x: x["date"] == selected_date].copy()
    calendar_row = get_calendar_row(data, customer_id, selected_date)

    metrics = [
        ("预测电量", row["forecast_energy_mwh"], fmt_mwh),
        ("日前申报电量", row["submitted_energy_mwh"], fmt_mwh),
        ("实际用电量", row["daily_energy_mwh"], fmt_mwh),
        ("批发采购分摊仓位", row["wholesale_contract_energy_mwh"], fmt_mwh),
        ("事前预测覆盖率", row["contract_coverage_ratio"], fmt_pct),
        ("事后实际覆盖率", row["actual_coverage_ratio"], fmt_pct),
        ("情景风险暴露", row["deviation_risk_exposure_yuan"], fmt_yuan),
    ]
    cols = st.columns(4)
    for index, (label, value, formatter) in enumerate(metrics):
        cols[index % 4].metric(label, formatter(value))

    fig = go.Figure()
    for column, label, color in (
        ("actual_power_mw", "实际负荷", "#0B6E99"),
        ("forecast_power_mw", "预测负荷", "#D97706"),
        ("submitted_power_mw", "日前申报", "#7C3AED"),
        ("wholesale_contract_power_mw", "批发采购分摊", "#059669"),
    ):
        fig.add_trace(
            go.Scatter(
                x=intraday["datetime_15m_start"].dt.strftime("%H:%M"),
                y=intraday[column],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            )
        )
    fig.update_yaxes(title="功率（MW）")
    fig.update_xaxes(title="时刻")
    st.plotly_chart(figure_layout(fig, 420), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"**事前判断**：批发采购分摊仓位 ÷ 预测电量 = **{fmt_pct(row['contract_coverage_ratio'])}**。\n\n"
            "这是日前申报前后可用于仓位判断的客户级口径。"
        )
    with c2:
        st.markdown(
            f"**事后复盘**：同一分摊仓位 ÷ 实际用电量 = **{fmt_pct(row['actual_coverage_ratio'])}**。\n\n"
            "它用于解释实际负荷变化如何改变风险暴露，不等于正式结算公式。"
        )
    if calendar_row is not None:
        st.caption(
            f"状态：{STATE_LABELS.get(str(calendar_row['day_state']), str(calendar_row['day_state']))}｜"
            f"事件：{calendar_row.get('event_type') or '无'}｜"
            f"事件获知时间：{calendar_row.get('event_known_time') or '无'}"
        )
    st.info("核心案例默认日期为2026-03-17；若切换到其他日期，页面仍按同一客户级分摊口径计算。")


def render_event_recap(data: dict[str, pd.DataFrame], selected_date: Any, customer_id: str) -> None:
    st.header("事件复盘：把数据异常转化为运营动作")
    st.caption("事件获知时间不等于实际风险发现时间；下方时间线用于模拟运营复盘，不代表正式结算时点。")
    row = metric_row(data, customer_id, selected_date)
    calendar_row = get_calendar_row(data, customer_id, selected_date)
    customer_data = customer_intraday(data, customer_id)
    day = customer_data.loc[customer_data["date"] == selected_date]
    forecast_time = day["forecast_issue_time"].dropna().iloc[0] if not day.empty else None
    submitted_time = day["submitted_time"].dropna().iloc[0] if not day.empty else None
    event_time = None if calendar_row is None else calendar_row.get("event_known_time")
    event_type = None if calendar_row is None else calendar_row.get("event_type")
    cause = None if calendar_row is None else calendar_row.get("触发条件")

    if calendar_row is None or pd.isna(event_type):
        st.success("所选日期没有模拟异常事件，建议按常态流程跟踪负荷与覆盖。")
        st.info(risk_explanation(row, calendar_row))
        return

    timeline = pd.DataFrame(
        [
            [forecast_time, "预测签发", "只使用当时可获得的天气、客户状态和历史负荷信息。"],
            [submitted_time, "日前申报", "形成日前申报与批发采购分摊判断。"],
            [event_time, "事件获知", f"模拟事件：{event_type}；原因台账：{cause}。"],
            [event_time, "风险发现", f"实际电量 {fmt_mwh(row['daily_energy_mwh'])}，预测电量 {fmt_mwh(row['forecast_energy_mwh'])}。"],
            [event_time, "运营动作", "联系客户、更新剩余时段预测、重算仓位并跟踪恢复。"],
        ],
        columns=["时间", "节点", "可用信息与运营含义"],
    )
    st.dataframe(timeline, width="stretch", hide_index=True)
    st.subheader("模拟运营建议")
    for action in (
        "联系客户确认设备和生产状态，核实事件是否影响剩余时段。",
        "根据已知信息更新剩余时段负荷判断，不回填事后信息到日前预测。",
        "重新计算客户级批发采购分摊覆盖与偏差风险。",
        "跟踪恢复日负荷曲线，并将本次事件纳入客户沟通记录。",
    ):
        st.markdown(f"- {action}")
    st.warning("以上为运营辅助建议，不是AI自动决策，也不替代正式交易或结算流程。")


def render_credibility(data: dict[str, pd.DataFrame]) -> None:
    st.header("数据可用于业务分析演示，但不构成真实交易依据")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("客户数量", 3)
    c2.metric("模拟时间范围", "2026全年")
    c3.metric("时间粒度", "15分钟")
    c4.metric("客户记录数", "105,120")

    st.subheader("单位与时间口径")
    st.markdown(
        "- 功率：MW；电量：MWh；电价：元/MWh；金额：元。\n"
        "- 15分钟电量 = 功率 × 0.25。\n"
        "- 天气D-1 12:00可得；预测D-1 15:00签发；申报D-1 17:00；突发事件按实际发生时间记录。"
    )

    st.subheader("模拟边界")
    st.markdown(
        "- 客户名称、负荷、价格、合同和风险金额均为模拟。\n"
        "- `wholesale_contract_*`为客户级批发采购仓位分摊曲线，不是公司整体仓位。\n"
        "- 日前与实时价格为`SIM_ASSUMPTION`，只用于验证负荷—价格—偏差风险链路。\n"
        "- 不代表安徽官方交易规则、正式结算公式或真实交易收益。"
    )

    st.subheader("数据质量检查结果")
    qa = data["qa"].copy()
    first = qa.loc[qa["检查类别"].notna(), ["检查类别", "检查项", "异常数/结果", "结论"]].dropna(how="all").copy()
    # The source QA sheet intentionally mixes counts and text in one column;
    # cast only the display copy so Streamlit's Arrow serialization is stable.
    for column in first.columns:
        first[column] = first[column].fillna("").astype(str)
    if not first.empty:
        st.dataframe(first, width="stretch", hide_index=True)
    st.caption("质量检查来源：年度模拟工作簿中的‘质量检查’工作表；本Demo只读展示，不回写源文件。")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="⚡", layout="wide")
    st.title(APP_TITLE)
    st.caption("客户负荷、合同覆盖与偏差风险监测｜公开展示Demo｜模拟数据，不构成交易或结算依据")

    try:
        data = load_data(str(DATA_PATH))
    except Exception as exc:  # pragma: no cover - displayed in the app
        st.error(f"数据加载失败：{exc}")
        st.stop()

    dates = sorted(data["state_calendar"]["date"].dropna().unique())
    default_date = pd.Timestamp("2026-03-17").date()
    if default_date not in dates:
        default_date = dates[0]
    with st.sidebar:
        st.header("分析控制")
        selected_date = st.date_input(
            "模拟分析日",
            value=default_date,
            min_value=min(dates),
            max_value=max(dates),
        )
        customer_filter_label = st.selectbox(
            "客户筛选",
            ["全部客户"] + [CUSTOMER_LABELS[cid] for cid in CUSTOMERS],
        )
        customer_filter = (
            "全部客户"
            if customer_filter_label == "全部客户"
            else customer_filter_label.split("｜", 1)[0]
        )
        profile_customer_label = st.selectbox(
            "画像/分析客户",
            [CUSTOMER_LABELS[cid] for cid in CUSTOMERS],
            index=0 if customer_filter in {"全部客户", "MFG01"} else CUSTOMERS.index(customer_filter),
        )
        profile_customer = profile_customer_label.split("｜", 1)[0]
        st.divider()
        st.caption(f"只读数据源：{DATA_PATH.name}")
        st.caption("风险等级为Demo筛选标签，不是正式市场规则。")

    pages = st.tabs(
        [
            "1｜风险总览",
            "2｜客户画像",
            "3｜负荷分析",
            "4｜覆盖与风险",
            "5｜事件复盘",
            "6｜数据可信度",
        ]
    )
    with pages[0]:
        render_dashboard(data, selected_date, customer_filter)
    with pages[1]:
        render_profile(data, profile_customer)
    with pages[2]:
        render_load_analysis(data, profile_customer, selected_date)
    with pages[3]:
        render_coverage(data, selected_date)
    with pages[4]:
        render_event_recap(data, selected_date, profile_customer)
    with pages[5]:
        render_credibility(data)


if __name__ == "__main__":
    main()
