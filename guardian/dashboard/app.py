"""Streamlit dashboard for LLM Data Quality Guardian."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from guardian.storage.metrics_store import MetricsStore


def get_store() -> MetricsStore:
    return MetricsStore(db_path=".data/metrics.db")


def render_overview(store: MetricsStore) -> None:
    """Summary cards and recent alerts."""
    run = store.get_latest_run()
    if not run:
        st.info("No pipeline runs yet. Run `guardian run` to get started.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents Processed", run.documents_processed)
    with col2:
        drifted = sum(1 for d in run.drift_results if d.is_drifted)
        st.metric("Drift Alerts", drifted)
    with col3:
        st.metric("Hallucination Risks", len(run.hallucination_risks))
    with col4:
        failed = sum(1 for q in run.quality_results if not q.passed)
        st.metric("Quality Failures", failed)

    # Health score
    total_checks = max(
        len(run.drift_results) + len(run.quality_results) + len(run.hallucination_risks), 1
    )
    issues = (
        drifted
        + failed
        + sum(1 for h in run.hallucination_risks if h.severity.value in ("WARNING", "CRITICAL"))
    )
    health = max(0, 100 - int((issues / total_checks) * 100))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health,
            title={"text": "Data Health Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "#ff4444"},
                    {"range": [40, 70], "color": "#ffaa00"},
                    {"range": [70, 100], "color": "#44bb44"},
                ],
            },
        )
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Recent alerts
    st.subheader("Recent Alerts")
    alerts_df = store.get_alerts(days=7)
    if alerts_df.empty:
        st.success("No alerts in the last 7 days.")
    else:
        st.dataframe(
            alerts_df[["severity", "title", "message", "created_at"]],
            use_container_width=True,
        )


def render_drift_dashboard(store: MetricsStore) -> None:
    """Drift detection metrics and trends."""
    df = store.get_drift_history(days=30)
    if df.empty:
        st.info("No drift data available yet.")
        return

    # Latest results
    st.subheader("Latest Drift Results")
    latest_run = df["run_id"].iloc[-1]
    latest = df[df["run_id"] == latest_run]
    for _, row in latest.iterrows():
        status = "DRIFTED" if row["is_drifted"] else "OK"
        color = "red" if row["is_drifted"] else "green"
        st.markdown(
            f"**{row['test_name']}**: :{color}[{status}] "
            f"(statistic={row['statistic']:.4f}, threshold={row['threshold']})"
        )

    # Trend chart
    st.subheader("Drift Trends Over Time")
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    fig = px.line(
        df,
        x="measured_at",
        y="statistic",
        color="test_name",
        title="Drift Statistics Over Time",
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Details
    if st.checkbox("Show raw drift data"):
        st.dataframe(df, use_container_width=True)


def render_hallucination_dashboard(store: MetricsStore) -> None:
    """Hallucination risk visualization."""
    df = store.get_hallucination_risks(days=30)
    if df.empty:
        st.info("No hallucination risks detected.")
        return

    st.subheader("Risk Distribution by Type")
    type_counts = df["risk_type"].value_counts().reset_index()
    type_counts.columns = ["risk_type", "count"]
    fig = px.bar(
        type_counts,
        x="risk_type",
        y="count",
        color="risk_type",
        title="Hallucination Risks by Type",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Severity Breakdown")
    sev_counts = df["severity"].value_counts().reset_index()
    sev_counts.columns = ["severity", "count"]
    colors = {"CRITICAL": "#ff4444", "WARNING": "#ffaa00", "INFO": "#4488ff", "DEBUG": "#888888"}
    fig2 = px.pie(
        sev_counts,
        values="count",
        names="severity",
        color="severity",
        color_discrete_map=colors,
        title="Risks by Severity",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detected Risks")
    display_df = df[["risk_type", "severity", "description", "confidence", "detected_at"]].copy()
    st.dataframe(display_df, use_container_width=True)


def render_quality_dashboard(store: MetricsStore) -> None:
    """Data quality check results."""
    df = store.get_quality_history(days=30)
    if df.empty:
        st.info("No quality check data available.")
        return

    st.subheader("Latest Quality Results")
    latest_run = df["run_id"].iloc[-1]
    latest = df[df["run_id"] == latest_run]

    pass_count = int(latest["passed"].sum())
    fail_count = len(latest) - pass_count
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Passed", pass_count)
    with col2:
        st.metric("Failed", fail_count)

    for _, row in latest.iterrows():
        icon = "white_check_mark" if row["passed"] else "x"
        st.markdown(f":{icon}: **{row['check_name']}** ({row['dataset']})")

    # Trend
    st.subheader("Quality Trend")
    df["checked_at"] = pd.to_datetime(df["checked_at"])
    trend = df.groupby(df["checked_at"].dt.date).agg(
        total=("passed", "count"),
        passed=("passed", "sum"),
    ).reset_index()
    trend["pass_rate"] = trend["passed"] / trend["total"] * 100
    fig = px.line(
        trend,
        x="checked_at",
        y="pass_rate",
        title="Quality Check Pass Rate (%)",
        markers=True,
    )
    fig.update_yaxes(range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)


def render_rag_dashboard(store: MetricsStore) -> None:
    """RAG pipeline quality metrics."""
    df = store.get_rag_history(days=30)
    if df.empty:
        st.info("No RAG quality data available.")
        return

    st.subheader("RAG Quality Metrics")
    latest_run = df["run_id"].iloc[-1]
    latest = df[df["run_id"] == latest_run]

    for _, row in latest.iterrows():
        details = json.loads(row["details_json"]) if row["details_json"] else {}
        st.metric(
            row["metric_name"].replace("_", " ").title(),
            f"{row['score']:.3f}",
        )
        if details and st.checkbox(f"Details: {row['metric_name']}", key=row["metric_name"]):
            st.json(details)

    # Trend
    st.subheader("RAG Quality Trends")
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    fig = px.line(
        df,
        x="measured_at",
        y="score",
        color="metric_name",
        title="RAG Metrics Over Time",
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="LLM Data Quality Guardian",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("LLM Data Quality Guardian")
    st.caption("Monitoring data quality for LLM/RAG systems")

    store = get_store()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Drift Detection", "Hallucination Risks", "Data Quality", "RAG Quality"]
    )

    with tab1:
        render_overview(store)
    with tab2:
        render_drift_dashboard(store)
    with tab3:
        render_hallucination_dashboard(store)
    with tab4:
        render_quality_dashboard(store)
    with tab5:
        render_rag_dashboard(store)


def run() -> None:
    """Entry point for `guardian dashboard` command."""
    main()


if __name__ == "__main__":
    main()
