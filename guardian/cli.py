"""Command-line interface for LLM Data Quality Guardian."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from rich.console import Console
from rich.table import Table

from guardian.pipeline import GuardianPipeline


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a full pipeline run."""
    console = Console()
    console.print("[bold]LLM Data Quality Guardian[/bold] — Pipeline Run\n")

    pipeline = GuardianPipeline(config_path=args.config)
    result = pipeline.run()

    # Summary table
    table = Table(title="Pipeline Run Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Run ID", result.run_id)
    table.add_row("Documents Processed", str(result.documents_processed))
    table.add_row("Drift Tests", str(len(result.drift_results)))
    table.add_row(
        "Drift Detected",
        str(sum(1 for d in result.drift_results if d.is_drifted)),
    )
    table.add_row("Hallucination Risks", str(len(result.hallucination_risks)))
    table.add_row("Quality Checks", str(len(result.quality_results)))
    table.add_row(
        "Quality Failures",
        str(sum(1 for q in result.quality_results if not q.passed)),
    )
    table.add_row("RAG Metrics", str(len(result.rag_results)))
    table.add_row("Alerts Generated", str(len(result.alerts)))

    if result.started_at and result.completed_at:
        duration = (result.completed_at - result.started_at).total_seconds()
        table.add_row("Duration", f"{duration:.1f}s")

    console.print(table)


def cmd_report(args: argparse.Namespace) -> None:
    """Print a text report of the latest pipeline run."""
    from guardian.storage.metrics_store import MetricsStore

    console = Console()
    store = MetricsStore(db_path=args.db_path)
    run = store.get_latest_run()

    if not run:
        console.print("[yellow]No pipeline runs found.[/yellow]")
        return

    console.print(f"\n[bold]Latest Run:[/bold] {run.run_id}")
    console.print(f"  Started:    {run.started_at}")
    console.print(f"  Completed:  {run.completed_at}")
    console.print(f"  Documents:  {run.documents_processed}\n")

    if run.drift_results:
        console.print("[bold underline]Drift Detection[/bold underline]")
        for dr in run.drift_results:
            status = "[red]DRIFTED[/red]" if dr.is_drifted else "[green]OK[/green]"
            console.print(
                f"  {dr.test_name:20s} stat={dr.statistic:.4f}  "
                f"threshold={dr.threshold}  {status}"
            )
        console.print()

    if run.hallucination_risks:
        console.print("[bold underline]Hallucination Risks[/bold underline]")
        for hr in run.hallucination_risks:
            color = "red" if hr.severity == "CRITICAL" else "yellow"
            console.print(
                f"  [{color}]{hr.severity}[/{color}] {hr.risk_type}: "
                f"{hr.description[:100]}"
            )
        console.print()

    if run.quality_results:
        console.print("[bold underline]Quality Checks[/bold underline]")
        for qc in run.quality_results:
            status = "[green]PASS[/green]" if qc.passed else "[red]FAIL[/red]"
            console.print(f"  {qc.check_name:30s} {status}")
        console.print()

    if run.alerts:
        console.print(f"[bold underline]Alerts ({len(run.alerts)})[/bold underline]")
        for al in run.alerts:
            console.print(f"  [{al.severity.value}] {al.title}")


def cmd_dashboard(_args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    import guardian.dashboard.app as app_module

    app_path = app_module.__file__
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Data Quality Guardian — monitor data quality for LLM/RAG systems"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command")

    # guardian run
    run_p = subparsers.add_parser("run", help="Execute a pipeline run")
    run_p.add_argument(
        "--config", default="config/default.yaml", help="Config file path"
    )
    run_p.set_defaults(func=cmd_run)

    # guardian report
    rep_p = subparsers.add_parser("report", help="Show latest run report")
    rep_p.add_argument(
        "--db-path", default=".data/metrics.db", help="Metrics database path"
    )
    rep_p.set_defaults(func=cmd_report)

    # guardian dashboard
    dash_p = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_p.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
