"""Soda Core integration for traditional data quality checks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from guardian.models import QualityCheckResult

logger = logging.getLogger(__name__)


class SodaQualityRunner:
    """Runs Soda Core checks against pandas DataFrames."""

    def __init__(self, config: dict) -> None:
        self._config = config

    def run_checks(
        self, dataframe: pd.DataFrame, dataset_name: str
    ) -> list[QualityCheckResult]:
        """Execute Soda Core checks on a pandas DataFrame."""
        try:
            from soda.scan import Scan
        except ImportError:
            logger.warning("soda-core not installed, skipping Soda checks")
            return self._run_fallback_checks(dataframe, dataset_name)

        scan = Scan()
        scan.set_scan_definition_name(f"guardian_{dataset_name}")
        scan.set_data_source_name("guardian_pandas")

        # Add pandas dataframe as data source
        scan.add_pandas_dataframe(
            dataset_name=dataset_name, pandas_df=dataframe
        )

        # Generate and add check YAML
        check_yaml = self._generate_checks(dataframe, dataset_name)
        scan.add_sodacl_yaml_str(check_yaml)

        scan.execute()
        results: list[QualityCheckResult] = []

        for check in scan.get_checks_fail() + scan.get_checks_warn():
            results.append(
                QualityCheckResult(
                    check_name=check.name,
                    dataset=dataset_name,
                    passed=False,
                    details={
                        "outcome": check.outcome.value if hasattr(check.outcome, "value") else str(check.outcome),
                        "diagnostics": str(check.get_log_diagnostic_dict()),
                    },
                )
            )

        for check in scan.get_checks_pass():
            results.append(
                QualityCheckResult(
                    check_name=check.name,
                    dataset=dataset_name,
                    passed=True,
                    details={},
                )
            )

        return results

    def _run_fallback_checks(
        self, dataframe: pd.DataFrame, dataset_name: str
    ) -> list[QualityCheckResult]:
        """Fallback checks when Soda Core is not available."""
        results: list[QualityCheckResult] = []

        # Row count check
        results.append(
            QualityCheckResult(
                check_name="row_count",
                dataset=dataset_name,
                passed=len(dataframe) > 0,
                metric_value=float(len(dataframe)),
            )
        )

        # Null checks for each column
        for col in dataframe.columns:
            null_count = int(dataframe[col].isna().sum())
            results.append(
                QualityCheckResult(
                    check_name=f"missing_count({col})",
                    dataset=dataset_name,
                    passed=null_count == 0,
                    metric_value=float(null_count),
                    details={"null_count": null_count, "total_rows": len(dataframe)},
                )
            )

        # Duplicate check on first column (assumed to be ID)
        if len(dataframe.columns) > 0:
            first_col = dataframe.columns[0]
            dup_count = int(dataframe[first_col].duplicated().sum())
            results.append(
                QualityCheckResult(
                    check_name=f"duplicate_count({first_col})",
                    dataset=dataset_name,
                    passed=dup_count == 0,
                    metric_value=float(dup_count),
                )
            )

        return results

    @staticmethod
    def _generate_checks(dataframe: pd.DataFrame, dataset_name: str) -> str:
        """Auto-generate Soda check YAML for a DataFrame's schema."""
        lines = [f"checks for {dataset_name}:"]
        lines.append("  - row_count > 0")
        for col in dataframe.columns:
            lines.append(f"  - missing_count({col}) = 0")
        if len(dataframe.columns) > 0:
            lines.append(f"  - duplicate_count({dataframe.columns[0]}) = 0")
        return "\n".join(lines)
