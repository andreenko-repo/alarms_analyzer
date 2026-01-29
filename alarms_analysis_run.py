import sys
import argparse
import logging
import warnings
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from fpdf import FPDF
import tempfile
import os

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTS AND BENCHMARKS
# ==============================================================================

# EEMUA 191
EEMUA_BENCHMARKS = {
    "Very Good": {"max": 1, "description": "Very likely to be acceptable"},
    "Acceptable": {"max": 2, "description": "Manageable"},
    "Too Many": {"max": 5, "description": "Requires improvement"},
    "Extremely Dangerous": {
        "max": float("inf"),
        "description": "Overloaded - immediate action required",
    },
}

# ISA-18.2 Flood threshold (alarms per 10 minutes)
ISA_FLOOD_THRESHOLD = 10

# ISA-18.2 Priority distribution recommendations
# TODO - Priority distrinution is important; synthetic alarms to be reviewed
ISA_PRIORITY_DISTRIBUTION = {
    "HIGH": {"min": 0, "max": 5},  # Critical/Emergency: <5%
    "MEDIUM": {"min": 10, "max": 15},  # High priority: 10-15%
    "LOW": {"min": 80, "max": 90},  # Low priority: 80%+
}


# ==============================================================================
# ALARM ANALYZER
# ==============================================================================
class AlarmAnalyzer:
    def __init__(
        self,
        file_path: str,
        min_support: float = 0.01,
        window_size: str = "5min",
        chatter_cutoff: float = 0.05,
        stale_threshold_hours: float = 24.0,
        repeatability_window_minutes: int = 10,
    ):
        self.file_path = file_path
        self.min_support = min_support
        self.window_size = window_size
        self.chatter_cutoff = chatter_cutoff
        self.stale_threshold_hours = stale_threshold_hours
        self.repeatability_window_minutes = repeatability_window_minutes

        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.df_alarms: Optional[pd.DataFrame] = None  # Only alarm activations
        self.df_recoveries: Optional[pd.DataFrame] = None  # Only recoveries

        # Metrics
        self.total_alarms = 0
        self.total_activations = 0
        self.total_recoveries = 0
        self.duration_hours = 0.0
        self.start_time: Optional[pd.Timestamp] = None
        self.end_time: Optional[pd.Timestamp] = None

    def load_data(self) -> None:
        # Expected CSV format:
        #    - Column 0: Index
        #    - Column 1: Timestamp
        #    - Column 2: Tag
        #    - Column 3: Description
        #    - Column 4: Condition (e.g., HIGH, LOW, FAULT, with RECOVERED suffix for returns)

        logger.info(f"Loading data from {self.file_path}...")

        # Check if file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        try:
            # Load CSV file
            self.df = pd.read_csv(
                self.file_path, index_col=0, parse_dates=["timestamp"]
            )

            # Check if the file columns match our expectations
            required_columns = ["timestamp", "tag", "description", "condition"]
            missing_columns = [
                col for col in required_columns if col not in self.df.columns
            ]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

        except (pd.errors.ParserError, ValueError, KeyError) as e:
            logger.warning(
                f"Standard parsing failed: {e}. Attempting flexible parsing..."
            )
            # Best Effort
            self.df = pd.read_csv(self.file_path)

            if len(self.df.columns) >= 4:
                # Assume first column is index, rest are timestamp, tag, description, condition
                if (
                    self.df.columns[0] in ["Unnamed: 0", ""]
                    or self.df.columns[0].isdigit()
                ):
                    self.df = self.df.iloc[:, 1:]

                self.df.columns = ["timestamp", "tag", "description", "condition"][
                    : len(self.df.columns)
                ]
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            else:
                raise ValueError(
                    f"CSV must have at least 4 columns (timestamp, tag, description, condition). "
                    f"Found {len(self.df.columns)} columns."
                )

        # Data type, use category type for optimization when big files used
        self.df["tag"] = self.df["tag"].astype("category")
        self.df["condition"] = (
            self.df["condition"].astype(str).str.strip().astype("category")
        )
        self.df["description"] = self.df["description"].astype(str)

        # Sort by timestamp
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # Find alarm activations and recoveries
        self.df["is_recovery"] = self.df["condition"].str.contains(
            "RECOVERED", case=False, na=False
        )
        self.df["is_activation"] = ~self.df["is_recovery"]

        # Create dataframes for activations and recoveries
        self.df_alarms = self.df[self.df["is_activation"]].copy()
        self.df_recoveries = self.df[self.df["is_recovery"]].copy()

        # Create unique alarm identifier (tag + base condition)
        self.df["base_condition"] = self.df["condition"].str.replace(
            " RECOVERED", "", case=False, regex=False
        )
        self.df["alarm_id"] = (
            self.df["tag"].astype(str) + "_" + self.df["base_condition"].astype(str)
        )
        self.df_alarms["alarm_id"] = (
            self.df_alarms["tag"].astype(str)
            + "_"
            + self.df_alarms["condition"].astype(str)
        )

        # Calculate metrics
        self.total_alarms = len(self.df)
        self.total_activations = len(self.df_alarms)
        self.total_recoveries = len(self.df_recoveries)
        self.start_time = self.df["timestamp"].min()
        self.end_time = self.df["timestamp"].max()

        duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.duration_hours = max(
            duration_seconds / 3600, 0.01
        )  # Minimum 0.01 to avoid division by zero

        logger.info(
            f"Loaded {self.total_alarms} records "
            f"({self.total_activations} activations, {self.total_recoveries} recoveries). "
            f"Duration: {self.duration_hours:.2f} hours."
        )

    def get_most_frequent(self, top_n: int = 10) -> pd.Series:
        """Return the most frequently occurring alarm tags."""
        return self.df_alarms["tag"].value_counts().head(top_n)

    # -------------------------------------------------------------------------
    # EEMUA 191 & ISA-18.2 REQUIREMENTS ANALYSIS
    # -------------------------------------------------------------------------

    def analyze_health(self) -> Dict[str, Any]:
        # Calculate metrics per EEMUA 191 and ISA-18.2.
        #
        # Metrics:
        #    - Average alarm rate per 10 minutes
        #    - EEMUA 191 system status classification
        #    - ISA-18.2 flood interval analysis
        #    - Peak-to-average ratio

        logger.info("Calculating EEMUA 191 and ISA-18.2 health metrics...")

        # Average alarm rate per 10 minutes (activations only)
        avg_rate_per_10min = (self.total_activations / self.duration_hours) / 6

        # EEMUA 191 classification
        if avg_rate_per_10min < EEMUA_BENCHMARKS["Very Good"]["max"]:
            status = "Very Good"
            status_desc = EEMUA_BENCHMARKS["Very Good"]["description"]
        elif avg_rate_per_10min < EEMUA_BENCHMARKS["Acceptable"]["max"]:
            status = "Acceptable"
            status_desc = EEMUA_BENCHMARKS["Acceptable"]["description"]
        elif avg_rate_per_10min < EEMUA_BENCHMARKS["Too Many"]["max"]:
            status = "Too Many"
            status_desc = EEMUA_BENCHMARKS["Too Many"]["description"]
        else:
            status = "Extremely Dangerous"
            status_desc = EEMUA_BENCHMARKS["Extremely Dangerous"]["description"]

        # ISA-18.2 Flood Analysis (>10 alarms per 10 minutes)
        floods = self.df_alarms.set_index("timestamp").resample("10min")["tag"].count()
        flood_intervals = floods[floods > ISA_FLOOD_THRESHOLD]
        flood_count = len(flood_intervals)
        total_intervals = len(floods)
        flood_pct = (flood_count / total_intervals) * 100 if total_intervals > 0 else 0

        # Peak-to-Average Ratio
        peak_rate = floods.max() if not floods.empty else 0
        avg_rate = floods.mean() if not floods.empty else 0
        peak_to_avg_ratio = peak_rate / avg_rate if avg_rate > 0 else 0

        return {
            "avg_10min": avg_rate_per_10min,
            "status": status,
            "status_description": status_desc,
            "flood_count": flood_count,
            "flood_pct": flood_pct,
            "total_intervals": total_intervals,
            "peak_rate": peak_rate,
            "avg_rate": avg_rate,
            "peak_to_avg_ratio": peak_to_avg_ratio,
        }

    # -------------------------------------------------------------------------
    # CHATTER ANALYSIS (Kondaveeti et al.)
    # -------------------------------------------------------------------------

    def analyze_chatter(self) -> pd.DataFrame:
        # Analyze alarm chatter using the Kondaveeti method (the most efficient method for detecting chattering alarms).
        #
        # Per ISA-18.2, chattering alarms should be addressed through:
        #    - Deadband adjustment
        #    - Timer/delay implementation
        #    - Signal filtering

        logger.info("Analyzing alarm chatter (Kondaveeti method)...")

        df_sorted = self.df_alarms.sort_values(by=["tag", "timestamp"]).copy()
        df_sorted["prev_time"] = df_sorted.groupby("tag")["timestamp"].shift(1)
        df_sorted["interval_seconds"] = (
            df_sorted["timestamp"] - df_sorted["prev_time"]
        ).dt.total_seconds()
        df_sorted["interval_seconds"] = df_sorted["interval_seconds"].clip(lower=1)

        clean_runs = df_sorted.dropna(subset=["interval_seconds"])

        def calc_chatter_index(group: pd.DataFrame) -> float:
            total = len(group)
            if total == 0:
                return 0.0
            intervals = group["interval_seconds"].value_counts()
            psi = sum(
                (count / total) * (1 / interval)
                for interval, count in intervals.items()
            )
            return psi

        results = (
            clean_runs.groupby("tag")
            .apply(calc_chatter_index, include_groups=False)
            .reset_index(name="chatter_index")
        )

        counts = self.df_alarms["tag"].value_counts().reset_index()
        counts.columns = ["tag", "total_activations"]

        results = results.merge(counts, on="tag", how="left")
        results["status"] = results["chatter_index"].apply(
            lambda x: "CRITICAL" if x >= self.chatter_cutoff else "Normal"
        )

        return results.sort_values(by="chatter_index", ascending=False)

    # -------------------------------------------------------------------------
    # STALE ALARMS (ISA-18.2)
    # -------------------------------------------------------------------------

    def analyze_stale_alarms(self) -> pd.DataFrame:
        # Identify stale alarms per ISA-18.2 guidelines.
        #
        # A stale alarm is one that remains active for an extended period without
        # being addressed. Per ISA-18.2, stale alarms:
        #    - Contribute to alarm overload
        #    - May indicate process issues requiring attention
        #    - Should be reviewed for proper alarm configuration

        logger.info(
            f"Analyzing stale alarms (threshold: {self.stale_threshold_hours} hours)..."
        )

        stale_records = []

        # Process each unique alarm
        for alarm_id in self.df["alarm_id"].unique():
            alarm_data = self.df[self.df["alarm_id"] == alarm_id].copy()

            # Get activation and recovery events
            activations = alarm_data[alarm_data["is_activation"]].sort_values(
                "timestamp"
            )
            recoveries = alarm_data[alarm_data["is_recovery"]].sort_values("timestamp")

            if activations.empty:
                continue

            # Match activations with their subsequent recoveries
            for _, activation in activations.iterrows():
                activation_time = activation["timestamp"]

                # Find the next recovery after this activation
                subsequent_recoveries = recoveries[
                    recoveries["timestamp"] > activation_time
                ]

                if subsequent_recoveries.empty:
                    # No recovery found - alarm may still be active
                    active_duration = (
                        self.end_time - activation_time
                    ).total_seconds() / 3600
                    recovery_time = None
                    still_active = True
                else:
                    recovery_time = subsequent_recoveries.iloc[0]["timestamp"]
                    active_duration = (
                        recovery_time - activation_time
                    ).total_seconds() / 3600
                    still_active = False

                if active_duration >= self.stale_threshold_hours:
                    stale_records.append(
                        {
                            "tag": activation["tag"],
                            "description": activation["description"],
                            "condition": activation["condition"],
                            "activation_time": activation_time,
                            "recovery_time": recovery_time,
                            "active_duration_hours": round(active_duration, 2),
                            "still_active": still_active,
                        }
                    )

        stale_df = pd.DataFrame(stale_records)

        if not stale_df.empty:
            stale_df = stale_df.sort_values("active_duration_hours", ascending=False)

        logger.info(f"Found {len(stale_df)} stale alarm instances.")
        return stale_df

    # -------------------------------------------------------------------------
    # STANDING ALARMS
    # -------------------------------------------------------------------------

    def analyze_standing_alarms(self) -> pd.DataFrame:
        # Identify standing alarms that never clear during the analysis period.
        #
        # Per EEMUA 191, standing alarms should be addressed to maintain
        # operator situational awareness.

        logger.info("Analyzing standing alarms...")

        standing_records = []

        for tag in self.df["tag"].unique():
            tag_data = self.df[self.df["tag"] == tag]

            # Get conditions for this tag
            conditions = tag_data["base_condition"].unique()

            for condition in conditions:
                condition_data = tag_data[tag_data["base_condition"] == condition]

                activations = condition_data[condition_data["is_activation"]]
                recoveries = condition_data[condition_data["is_recovery"]]

                activation_count = len(activations)
                recovery_count = len(recoveries)

                # If activations exceed recoveries, alarm may be standing
                if activation_count > recovery_count:
                    uncleared = activation_count - recovery_count

                    # Get the first uncleared activation
                    if not activations.empty:
                        first_activation = activations.iloc[0]
                        last_activation = activations.iloc[-1]

                        standing_records.append(
                            {
                                "tag": tag,
                                "description": first_activation["description"],
                                "condition": condition,
                                "total_activations": activation_count,
                                "total_recoveries": recovery_count,
                                "uncleared_count": uncleared,
                                "first_activation": first_activation["timestamp"],
                                "last_activation": last_activation["timestamp"],
                            }
                        )

        standing_df = pd.DataFrame(standing_records)

        if not standing_df.empty:
            standing_df = standing_df.sort_values("uncleared_count", ascending=False)

        logger.info(f"Found {len(standing_df)} potentially standing alarms.")
        return standing_df

    # -------------------------------------------------------------------------
    # BAD ACTOR (ISA-18.2 Section 11)
    # -------------------------------------------------------------------------

    def analyze_bad_actors(
        self, contribution_threshold: float = 0.80
    ) -> Dict[str, Any]:
        # SA-18.2 recommends regular review of the 'top 10' most
        # frequent alarms to identify improvement opportunities

        logger.info(
            f"Analyzing bad actors (Pareto analysis at {contribution_threshold*100:.0f}%)..."
        )

        counts = self.df_alarms["tag"].value_counts()
        total = counts.sum()

        cumulative = counts.cumsum() / total

        # Find alarms contributing to threshold% of total
        bad_actors = counts[cumulative <= contribution_threshold]

        # Calculate percentages
        bad_actor_pct = len(bad_actors) / len(counts) * 100 if len(counts) > 0 else 0
        contribution_pct = bad_actors.sum() / total * 100 if total > 0 else 0

        # Detailed bad actor table
        bad_actor_df = pd.DataFrame(
            {
                "tag": bad_actors.index,
                "activations": bad_actors.values,
                "percentage": (bad_actors.values / total * 100).round(2),
                "cumulative_pct": (cumulative[bad_actors.index].values * 100).round(2),
            }
        )

        return {
            "bad_actor_count": len(bad_actors),
            "total_alarm_count": len(counts),
            "bad_actor_pct": bad_actor_pct,
            "contribution_pct": contribution_pct,
            "threshold": contribution_threshold * 100,
            "bad_actor_df": bad_actor_df,
            "all_counts": counts,
        }

    # -------------------------------------------------------------------------
    # REPEATABILITY INDEX
    # -------------------------------------------------------------------------

    def analyze_repeatability(self) -> pd.DataFrame:
        # Repeatability measures how often the same alarm activates multiple times
        # within a short window. High repeatability may indicate:
        #     - Insufficient deadband
        #     - Process instability
        #     - Need for alarm delay/timer

        logger.info(
            f"Analyzing repeatability (window: {self.repeatability_window_minutes} minutes)..."
        )

        window_td = timedelta(minutes=self.repeatability_window_minutes)

        repeatability_records = []

        for tag in self.df_alarms["tag"].unique():
            tag_data = self.df_alarms[self.df_alarms["tag"] == tag].sort_values(
                "timestamp"
            )

            if len(tag_data) < 2:
                continue

            # Count repeated activations within window
            repeat_count = 0
            timestamps = tag_data["timestamp"].values

            for i in range(1, len(timestamps)):
                time_diff = pd.Timestamp(timestamps[i]) - pd.Timestamp(
                    timestamps[i - 1]
                )
                if time_diff <= window_td:
                    repeat_count += 1

            total_activations = len(tag_data)
            repeatability_index = (
                repeat_count / (total_activations - 1) if total_activations > 1 else 0
            )

            repeatability_records.append(
                {
                    "tag": tag,
                    "description": tag_data.iloc[0]["description"],
                    "total_activations": total_activations,
                    "repeat_count": repeat_count,
                    "repeatability_index": round(repeatability_index, 4),
                }
            )

        repeat_df = pd.DataFrame(repeatability_records)

        if not repeat_df.empty:
            repeat_df = repeat_df.sort_values("repeatability_index", ascending=False)

        return repeat_df

    # -------------------------------------------------------------------------
    # ALARM FLOOD DETECTION (ISA-18.2)
    # -------------------------------------------------------------------------

    def detect_avalanches(
        self, window: str = "1min", threshold: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # An alarm flod (or avakanche) occurs when many alarms activate in rapid
        # succession, often due to a single root cause. ISA-18.2 recommends:
        #     - Identifying root cause alarms
        #     - Implementing state-based alarm suppression
        #     - Using alarm shelving during known conditions

        logger.info(
            f"Detecting alarm avalanches (window: {window}, threshold: {threshold})..."
        )

        # Resample to count alarms per window
        resampled = (
            self.df_alarms.set_index("timestamp").resample(window)["tag"].count()
        )
        avalanche_windows = resampled[resampled >= threshold]

        if avalanche_windows.empty:
            logger.info("No avalanches detected.")
            return pd.DataFrame(), pd.DataFrame()

        avalanche_details = []
        window_td = pd.Timedelta(window)

        for timestamp in avalanche_windows.index:
            window_end = timestamp + window_td

            # Get alarms in this window
            window_alarms = self.df_alarms[
                (self.df_alarms["timestamp"] >= timestamp)
                & (self.df_alarms["timestamp"] < window_end)
            ]

            unique_tags = window_alarms["tag"].unique()
            tag_counts = window_alarms["tag"].value_counts()

            # Identify potential root cause (first alarm in sequence)
            first_alarm = window_alarms.iloc[0] if not window_alarms.empty else None

            avalanche_details.append(
                {
                    "start_time": timestamp,
                    "end_time": window_end,
                    "total_alarms": avalanche_windows[timestamp],
                    "unique_tags": len(unique_tags),
                    "first_alarm_tag": (
                        first_alarm["tag"] if first_alarm is not None else None
                    ),
                    "first_alarm_desc": (
                        first_alarm["description"] if first_alarm is not None else None
                    ),
                    "top_tags": ", ".join(tag_counts.head(5).index.tolist()),
                }
            )

        avalanche_df = pd.DataFrame(avalanche_details)

        # Summary statistics
        summary = {
            "total_avalanches": len(avalanche_df),
            "avg_alarms_per_avalanche": avalanche_df["total_alarms"].mean(),
            "max_alarms_in_avalanche": avalanche_df["total_alarms"].max(),
            "most_common_first_alarm": (
                avalanche_df["first_alarm_tag"].mode().iloc[0]
                if not avalanche_df.empty
                else None
            ),
        }

        logger.info(f"Detected {len(avalanche_df)} avalanche events.")
        return avalanche_df, pd.DataFrame([summary])

    # -------------------------------------------------------------------------
    # NUISANCE ALARM
    # -------------------------------------------------------------------------

    def identify_nuisance_alarms(self, chatter_df: pd.DataFrame) -> pd.DataFrame:
        # Nuisance alarms are those that:
        #    - Activate frequently (high occurrence)
        #    - Chatter (rapid on/off cycling)
        #    - Have high repeatability
        #    - May not require operator action
        #
        # Per ISA-18.2, nuisance alarms should be:
        #    - Rationalized for necessity
        #    - Reconfigured (setpoints, deadbands, delays)
        #    - Potentially removed if not required

        logger.info("Identifying nuisance alarms...")

        # Get repeatability data
        repeat_df = self.analyze_repeatability()

        # Merge chatter and repeatability data
        if chatter_df.empty or repeat_df.empty:
            return pd.DataFrame()

        nuisance_df = chatter_df.merge(
            repeat_df[["tag", "repeatability_index", "repeat_count"]],
            on="tag",
            how="outer",
        ).fillna(0)

        # Calculate nuisance score (weighted combination)
        # Normalize metrics to 0-1 scale
        if nuisance_df["chatter_index"].max() > 0:
            nuisance_df["chatter_norm"] = (
                nuisance_df["chatter_index"] / nuisance_df["chatter_index"].max()
            )
        else:
            nuisance_df["chatter_norm"] = 0

        if nuisance_df["repeatability_index"].max() > 0:
            nuisance_df["repeat_norm"] = (
                nuisance_df["repeatability_index"]
                / nuisance_df["repeatability_index"].max()
            )
        else:
            nuisance_df["repeat_norm"] = 0

        if nuisance_df["total_activations"].max() > 0:
            nuisance_df["freq_norm"] = (
                nuisance_df["total_activations"]
                / nuisance_df["total_activations"].max()
            )
        else:
            nuisance_df["freq_norm"] = 0

        # Weighted nuisance score
        nuisance_df["nuisance_score"] = (
            0.35 * nuisance_df["chatter_norm"]
            + 0.35 * nuisance_df["repeat_norm"]
            + 0.30 * nuisance_df["freq_norm"]
        )

        # Classify nuisance level
        nuisance_df["nuisance_level"] = pd.cut(
            nuisance_df["nuisance_score"],
            bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
            labels=["Low", "Moderate", "High", "Critical"],
        )

        result = nuisance_df[
            [
                "tag",
                "total_activations",
                "chatter_index",
                "repeatability_index",
                "nuisance_score",
                "nuisance_level",
            ]
        ].sort_values("nuisance_score", ascending=False)

        return result

    # -------------------------------------------------------------------------
    # TEMPORAL DISTRIBUTION ANALYSIS
    # -------------------------------------------------------------------------

    def analyze_temporal(self) -> Dict[str, pd.Series]:
        logger.info("Analyzing temporal distribution...")

        df_temp = self.df_alarms.copy()
        df_temp["hour"] = df_temp["timestamp"].dt.hour
        df_temp["day_of_week"] = df_temp["timestamp"].dt.day_name()

        # Hourly distribution
        hourly_counts = df_temp["hour"].value_counts().sort_index()
        for h in range(24):
            if h not in hourly_counts.index:
                hourly_counts[h] = 0
        hourly_counts = hourly_counts.sort_index()

        # Day of week distribution
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        dow_counts = df_temp["day_of_week"].value_counts()
        dow_counts = dow_counts.reindex(day_order, fill_value=0)

        return {"hourly": hourly_counts, "day_of_week": dow_counts}

    # -------------------------------------------------------------------------
    # PATTERN ANALYSIS (FP-Growth Algorithm)
    # -------------------------------------------------------------------------

    def analyze_patterns(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Identifies alarms that frequently occur together, which may indicate:
        #    - Redundant alarms (same root cause)
        #    - Consequential alarms (cause-effect relationships)
        #    - Opportunities for alarm suppression or grouping

        logger.info("Mining frequent alarm patterns (FP-Growth)...")

        # Create transaction basket (alarms occurring in same time window)
        basket = (
            self.df_alarms.groupby(
                [pd.Grouper(key="timestamp", freq=self.window_size), "tag"]
            )["tag"]
            .count()
            .unstack()
            .reset_index()
            .fillna(0)
            .set_index("timestamp")
        )

        # Convert to binary
        basket_binary = basket > 0
        basket_binary = basket_binary[basket_binary.sum(axis=1) > 0]

        if basket_binary.empty or basket_binary.shape[1] < 2:
            logger.warning("Insufficient data for pattern mining.")
            return pd.DataFrame(), pd.DataFrame()

        try:
            frequent_itemsets = fpgrowth(
                basket_binary, min_support=self.min_support, use_colnames=True
            )

            rules = pd.DataFrame()
            if not frequent_itemsets.empty:
                rules = association_rules(
                    frequent_itemsets, metric="lift", min_threshold=1.0
                )

            return frequent_itemsets, rules

        except Exception as e:
            logger.error(f"Pattern mining failed: {e}")
            return pd.DataFrame(), pd.DataFrame()

    # -------------------------------------------------------------------------
    # SUPPRESSION RECOMMENDATIONS
    # -------------------------------------------------------------------------

    def recommend_suppressions(
        self,
        rules_df: pd.DataFrame,
        lift_threshold: float = 3.0,
        confidence_threshold: float = 0.7,
    ) -> pd.DataFrame:
        # Generating alarm suppression recommendations based on association rules.
        #
        # When alarms have strong causal relationships (high lift and confidence),
        # the consequent alarm may be a candidate for suppression when the
        # antecedent alarm is active.
        #
        # Uses FP-Growth, as it is more efficient than Apriori

        logger.info("Generating suppression recommendations...")

        if rules_df.empty:
            return pd.DataFrame()

        # Filter for strong rules
        strong_rules = rules_df[
            (rules_df["lift"] > lift_threshold)
            & (rules_df["confidence"] > confidence_threshold)
        ].copy()

        if strong_rules.empty:
            return pd.DataFrame()

        recommendations = []

        for _, rule in strong_rules.iterrows():
            antecedents = ", ".join(sorted([str(x) for x in rule["antecedents"]]))
            consequents = ", ".join(sorted([str(x) for x in rule["consequents"]]))

            recommendations.append(
                {
                    "trigger_alarms": antecedents,
                    "suppress_candidates": consequents,
                    "confidence": round(rule["confidence"] * 100, 1),
                    "lift": round(rule["lift"], 2),
                    "support": round(rule["support"] * 100, 2),
                    "recommendation": f"Consider suppressing '{consequents}' when '{antecedents}' is active",
                }
            )

        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values("lift", ascending=False)

        return rec_df

    # -------------------------------------------------------------------------
    # SEQUENCE ANALYSIS (Markov Transition)
    # -------------------------------------------------------------------------

    def analyze_sequences(self, top_n: int = 15) -> pd.DataFrame:
        # Identifies which alarms tend to follow other alarms, useful for:
        #   - Root cause analysis
        #   - Predictive alarm management
        #   - Operator training on alarm sequences

        logger.info(f"Analyzing alarm sequences (top {top_n} alarms)...")

        top_alarms = self.df_alarms["tag"].value_counts().head(top_n).index.tolist()
        df_seq = (
            self.df_alarms[self.df_alarms["tag"].isin(top_alarms)]
            .sort_values("timestamp")
            .copy()
        )

        if len(df_seq) < 2:
            return pd.DataFrame()

        df_seq["next_alarm"] = df_seq["tag"].shift(-1)
        df_seq = df_seq.dropna(subset=["next_alarm"])

        # Create transition matrix
        transition_matrix = pd.crosstab(
            df_seq["tag"], df_seq["next_alarm"], normalize="index"
        )

        return transition_matrix


# ==============================================================================
# PDF REPORT GENERATION
# ==============================================================================


# Built it as a class for easier management, however, not neccessary
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Alarms Analysis Report", 0, 1, "C")
        self.set_font("Arial", "I", 10)
        self.cell(0, 5, "Per ISA-18.2 and EEMUA 191 Standards", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def chapter_title(self, title: str):
        self.set_font("Arial", "B", 12)
        self.set_fill_color(70, 130, 180)  # Steel blue
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, title, 0, 1, "L", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def section_subtitle(self, title: str):
        self.set_font("Arial", "B", 10)
        self.set_text_color(70, 130, 180)
        self.cell(0, 6, title, 0, 1, "L")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_chart(self, image_path: str, width: int = 170):
        if os.path.exists(image_path):
            self.image(image_path, x=20, w=width)
            self.ln(5)

    def create_table(self, df: pd.DataFrame, col_widths: Optional[List[int]] = None):
        if df.empty:
            self.body_text("No data available.")
            return

        # Calculate column widths
        if col_widths is None:
            available_width = self.w - 40
            col_widths = [available_width / len(df.columns)] * len(df.columns)

        row_height = 6

        # Header
        self.set_font("Arial", "B", 8)
        self.set_fill_color(240, 240, 240)
        for i, col in enumerate(df.columns):
            text = str(col)[:20]
            self.cell(col_widths[i], row_height, text, border=1, fill=True)
        self.ln(row_height)

        # Data rows
        self.set_font("Courier", "", 7)
        for _, row in df.head(15).iterrows():
            for i, item in enumerate(row):
                text = str(item)
                text = text.encode("latin-1", "replace").decode("latin-1")
                if len(text) > 25:
                    text = text[:22] + "..."
                self.cell(col_widths[i], row_height, text, border=1)
            self.ln(row_height)

        if len(df) > 15:
            self.set_font("Arial", "I", 8)
            self.cell(0, 5, f"... and {len(df) - 15} more rows", 0, 1)

        self.ln(3)

    def status_box(self, status: str, description: str):
        colors = {
            "Very Good": (46, 139, 87),  # Sea green
            "Acceptable": (60, 179, 113),  # Medium sea green
            "Too Many": (255, 165, 0),  # Orange
            "Extremely Dangerous": (220, 20, 60),  # Crimson
        }

        color = colors.get(status, (128, 128, 128))

        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 11)
        self.cell(50, 8, f"  {status}", 0, 0, "L", fill=True)
        self.set_text_color(0, 0, 0)
        self.set_font("Arial", "", 10)
        self.cell(0, 8, f"  {description}", 0, 1)
        self.ln(2)


def generate_charts(
    analyzer: AlarmAnalyzer,
    freq_counts: pd.Series,
    chatter_df: pd.DataFrame,
    temporal_data: Dict[str, pd.Series],
    transition_matrix: pd.DataFrame,
    bad_actor_data: Dict[str, Any],
    nuisance_df: pd.DataFrame,
    top_n: int,
) -> Dict[str, Optional[str]]:

    charts = {}

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "primary": "#4C72B0",
        "secondary": "#55A868",
        "warning": "#DD8452",
        "danger": "#C44E52",
        "info": "#8172B3",
    }

    # Frequency Bar Chart
    plt.figure(figsize=(10, 6))
    freq_counts.sort_values().plot(
        kind="barh", color=colors["primary"], edgecolor="white"
    )
    plt.title(f"Top {top_n} Most Frequent Alarms", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Activations")
    plt.ylabel("Alarm Tag")
    plt.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    charts["freq"] = path

    # Hourly Distribution
    plt.figure(figsize=(10, 5))
    hourly = temporal_data["hourly"]
    bars = plt.bar(
        hourly.index,
        hourly.values,
        color=colors["secondary"],
        edgecolor="white",
        width=0.8,
    )
    plt.title("Alarm Distribution by Hour of Day", fontsize=12, fontweight="bold")
    plt.xlabel("Hour (24-hour format)")
    plt.ylabel("Number of Alarms")
    plt.xticks(range(0, 24))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Highlight peak hours
    max_val = hourly.max()
    for bar, val in zip(bars, hourly.values):
        if val >= max_val * 0.9:
            bar.set_color(colors["warning"])

    plt.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    charts["hourly"] = path

    # High Density Alarm Plot (HDAP)
    top_25 = freq_counts.head(25).index
    df_hdap = analyzer.df_alarms[analyzer.df_alarms["tag"].isin(top_25)].copy()

    plt.figure(figsize=(12, 8))
    plt.scatter(
        df_hdap["timestamp"],
        df_hdap["tag"],
        marker="|",
        alpha=0.7,
        s=100,
        color=colors["danger"],
    )
    plt.title(
        "High Density Alarm Plot (HDAP) - Top 25 Tags", fontsize=12, fontweight="bold"
    )
    plt.xlabel("Time")
    plt.ylabel("Alarm Tag")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.gcf().autofmt_xdate()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    charts["hdap"] = path

    # Chatter Index Chart
    if not chatter_df.empty:
        plt.figure(figsize=(10, 6))
        top_chatter = chatter_df.head(10).sort_values(
            by="chatter_index", ascending=True
        )
        bars = plt.barh(
            top_chatter["tag"],
            top_chatter["chatter_index"],
            color=colors["warning"],
            edgecolor="white",
        )
        plt.axvline(
            x=analyzer.chatter_cutoff,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({analyzer.chatter_cutoff})",
        )
        plt.title("Top 10 Chattering Alarms", fontsize=12, fontweight="bold")
        plt.xlabel("Chatter Index")
        plt.ylabel("Alarm Tag")
        plt.legend(loc="lower right")
        plt.tight_layout()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        charts["chatter"] = path
    else:
        charts["chatter"] = None

    # Bad Actor Pareto Chart
    if bad_actor_data and "all_counts" in bad_actor_data:
        plt.figure(figsize=(12, 6))

        counts = bad_actor_data["all_counts"].head(20)
        cumulative_pct = counts.cumsum() / counts.sum() * 100

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.bar(
            range(len(counts)),
            counts.values,
            color=colors["primary"],
            edgecolor="white",
        )
        ax1.set_xlabel("Alarm Tag")
        ax1.set_ylabel("Number of Activations", color=colors["primary"])
        ax1.set_xticks(range(len(counts)))
        ax1.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)

        ax2 = ax1.twinx()
        ax2.plot(
            range(len(counts)),
            cumulative_pct.values,
            color=colors["danger"],
            marker="o",
            linewidth=2,
        )
        ax2.axhline(y=80, color="gray", linestyle="--", alpha=0.7)
        ax2.set_ylabel("Cumulative Percentage (%)", color=colors["danger"])
        ax2.set_ylim(0, 105)

        plt.title("Bad Actor Analysis (Pareto Chart)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        charts["pareto"] = path
    else:
        charts["pareto"] = None

    # Nuisance Alarm Chart
    if not nuisance_df.empty:
        plt.figure(figsize=(10, 6))
        top_nuisance = nuisance_df.head(10).sort_values(
            by="nuisance_score", ascending=True
        )

        level_colors = {
            "Critical": colors["danger"],
            "High": colors["warning"],
            "Moderate": colors["info"],
            "Low": colors["secondary"],
        }
        bar_colors = [
            level_colors.get(str(level), "gray")
            for level in top_nuisance["nuisance_level"]
        ]

        plt.barh(
            top_nuisance["tag"],
            top_nuisance["nuisance_score"],
            color=bar_colors,
            edgecolor="white",
        )
        plt.title("Top 10 Nuisance Alarm Candidates", fontsize=12, fontweight="bold")
        plt.xlabel("Nuisance Score")
        plt.ylabel("Alarm Tag")
        plt.tight_layout()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        charts["nuisance"] = path
    else:
        charts["nuisance"] = None

    # Sequence Heatmap
    if not transition_matrix.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            transition_matrix,
            annot=False,
            cmap="Blues",
            linewidths=0.5,
            cbar_kws={"label": "Transition Probability"},
        )
        plt.title(
            f"Alarm Sequence Probability Matrix (Top {top_n} Alarms)",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Subsequent Alarm")
        plt.ylabel("Preceding Alarm")
        plt.tight_layout()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        charts["sequence"] = path
    else:
        charts["sequence"] = None

    # Day of Week Distribution
    plt.figure(figsize=(10, 5))
    dow = temporal_data["day_of_week"]
    plt.bar(dow.index, dow.values, color=colors["info"], edgecolor="white")
    plt.title("Alarm Distribution by Day of Week", fontsize=12, fontweight="bold")
    plt.xlabel("Day")
    plt.ylabel("Number of Alarms")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    charts["dow"] = path

    return charts


def build_pdf_report(
    analyzer: AlarmAnalyzer,
    health_metrics: Dict[str, Any],
    freq_counts: pd.Series,
    temporal_data: Dict[str, pd.Series],
    chatter_df: pd.DataFrame,
    stale_df: pd.DataFrame,
    standing_df: pd.DataFrame,
    bad_actor_data: Dict[str, Any],
    repeatability_df: pd.DataFrame,
    avalanche_df: pd.DataFrame,
    nuisance_df: pd.DataFrame,
    itemsets: pd.DataFrame,
    rules: pd.DataFrame,
    suppression_df: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    charts: Dict[str, Optional[str]],
    output_path: str,
    args: argparse.Namespace,
) -> None:

    logger.info("Building PDF report...")

    pdf = PDFReport()
    pdf.add_page()

    # =========================================================================
    # SECTION 1: EXECUTIVE SUMMARY
    # =========================================================================
    pdf.chapter_title("1. Executive Summary")

    pdf.section_subtitle("Analysis Overview")
    pdf.body_text(f"Data File: {os.path.basename(analyzer.file_path)}")
    pdf.body_text(f"Analysis Period: {analyzer.start_time} to {analyzer.end_time}")
    pdf.body_text(f"Duration: {analyzer.duration_hours:.1f} hours")
    pdf.body_text(f"Total Records: {analyzer.total_alarms:}")
    pdf.body_text(f"Alarm Activations: {analyzer.total_activations:}")
    pdf.body_text(f"Alarm Recoveries: {analyzer.total_recoveries:}")
    pdf.ln(5)

    pdf.section_subtitle("Alarm System Health Assessment (EEMUA 191)")
    pdf.status_box(health_metrics["status"], health_metrics["status_description"])
    pdf.body_text(f"Average Alarms per 10 Minutes: {health_metrics['avg_10min']:.2f}")
    pdf.body_text(f"Peak Alarms per 10 Minutes: {health_metrics['peak_rate']:.0f}")
    pdf.body_text(f"Peak-to-Average Ratio: {health_metrics['peak_to_avg_ratio']:.2f}")
    pdf.ln(3)

    pdf.section_subtitle("Flood Analysis (ISA-18.2)")
    pdf.body_text(
        f"ISA-18.2 defines an alarm flood as more than {ISA_FLOOD_THRESHOLD} alarms "
        f"within a 10-minute period. During this analysis period:"
    )
    pdf.body_text(f"Flood Intervals Detected: {health_metrics['flood_count']}")
    pdf.body_text(f"Percentage of Time in Flood: {health_metrics['flood_pct']:.1f}%")
    pdf.body_text(f"Total 10-Minute Intervals: {health_metrics['total_intervals']}")

    # =========================================================================
    # SECTION 2: TEMPORAL ANALYSIS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("2. Temporal Analysis")

    pdf.body_text(
        "Temporal analysis identifies patterns in alarm occurrence by time of day "
        "and day of week. This information supports staffing decisions and helps "
        "identify process conditions that generate excessive alarms."
    )

    pdf.section_subtitle("Hourly Distribution")
    if charts.get("hourly"):
        pdf.add_chart(charts["hourly"])

    pdf.section_subtitle("Day of Week Distribution")
    if charts.get("dow"):
        pdf.add_chart(charts["dow"])

    pdf.add_page()
    pdf.section_subtitle("High Density Alarm Plot (HDAP)")
    pdf.body_text(
        "The HDAP displays individual alarm activations over time for the most "
        "frequent alarm tags. Vertical clustering indicates periods of high activity."
    )
    if charts.get("hdap"):
        pdf.add_chart(charts["hdap"])

    # =========================================================================
    # SECTION 3: BAD ACTOR ANALYSIS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("3. Bad Actor Analysis (ISA-18.2 Section 11)")

    pdf.body_text(
        "Per ISA-18.2, 'bad actors' are alarms that contribute disproportionately "
        "to the total alarm load. Regular review of the most frequent alarms is "
        "essential for continuous improvement. The Pareto principle typically applies: "
        "a small percentage of alarms generate the majority of activations."
    )

    if bad_actor_data:
        pdf.body_text(
            f"Alarms Contributing to 80% of Load: {bad_actor_data['bad_actor_count']}"
        )
        pdf.body_text(f"Total Unique Alarms: {bad_actor_data['total_alarm_count']}")
        pdf.body_text(f"Bad Actor Percentage: {bad_actor_data['bad_actor_pct']:.1f}%")
        pdf.ln(3)

        if charts.get("pareto"):
            pdf.add_chart(charts["pareto"])

        pdf.section_subtitle(f"Top {args.top_n} Most Frequent Alarms")
        if charts.get("freq"):
            pdf.add_chart(charts["freq"])

    # =========================================================================
    # SECTION 4: CHATTER ANALYSIS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("4. Alarm Chatter Analysis")

    pdf.body_text(
        "Chattering alarms repeatedly activate and clear in rapid succession. "
        "Per ISA-18.2, chattering should be addressed through deadband adjustment, "
        "timer/delay implementation, or signal filtering. The chatter index quantifies "
        "the frequency of rapid cycling."
    )
    pdf.body_text(
        'Chattering index is calculated based on "Quantification of Alarm Chatter Based on Run Length Distributions" (2010 Kondaveeti et al.)'
    )
    pdf.body_text(f"Chatter Threshold: {analyzer.chatter_cutoff}")
    pdf.ln(3)

    if charts.get("chatter"):
        pdf.add_chart(charts["chatter"])

    if not chatter_df.empty:
        pdf.section_subtitle("Top Chattering Alarms")
        display_df = (
            chatter_df[["tag", "chatter_index", "total_activations", "status"]]
            .head(10)
            .copy()
        )
        display_df["chatter_index"] = display_df["chatter_index"].round(4)
        pdf.create_table(display_df)

    # =========================================================================
    # SECTION 5: REPEATABILITY ANALYSIS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("5. Repeatability Analysis")

    pdf.body_text(
        f"Repeatability measures how often an alarm re-activates within a "
        f"{analyzer.repeatability_window_minutes}-minute window. High repeatability "
        f"may indicate insufficient deadband, process instability, or need for "
        f"alarm delay configuration."
    )

    if not repeatability_df.empty:
        pdf.section_subtitle("Top Repeating Alarms")
        display_df = repeatability_df[
            ["tag", "total_activations", "repeat_count", "repeatability_index"]
        ].head(10)
        pdf.create_table(display_df)

    # =========================================================================
    # SECTION 6: NUISANCE ALARM IDENTIFICATION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("6. Nuisance Alarm Identification")

    pdf.body_text(
        "Nuisance alarms are those that activate frequently, chatter, or have "
        "high repeatability. Per ISA-18.2, nuisance alarms should be rationalized "
        "for necessity and reconfigured or removed as appropriate. The nuisance score "
        "combines multiple factors: chatter index (35%), repeatability (35%), and "
        "occurrence frequency (30%)."
    )

    if charts.get("nuisance"):
        pdf.add_chart(charts["nuisance"])

    if not nuisance_df.empty:
        pdf.section_subtitle("Nuisance Alarm Ranking")
        display_df = (
            nuisance_df[
                ["tag", "total_activations", "nuisance_score", "nuisance_level"]
            ]
            .head(10)
            .copy()
        )
        display_df["nuisance_score"] = display_df["nuisance_score"].round(3)
        pdf.create_table(display_df)

    # =========================================================================
    # SECTION 7: STALE ALARM ANALYSIS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("7. Stale Alarm Analysis (ISA-18.2)")

    pdf.body_text(
        f"Stale alarms remain active for extended periods (>{analyzer.stale_threshold_hours} hours) "
        f"without being addressed. Per ISA-18.2, stale alarms contribute to operator "
        f"overload and may indicate process issues requiring attention."
    )

    if not stale_df.empty:
        pdf.body_text(f"Stale Alarm Instances Found: {len(stale_df)}")
        pdf.ln(3)

        pdf.section_subtitle("Stale Alarm Details")
        display_df = stale_df[
            ["tag", "condition", "active_duration_hours", "still_active"]
        ].head(10)
        pdf.create_table(display_df)
    else:
        pdf.body_text("No stale alarms detected during the analysis period.")

    # =========================================================================
    # SECTION 8: STANDING ALARM ANALYSIS
    # =========================================================================
    pdf.chapter_title("8. Standing Alarm Analysis")

    pdf.body_text(
        "Standing alarms are those that activate but never return to normal. "
        "Per EEMUA 191, standing alarms reduce operator situational awareness "
        "and may indicate permanent process deviations or configuration issues."
    )

    if not standing_df.empty:
        pdf.body_text(f"Standing Alarms Found: {len(standing_df)}")
        pdf.ln(3)

        pdf.section_subtitle("Standing Alarm Details")
        display_df = standing_df[
            [
                "tag",
                "condition",
                "total_activations",
                "total_recoveries",
                "uncleared_count",
            ]
        ].head(10)
        pdf.create_table(display_df)
    else:
        pdf.body_text("No standing alarms detected during the analysis period.")

    # =========================================================================
    # SECTION 9: AVALANCHE DETECTION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("9. Alarm Avalanche Detection (ISA-18.2)")

    pdf.body_text(
        "An alarm avalanche occurs when many alarms activate in rapid succession, "
        "often due to a single root cause. ISA-18.2 recommends identifying root "
        "cause alarms and implementing state-based suppression strategies."
    )

    if not avalanche_df.empty:
        pdf.body_text(f"Avalanche Events Detected: {len(avalanche_df)}")
        pdf.body_text(
            f"Average Alarms per Avalanche: {avalanche_df['total_alarms'].mean():.1f}"
        )
        pdf.body_text(
            f"Maximum Alarms in Single Avalanche: {avalanche_df['total_alarms'].max()}",
        )
        pdf.ln(3)

        pdf.section_subtitle("Avalanche Event Details")
        display_df = (
            avalanche_df[
                ["start_time", "total_alarms", "unique_tags", "first_alarm_tag"]
            ]
            .head(10)
            .copy()
        )
        display_df["start_time"] = display_df["start_time"].dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        pdf.create_table(display_df)
    else:
        pdf.body_text("No alarm avalanches detected during the analysis period.")

    # =========================================================================
    # SECTION 10: PATTERN ANALYSIS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("10. Alarm Pattern Analysis (Co-occurrence)")

    pdf.body_text(
        f"Pattern mining identifies alarms that frequently occur together within "
        f"a {args.window_size} window. Co-occurring alarms may indicate redundancy "
        f"or opportunities for alarm grouping and suppression."
    )
    pdf.body_text(f"Analysis Window: {args.window_size}")
    pdf.body_text(f"Minimum Support Threshold: {args.min_support*100:.1f}%")
    pdf.ln(3)

    if not itemsets.empty:
        itemsets_copy = itemsets.copy()
        itemsets_copy["length"] = itemsets_copy["itemsets"].apply(len)
        multi_item = (
            itemsets_copy[itemsets_copy["length"] > 1]
            .sort_values("support", ascending=False)
            .head(10)
        )

        if not multi_item.empty:
            pdf.section_subtitle("Frequent Alarm Combinations")
            pdf.set_font("Courier", "", 8)
            for _, row in multi_item.iterrows():
                items = ", ".join(sorted([str(x) for x in row["itemsets"]]))
                items = items.encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 5, f"Support {row['support']*100:.1f}%: {items}")
                pdf.ln(1)
    else:
        pdf.body_text(
            "Insufficient data for pattern mining or no patterns found above threshold."
        )

    # =========================================================================
    # SECTION 11: SUPPRESSION RECOMMENDATIONS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("11. Suppression Recommendations")

    pdf.body_text(
        "Based on association rule mining, the following alarms are candidates for "
        "state-based suppression per ISA-18.2. When the trigger alarm is active, "
        "the consequent alarm may be suppressed to reduce operator load."
    )

    if not suppression_df.empty:
        pdf.section_subtitle("Recommended Suppressions")
        pdf.set_font("Courier", "", 8)
        for _, row in suppression_df.head(8).iterrows():
            trigger = (
                str(row["trigger_alarms"])
                .encode("latin-1", "replace")
                .decode("latin-1")
            )
            suppress = (
                str(row["suppress_candidates"])
                .encode("latin-1", "replace")
                .decode("latin-1")
            )
            pdf.multi_cell(
                0,
                4,
                f"Confidence: {row['confidence']:.0f}%, Lift: {row['lift']:.1f}\n"
                f"  Trigger: {trigger}\n"
                f"  Suppress: {suppress}",
            )
            pdf.ln(2)
    else:
        pdf.body_text(
            "No strong suppression candidates identified. Consider adjusting "
            "the analysis parameters (window size, support threshold) or "
            "review association rules manually."
        )

    # =========================================================================
    # SECTION 12: SEQUENCE ANALYSIS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("12. Alarm Sequence Analysis")

    pdf.body_text(
        "Sequence analysis identifies which alarms tend to follow other alarms. "
        "The transition probability matrix shows the likelihood of each subsequent "
        "alarm given the preceding alarm. This supports root cause analysis and "
        "operator training."
    )

    if charts.get("sequence"):
        pdf.add_chart(charts["sequence"])
    else:
        pdf.body_text("Insufficient data for sequence analysis.")

    # =========================================================================
    # SECTION 13: RECOMMENDATIONS SUMMARY
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("13. Summary of Recommendations")

    pdf.body_text(
        "Based on the analysis results, the following actions are recommended "
        "for alarm system improvement per ISA-18.2 and EEMUA 191 guidelines:"
    )
    pdf.ln(3)

    recommendations = []

    # Health-based recommendations
    if health_metrics["status"] in ["WARNING", "CRITICAL"]:
        recommendations.append(
            f"PRIORITY: System is {health_metrics['status']}. Initiate alarm rationalization "
            f"program to reduce average alarm rate to <2 per 10 minutes."
        )

    if health_metrics["flood_pct"] > 5:
        recommendations.append(
            f"Address alarm flood conditions ({health_metrics['flood_pct']:.1f}% of time). "
            f"Review root causes and implement state-based suppression."
        )

    # Chatter recommendations
    critical_chatter = (
        chatter_df[chatter_df["status"] == "CRITICAL"]
        if not chatter_df.empty
        else pd.DataFrame()
    )
    if not critical_chatter.empty:
        recommendations.append(
            f"Review {len(critical_chatter)} chattering alarms. Consider deadband adjustment, "
            f"timer implementation, or signal filtering."
        )

    # Stale alarm recommendations
    if not stale_df.empty:
        recommendations.append(
            f"Investigate {len(stale_df)} stale alarm instances. Review process conditions "
            f"and alarm configuration."
        )

    # Standing alarm recommendations
    if not standing_df.empty:
        recommendations.append(
            f"Address {len(standing_df)} standing alarms. Verify process conditions and "
            f"implement corrective actions."
        )

    # Bad actor recommendations
    if bad_actor_data and bad_actor_data["bad_actor_pct"] < 20:
        recommendations.append(
            f"Focus improvement efforts on the top {bad_actor_data['bad_actor_count']} alarms "
            f"({bad_actor_data['bad_actor_pct']:.1f}% of tags) that generate 80% of activations."
        )

    # Suppression recommendations
    if not suppression_df.empty:
        recommendations.append(
            f"Evaluate {len(suppression_df)} suppression opportunities identified through "
            f"pattern analysis."
        )

    if not recommendations:
        recommendations.append(
            "System performance is within acceptable limits. Continue monitoring."
        )

    pdf.set_font("Arial", "", 10)
    for i, rec in enumerate(recommendations, 1):
        pdf.multi_cell(0, 5, f"{i}. {rec}")
        pdf.ln(2)

    # =========================================================================
    # SAVE PDF
    # =========================================================================
    try:
        pdf.output(output_path)
        logger.info(f"Report saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}")
        raise


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Alarms Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alarms_analyzer_enhanced.py alarms.csv report.pdf
  python alarms_analyzer_enhanced.py alarms.csv report.pdf --top_n 20 --window_size 10min
  python alarms_analyzer_enhanced.py alarms.csv report.pdf --stale_hours 12 --chatter_cutoff 0.03

CSV Format Expected:
  Column 0: Index
  Column 1: Timestamp (datetime)
  Column 2: Tag (alarm identifier)
  Column 3: Description
  Column 4: Condition (e.g., HIGH, LOW, FAULT, with RECOVERED suffix for returns)
        """,
    )

    parser.add_argument("csv_path", type=str, help="Path to alarm data CSV file")
    parser.add_argument("output_name", type=str, help="Output PDF filename")
    parser.add_argument(
        "--top_n",
        type=int,
        default=15,
        help="Number of top alarms to analyze (default: 15)",
    )
    parser.add_argument(
        "--min_support",
        type=float,
        default=0.01,
        help="FP-Growth minimum support, e.g., 0.01 for 1%% (default: 0.01)",
    )
    parser.add_argument(
        "--window_size",
        type=str,
        default="5min",
        help="Time window for pattern analysis (default: 5min)",
    )
    parser.add_argument(
        "--chatter_cutoff",
        type=float,
        default=0.05,
        help="Chatter index threshold (default: 0.05)",
    )
    parser.add_argument(
        "--stale_hours",
        type=float,
        default=24.0,
        help="Stale alarm threshold in hours (default: 24)",
    )
    parser.add_argument(
        "--repeat_window",
        type=int,
        default=10,
        help="Repeatability window in minutes (default: 10)",
    )
    parser.add_argument(
        "--avalanche_threshold",
        type=int,
        default=10,
        help="Minimum alarms for avalanche detection (default: 10)",
    )
    parser.add_argument(
        "--lift_threshold",
        type=float,
        default=3.0,
        help="Minimum lift for suppression recommendations (default: 3.0)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file
    if not os.path.exists(args.csv_path):
        logger.error(f"Input file not found: {args.csv_path}")
        sys.exit(1)

    # Initialize analyzer
    analyzer = AlarmAnalyzer(
        file_path=args.csv_path,
        min_support=args.min_support,
        window_size=args.window_size,
        chatter_cutoff=args.chatter_cutoff,
        stale_threshold_hours=args.stale_hours,
        repeatability_window_minutes=args.repeat_window,
    )

    try:
        # Load data
        analyzer.load_data()

        # Run analyses
        logger.info("Running analyses...")

        freq_counts = analyzer.get_most_frequent(args.top_n)
        health_metrics = analyzer.analyze_health()
        temporal_data = analyzer.analyze_temporal()
        chatter_df = analyzer.analyze_chatter()
        stale_df = analyzer.analyze_stale_alarms()
        standing_df = analyzer.analyze_standing_alarms()
        bad_actor_data = analyzer.analyze_bad_actors()
        repeatability_df = analyzer.analyze_repeatability()
        avalanche_df, avalanche_summary = analyzer.detect_avalanches(
            threshold=args.avalanche_threshold
        )
        nuisance_df = analyzer.identify_nuisance_alarms(chatter_df)
        itemsets, rules = analyzer.analyze_patterns()
        suppression_df = analyzer.recommend_suppressions(
            rules, lift_threshold=args.lift_threshold
        )
        transition_matrix = analyzer.analyze_sequences(top_n=args.top_n)

        # Generate charts
        logger.info("Generating visualizations...")
        charts = generate_charts(
            analyzer=analyzer,
            freq_counts=freq_counts,
            chatter_df=chatter_df,
            temporal_data=temporal_data,
            transition_matrix=transition_matrix,
            bad_actor_data=bad_actor_data,
            nuisance_df=nuisance_df,
            top_n=args.top_n,
        )

        # Build PDF report
        build_pdf_report(
            analyzer=analyzer,
            health_metrics=health_metrics,
            freq_counts=freq_counts,
            temporal_data=temporal_data,
            chatter_df=chatter_df,
            stale_df=stale_df,
            standing_df=standing_df,
            bad_actor_data=bad_actor_data,
            repeatability_df=repeatability_df,
            avalanche_df=avalanche_df,
            nuisance_df=nuisance_df,
            itemsets=itemsets,
            rules=rules,
            suppression_df=suppression_df,
            transition_matrix=transition_matrix,
            charts=charts,
            output_path=args.output_name,
            args=args,
        )

        # Cleanup
        for path in charts.values():
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        logger.info("Analysis complete.")
        print(f"\n{'='*60}")
        print(f"SUCCESS: Report generated at {args.output_name}")
        print(f"{'='*60}")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
