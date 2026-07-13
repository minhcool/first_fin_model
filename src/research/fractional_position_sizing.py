from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.research.adaptive_retraining import (
    SHORT_BORROW_COST_BPS,
    TARGET_ASSET,
    TRAIN_MONTHS,
    TRANSACTION_COST_BPS,
    RefreshSchedule,
    estimate_expected_spy_return,
    make_logistic_regression_model,
    make_refresh_periods,
)
from src.research.backtest import TRADING_DAYS, max_drawdown, performance_metrics
from src.research.decisions import choose_cost_aware_positions, choose_threshold_ladder_positions
from src.research.features import make_next_day_direction_dataset


@dataclass(frozen=True)
class SizingConfig:
    name: str
    allowed_positions: tuple[float, ...]
    full_position_edge: float | None
    threshold_ladder_bps: tuple[float, ...] | None = None
    threshold_ladder_sizes: tuple[float, ...] | None = None


QUARTER_GRID = tuple(np.round(np.arange(-1.0, 1.0 + 0.25, 0.25), 2))
HALF_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)


SIZING_CONFIGS = (
    SizingConfig("discrete_full", (-1.0, 0.0, 1.0), None),
    SizingConfig(
        "threshold_ladder_5_20bp",
        QUARTER_GRID,
        None,
        threshold_ladder_bps=(5.0, 10.0, 15.0, 20.0),
        threshold_ladder_sizes=(0.25, 0.50, 0.75, 1.0),
    ),
    SizingConfig("fractional_quarter_10bp_edge", QUARTER_GRID, 0.0010),
    SizingConfig("fractional_quarter_20bp_edge", QUARTER_GRID, 0.0020),
    SizingConfig("fractional_quarter_30bp_edge", QUARTER_GRID, 0.0030),
    SizingConfig("fractional_quarter_50bp_edge", QUARTER_GRID, 0.0050),
    SizingConfig("fractional_quarter_100bp_edge", QUARTER_GRID, 0.0100),
    SizingConfig("fractional_half_20bp_edge", HALF_GRID, 0.0020),
)


def make_base_predictions(dataset: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    target_return_col = f"{TARGET_ASSET}_next_day_return"
    target_up_col = f"{TARGET_ASSET}_next_day_up"
    schedule = RefreshSchedule("six_month", pd.DateOffset(months=6))
    periods = make_refresh_periods(dataset["Date"], train_months=TRAIN_MONTHS, schedule=schedule)
    rows: list[dict[str, object]] = []

    for period in periods:
        train = dataset.loc[period["train_index"]]
        test = dataset.loc[period["test_index"]]

        X_train = train[feature_cols]
        y_train = train[target_up_col].astype(int)
        X_test = test[feature_cols]
        y_test = test[target_up_col].astype(int)

        fitted = make_logistic_regression_model().fit(X_train, y_train)
        predicted_up = fitted.predict(X_test).astype(int)
        score = fitted.predict_proba(X_test)[:, 1]
        expected_spy_return = estimate_expected_spy_return(
            score,
            train[target_return_col],
            y_train,
        )

        for row_idx, date, actual_return, actual_up, pred, proba, expected_return in zip(
            test.index,
            test["Date"],
            test[target_return_col],
            y_test,
            predicted_up,
            score,
            expected_spy_return,
        ):
            rows.append(
                {
                    "split_id": period["period_id"],
                    "train_start": period["train_start"],
                    "train_end": period["train_end"],
                    "test_start": period["test_start"],
                    "test_end": period["test_end"],
                    "model": "logistic_regression",
                    "Date": date,
                    "actual_return": actual_return,
                    "actual_up": actual_up,
                    "predicted_up": pred,
                    "score": proba,
                    "expected_spy_return": expected_return,
                    "row_idx": row_idx,
                }
            )

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def apply_sizing_config(base_predictions: pd.DataFrame, config: SizingConfig) -> pd.DataFrame:
    predictions = base_predictions.copy()
    predictions["sizing_config"] = config.name
    if config.threshold_ladder_bps is not None and config.threshold_ladder_sizes is not None:
        positions = choose_threshold_ladder_positions(
            predictions["expected_spy_return"],
            thresholds_bps=config.threshold_ladder_bps,
            sizes=config.threshold_ladder_sizes,
        )
    else:
        positions = choose_cost_aware_positions(
            predictions["expected_spy_return"],
            mode="long_short",
            allowed_positions=config.allowed_positions,
            full_position_edge=config.full_position_edge,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            short_borrow_cost_bps=SHORT_BORROW_COST_BPS,
        )
    predictions["position"] = positions.to_numpy()
    predictions["turnover"] = predictions["position"].diff().abs().fillna(predictions["position"].abs())
    predictions["transaction_cost"] = predictions["turnover"] * (TRANSACTION_COST_BPS / 10_000)
    predictions["short_borrow_cost"] = (
        predictions["position"].clip(upper=0).abs() * (SHORT_BORROW_COST_BPS / 10_000 / TRADING_DAYS)
    )
    predictions["strategy_return"] = (
        predictions["position"] * predictions["actual_return"]
        - predictions["transaction_cost"]
        - predictions["short_borrow_cost"]
    )
    return predictions


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for config_name, frame in predictions.groupby("sizing_config", sort=True):
        frame = frame.sort_values("Date").reset_index(drop=True)
        metrics = performance_metrics(
            frame["strategy_return"],
            name="logistic_regression",
            exposure=frame["position"].abs(),
            turnover=frame["turnover"],
            transaction_cost=frame["transaction_cost"],
        )
        metrics["sizing_config"] = config_name
        metrics["split_count"] = frame["split_id"].nunique()
        metrics["first_test_date"] = frame["Date"].min().date().isoformat()
        metrics["last_test_date"] = frame["Date"].max().date().isoformat()
        metrics["directional_accuracy"] = accuracy_score(frame["actual_up"], frame["predicted_up"])
        metrics["avg_net_exposure"] = frame["position"].mean()
        metrics["long_day_rate"] = (frame["position"] > 0).mean()
        metrics["short_day_rate"] = (frame["position"] < 0).mean()
        metrics["cash_day_rate"] = (frame["position"] == 0).mean()
        metrics["total_short_borrow_cost"] = frame["short_borrow_cost"].sum()
        summary_rows.append(metrics)

        distribution = frame["position"].value_counts(normalize=True).sort_index()
        for position, day_rate in distribution.items():
            distribution_rows.append(
                {
                    "sizing_config": config_name,
                    "position": position,
                    "day_rate": day_rate,
                    "days": int((frame["position"] == position).sum()),
                }
            )

        for split_id, split in frame.groupby("split_id", sort=True):
            split = split.sort_values("Date")
            spy_return = (1 + split["actual_return"]).prod() - 1
            model_return = (1 + split["strategy_return"]).prod() - 1
            split_rows.append(
                {
                    "sizing_config": config_name,
                    "split_id": int(split_id),
                    "test_start": split["Date"].min().date().isoformat(),
                    "test_end": split["Date"].max().date().isoformat(),
                    "days": len(split),
                    "spy_return": spy_return,
                    "our_return": model_return,
                    "our_minus_spy": model_return - spy_return,
                    "spy_max_drawdown": max_drawdown(split["actual_return"]),
                    "our_max_drawdown": max_drawdown(split["strategy_return"]),
                    "avg_abs_position": split["position"].abs().mean(),
                    "avg_net_position": split["position"].mean(),
                    "long_day_rate": (split["position"] > 0).mean(),
                    "short_day_rate": (split["position"] < 0).mean(),
                    "cash_day_rate": (split["position"] == 0).mean(),
                    "turnover": split["turnover"].sum(),
                    "transaction_cost": split["transaction_cost"].sum(),
                    "short_borrow_cost": split["short_borrow_cost"].sum(),
                    "winner": "our_model" if model_return > spy_return else "SPY",
                }
            )

    summary = pd.DataFrame(summary_rows)
    leading_cols = [
        "sizing_config",
        "strategy",
        "split_count",
        "rows",
        "first_test_date",
        "last_test_date",
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "positive_day_rate",
        "directional_accuracy",
        "avg_exposure",
        "avg_net_exposure",
        "long_day_rate",
        "short_day_rate",
        "cash_day_rate",
        "annual_turnover",
        "total_transaction_cost",
        "total_short_borrow_cost",
    ]
    summary = summary[[col for col in leading_cols if col in summary.columns]]
    summary = summary.sort_values("sharpe", ascending=False).reset_index(drop=True)

    split_results = pd.DataFrame(split_rows).sort_values(["sizing_config", "split_id"])
    distribution = pd.DataFrame(distribution_rows).sort_values(["sizing_config", "position"])
    return summary, split_results, distribution


def main() -> None:
    project_root = Path.cwd()
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    close = pd.read_csv(processed_dir / "extended_close_prices.csv", index_col="Date", parse_dates=True)
    dataset = make_next_day_direction_dataset(close, target_asset=TARGET_ASSET).reset_index().rename(columns={"index": "Date"})
    target_cols = {f"{TARGET_ASSET}_next_day_return", f"{TARGET_ASSET}_next_day_up", "Date"}
    feature_cols = [col for col in dataset.columns if col not in target_cols]

    base_predictions = make_base_predictions(dataset, feature_cols)
    predictions = pd.concat(
        [apply_sizing_config(base_predictions, config) for config in SIZING_CONFIGS],
        ignore_index=True,
    )
    summary, split_results, distribution = summarize(predictions)

    predictions.to_csv(processed_dir / "fractional_position_sizing_predictions.csv", index=False)
    summary.to_csv(reports_dir / "fractional_position_sizing_summary.csv", index=False)
    split_results.to_csv(reports_dir / "fractional_position_sizing_split_results.csv", index=False)
    distribution.to_csv(reports_dir / "fractional_position_distribution.csv", index=False)

    display_cols = [
        "sizing_config",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "avg_net_exposure",
        "annual_turnover",
        "total_transaction_cost",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"Saved {reports_dir / 'fractional_position_sizing_summary.csv'}")
    print(f"Saved {reports_dir / 'fractional_position_sizing_split_results.csv'}")
    print(f"Saved {reports_dir / 'fractional_position_distribution.csv'}")


if __name__ == "__main__":
    main()
