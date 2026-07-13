from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.backtest import TRADING_DAYS, max_drawdown, performance_metrics
from src.research.decisions import choose_cost_aware_positions
from src.research.features import make_next_day_direction_dataset


TARGET_ASSET = "SPY"
TRAIN_MONTHS = 36
TRANSACTION_COST_BPS = 1.0
SHORT_BORROW_COST_BPS = 25.0
POSITION_MODE = "long_short"


@dataclass(frozen=True)
class RefreshSchedule:
    name: str
    offset: pd.DateOffset


SCHEDULES = (
    RefreshSchedule("six_month", pd.DateOffset(months=6)),
    RefreshSchedule("monthly", pd.DateOffset(months=1)),
    RefreshSchedule("biweekly", pd.DateOffset(weeks=2)),
)


def make_logistic_regression_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def make_refresh_periods(
    dates: pd.Series,
    *,
    train_months: int,
    schedule: RefreshSchedule,
) -> list[dict[str, pd.Timestamp | pd.Index | int]]:
    dates = pd.Series(pd.to_datetime(dates)).sort_values()
    first_date = dates.min()
    last_date = dates.max()
    test_start = first_date + pd.DateOffset(months=train_months)

    periods: list[dict[str, pd.Timestamp | pd.Index | int]] = []
    period_id = 1
    while test_start <= last_date:
        train_start = test_start - pd.DateOffset(months=train_months)
        test_end = test_start + schedule.offset

        train_mask = (dates >= train_start) & (dates < test_start)
        test_mask = (dates >= test_start) & (dates < test_end)

        if train_mask.sum() > 50 and test_mask.sum() > 0:
            periods.append(
                {
                    "period_id": period_id,
                    "train_start": train_start,
                    "train_end": test_start,
                    "test_start": test_start,
                    "test_end": min(test_end, last_date),
                    "train_index": dates[train_mask].index,
                    "test_index": dates[test_mask].index,
                }
            )
            period_id += 1

        test_start = test_end

    return periods


def estimate_expected_spy_return(
    score: np.ndarray,
    train_returns: pd.Series,
    y_train: pd.Series,
) -> np.ndarray:
    avg_up_return = train_returns[y_train == 1].mean()
    avg_down_return = train_returns[y_train == 0].mean()

    if pd.isna(avg_up_return):
        avg_up_return = train_returns.mean()
    if pd.isna(avg_down_return):
        avg_down_return = train_returns.mean()

    return score * avg_up_return + (1 - score) * avg_down_return


def predict_schedule(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    *,
    schedule: RefreshSchedule,
    train_months: int,
) -> pd.DataFrame:
    target_return_col = f"{TARGET_ASSET}_next_day_return"
    target_up_col = f"{TARGET_ASSET}_next_day_up"
    periods = make_refresh_periods(dataset["Date"], train_months=train_months, schedule=schedule)
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
                    "refresh_schedule": schedule.name,
                    "period_id": period["period_id"],
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

    predictions = pd.DataFrame(rows).sort_values(["refresh_schedule", "model", "Date"])
    if predictions.empty:
        return predictions

    position_blocks = []
    for (schedule_name, model_name), frame in predictions.groupby(["refresh_schedule", "model"], sort=False):
        frame = frame.sort_values("Date")
        positions = choose_cost_aware_positions(
            frame["expected_spy_return"],
            mode=POSITION_MODE,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            short_borrow_cost_bps=SHORT_BORROW_COST_BPS,
        )
        block = frame.copy()
        block["position"] = positions.to_numpy()
        position_blocks.append(block)

    predictions = pd.concat(position_blocks, ignore_index=True).sort_values(["refresh_schedule", "model", "Date"])
    predictions["turnover"] = predictions.groupby(["refresh_schedule", "model"])["position"].diff().abs()
    first_trade = predictions.groupby(["refresh_schedule", "model"])["position"].transform("first").abs()
    predictions["turnover"] = predictions["turnover"].fillna(first_trade)
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


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []

    for (schedule_name, model_name), frame in predictions.groupby(["refresh_schedule", "model"], sort=True):
        frame = frame.sort_values("Date").reset_index(drop=True)
        metrics = performance_metrics(
            frame["strategy_return"],
            name=model_name,
            exposure=frame["position"].abs(),
            turnover=frame["turnover"],
            transaction_cost=frame["transaction_cost"],
        )
        metrics["refresh_schedule"] = schedule_name
        metrics["train_months"] = TRAIN_MONTHS
        metrics["period_count"] = frame["period_id"].nunique()
        metrics["first_test_date"] = frame["Date"].min().date().isoformat()
        metrics["last_test_date"] = frame["Date"].max().date().isoformat()
        metrics["directional_accuracy"] = accuracy_score(frame["actual_up"], frame["predicted_up"])
        metrics["avg_net_exposure"] = frame["position"].mean()
        metrics["long_day_rate"] = (frame["position"] > 0).mean()
        metrics["short_day_rate"] = (frame["position"] < 0).mean()
        metrics["cash_day_rate"] = (frame["position"] == 0).mean()
        metrics["total_short_borrow_cost"] = frame["short_borrow_cost"].sum()
        summary_rows.append(metrics)

        buy_hold_metrics = performance_metrics(frame["actual_return"], name="oos_buy_hold_SPY")
        buy_hold_metrics["refresh_schedule"] = schedule_name
        buy_hold_metrics["train_months"] = TRAIN_MONTHS
        buy_hold_metrics["period_count"] = frame["period_id"].nunique()
        buy_hold_metrics["first_test_date"] = frame["Date"].min().date().isoformat()
        buy_hold_metrics["last_test_date"] = frame["Date"].max().date().isoformat()
        buy_hold_metrics["directional_accuracy"] = (frame["actual_return"] > 0).mean()
        buy_hold_metrics["avg_exposure"] = 1.0
        buy_hold_metrics["annual_turnover"] = 0.0
        buy_hold_metrics["total_transaction_cost"] = 0.0
        buy_hold_metrics["avg_net_exposure"] = 1.0
        buy_hold_metrics["long_day_rate"] = 1.0
        buy_hold_metrics["short_day_rate"] = 0.0
        buy_hold_metrics["cash_day_rate"] = 0.0
        buy_hold_metrics["total_short_borrow_cost"] = 0.0
        summary_rows.append(buy_hold_metrics)

        for period_id, period in frame.groupby("period_id", sort=True):
            period = period.sort_values("Date")
            spy_return = (1 + period["actual_return"]).prod() - 1
            model_return = (1 + period["strategy_return"]).prod() - 1
            period_rows.append(
                {
                    "refresh_schedule": schedule_name,
                    "model": model_name,
                    "period_id": int(period_id),
                    "test_start": period["Date"].min().date().isoformat(),
                    "test_end": period["Date"].max().date().isoformat(),
                    "days": len(period),
                    "spy_return": spy_return,
                    "our_return": model_return,
                    "our_minus_spy": model_return - spy_return,
                    "spy_max_drawdown": max_drawdown(period["actual_return"]),
                    "our_max_drawdown": max_drawdown(period["strategy_return"]),
                    "long_days": int((period["position"] > 0).sum()),
                    "short_days": int((period["position"] < 0).sum()),
                    "cash_days": int((period["position"] == 0).sum()),
                    "turnover": period["turnover"].sum(),
                    "transaction_cost": period["transaction_cost"].sum(),
                    "short_borrow_cost": period["short_borrow_cost"].sum(),
                    "winner": "our_model" if model_return > spy_return else "SPY",
                }
            )

    summary = pd.DataFrame(summary_rows)
    leading_cols = [
        "refresh_schedule",
        "strategy",
        "train_months",
        "period_count",
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
    summary = summary.sort_values(["strategy", "refresh_schedule"]).reset_index(drop=True)

    period_results = pd.DataFrame(period_rows).sort_values(["refresh_schedule", "period_id"])
    return summary, period_results


def main() -> None:
    project_root = Path.cwd()
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    close = pd.read_csv(processed_dir / "extended_close_prices.csv", index_col="Date", parse_dates=True)
    dataset = make_next_day_direction_dataset(close, target_asset=TARGET_ASSET).reset_index().rename(columns={"index": "Date"})
    target_cols = {f"{TARGET_ASSET}_next_day_return", f"{TARGET_ASSET}_next_day_up", "Date"}
    feature_cols = [col for col in dataset.columns if col not in target_cols]

    predictions = pd.concat(
        [
            predict_schedule(dataset, feature_cols, schedule=schedule, train_months=TRAIN_MONTHS)
            for schedule in SCHEDULES
        ],
        ignore_index=True,
    )
    summary, period_results = summarize_predictions(predictions)

    predictions.to_csv(processed_dir / "adaptive_retraining_ml_predictions.csv", index=False)
    summary.to_csv(processed_dir / "adaptive_retraining_ml_results.csv", index=False)
    summary.to_csv(reports_dir / "adaptive_retraining_summary.csv", index=False)
    period_results.to_csv(reports_dir / "adaptive_retraining_period_results.csv", index=False)

    display_cols = [
        "refresh_schedule",
        "strategy",
        "period_count",
        "rows",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "total_transaction_cost",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"Saved {reports_dir / 'adaptive_retraining_summary.csv'}")
    print(f"Saved {reports_dir / 'adaptive_retraining_period_results.csv'}")


if __name__ == "__main__":
    main()
