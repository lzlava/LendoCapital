import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import time
from scipy import stats
from pathlib import Path
from src.data_utils import load_ohlcv
from src.data_utils import compute_returns

def fetch_earnings_events(tickers, start_date="2020-01-01",
                          end_date="2026-05-01", limit=50):
    """
    Simple function that grabs raw earning events data from Yahoo Finance.

    Input:
        tickers: list of tickers from portfolio (list)
        start_date: usually "2020-01-01" (str)
        end_date: here "2026-05-01" (str)
        limit: sets a limit to how many events we pull. Not too important here

    Output:
        dataframe consisting of event data
    """

    event_dfs = []
    failed_tickers = []

    for symbol in tickers:
        try:
            earnings = yf.Ticker(symbol).get_earnings_dates(limit=limit)

            if earnings is None or earnings.empty:
                failed_tickers.append(symbol)
                continue

            df = earnings.reset_index().copy()

            df = df.rename(columns={
                "Earnings Date": "announcement_datetime",
                "EPS Estimate": "eps_estimate",
                "Reported EPS": "reported_eps",
                "Surprise(%)": "surprise_pct"
            })

            df["ticker"] = symbol
            df["event_type"] = "earnings"

            # Convert timestamps to New York time
            df["announcement_datetime"] = (
                pd.to_datetime(df["announcement_datetime"], utc=True)
                .dt.tz_convert("America/New_York")
            )

            # Filter to requested period
            df = df[
                (df["announcement_datetime"] >= pd.Timestamp(start_date, tz="America/New_York")) &
                (df["announcement_datetime"] < pd.Timestamp(end_date, tz="America/New_York") + pd.Timedelta(days=1))
            ].copy()

            event_dfs.append(df)

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            failed_tickers.append(symbol)

    if not event_dfs:
        return pd.DataFrame(), failed_tickers

    events_df = pd.concat(event_dfs, ignore_index=True)

    events_df = events_df[
        ["ticker", "announcement_datetime", "event_type",
         "eps_estimate", "reported_eps", "surprise_pct"]
    ]

    events_df = (
        events_df
        .drop_duplicates(subset=["ticker", "announcement_datetime"])
        .sort_values(["ticker", "announcement_datetime"])
        .reset_index(drop=True)
    )

    return events_df, failed_tickers

def align_event_dates(events_df, ohlcv_df, benchmark_ticker="SPY"):
    """
    Aligns events to a trading day based on when the announcement was made. Produces a new column to the dataframe.

    Input:
        events_df: dataframe of events from the previous function
        ohlcv_df: loaded-in OHLCV data from load_ohlcv containing tickers & basic stock information
        benchmark_ticker: usually SPY (str)
    Output:
        appended column to events_df with the aligned date
    """

    events = events_df.copy()
    prices = ohlcv_df.copy()

    # Get trading calendar from SPY
    trading_days = (
        pd.to_datetime(
            prices.loc[
                prices["ticker"] == benchmark_ticker, "date"
            ]
        )
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
    )

    # Ensure announcement timestamps are in New York time
    events["announcement_datetime"] = pd.to_datetime(
        events["announcement_datetime"],
        utc=True
    ).dt.tz_convert("America/New_York")

    aligned_dates = []
    timing_labels = []

    for timestamp in events["announcement_datetime"]:

        announcement_date = timestamp.date()
        announcement_time = timestamp.time()

        if announcement_time < time(9, 30):
            timing = "before_market"
            candidate_date = announcement_date

        elif announcement_time >= time(16, 0):
            timing = "after_market"
            candidate_date = (
                pd.Timestamp(announcement_date)
                + pd.Timedelta(days=1)
            ).date()

        else:
            timing = "during_market"
            candidate_date = announcement_date

        # Find first trading day on or after candidate date
        position = trading_days.searchsorted(
            pd.Timestamp(candidate_date).to_datetime64()
        )

        if position < len(trading_days):
            aligned_dates.append(pd.Timestamp(trading_days[position]))
        else:
            aligned_dates.append(pd.NaT)

        timing_labels.append(timing)

    events["trading_event_date"] = aligned_dates
    events["announcement_timing"] = timing_labels

    return events

def build_event_windows(
    aligned_events_df,
    returns_df,
    benchmark_ticker="SPY",
    estimation_start=-130,
    estimation_end=-11,
    event_start=-1,
    event_end=1
):
    """
    This function produces three dataframes to include estimation windows, event windows, and a validation dataframe.

    Input:
        aligned_events_df: dataframe containing the aligned event days on the trading calendar
        returns_df: basic portfolio stock information with daily simple returns
        benchmark_ticker: usually "SPY" (str)
        estimation_start: start date of estimation window relative to event day (int)
        estimation_end: end date of estimation window relative to event day (int)
        event_start: start date of event window relative to event day (int)
        event_end: end date of event window relative to event day

    Output:
        estimation_windows_df: dataframe containing data of all estimation window data for each event
        event_windows_df: dataframe containing data of all event window data for each event
        validation_df: dataframe catalog of each event and whether or not the event has sufficient return history in the estimation window to be used
    """
    # Work on copies
    events = aligned_events_df.copy()
    returns = returns_df.copy()

    events["trading_event_date"] = pd.to_datetime(
        events["trading_event_date"]
    ).dt.normalize()

    returns["date"] = pd.to_datetime(
        returns["date"]
    ).dt.normalize()

    return_col = "daily_simple_return"

    # Validate unique ticker-date observations
    if returns.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker-date rows found.")

    # Create a wide return table:
    # index = trading date, columns = tickers
    return_matrix = returns.pivot(
        index="date",
        columns="ticker",
        values=return_col
    )

    # SPY determines the trading calendar
    trading_days = (
        returns.loc[
            returns["ticker"] == benchmark_ticker, "date"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    calendar = pd.DatetimeIndex(trading_days)

    estimation_records = []
    event_records = []
    validation_records = []

    # Assign a unique ID to each event
    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(len(events))

    for event in events.itertuples(index=False):

        ticker = event.ticker
        event_date = event.trading_event_date

        # Find the event's location on the trading calendar
        position = calendar.get_indexer([event_date])[0]

        if position == -1:
            validation_records.append({
                "event_id": event.event_id,
                "ticker": ticker,
                "event_date": event_date,
                "status": "invalid_event_date"
            })
            continue

        # Define relative trading days
        estimation_offsets = np.arange(
            estimation_start, estimation_end + 1
        )

        event_offsets = np.arange(
            event_start, event_end + 1
        )

        estimation_positions = position + estimation_offsets
        event_positions = position + event_offsets

        # Check that all requested dates exist
        if (
            estimation_positions.min() < 0
            or event_positions.min() < 0
            or estimation_positions.max() >= len(calendar)
            or event_positions.max() >= len(calendar)
        ):
            validation_records.append({
                "event_id": event.event_id,
                "ticker": ticker,
                "event_date": event_date,
                "status": "insufficient_history"
            })
            continue

        estimation_dates = calendar[estimation_positions]
        event_dates = calendar[event_positions]

        # Check ticker exists in return data
        if ticker not in return_matrix.columns:
            validation_records.append({
                "event_id": event.event_id,
                "ticker": ticker,
                "event_date": event_date,
                "status": "missing_ticker"
            })
            continue

        # Extract aligned stock and benchmark returns
        required_dates = estimation_dates.append(event_dates)

        selected = return_matrix.reindex(required_dates)

        stock_returns = selected[ticker]
        market_returns = selected[benchmark_ticker]

        # Require complete observations
        if stock_returns.isna().any() or market_returns.isna().any():
            validation_records.append({
                "event_id": event.event_id,
                "ticker": ticker,
                "event_date": event_date,
                "status": "missing_returns"
            })
            continue

        # Store estimation-window observations
        for relative_day, date in zip(
            estimation_offsets, estimation_dates
        ):
            estimation_records.append({
                "event_id": event.event_id,
                "ticker": ticker,
                "event_date": event_date,
                "date": date,
                "relative_day": relative_day,
                "stock_return": return_matrix.at[date, ticker],
                "market_return": return_matrix.at[
                    date, benchmark_ticker
                ]
            })

        # Store event-window observations
        for relative_day, date in zip(
            event_offsets, event_dates
        ):
            event_records.append({
                "event_id": event.event_id,
                "ticker": ticker,
                "event_date": event_date,
                "date": date,
                "relative_day": relative_day,
                "stock_return": return_matrix.at[date, ticker],
                "market_return": return_matrix.at[
                    date, benchmark_ticker
                ]
            })

        validation_records.append({
            "event_id": event.event_id,
            "ticker": ticker,
            "event_date": event_date,
            "status": "valid"
        })

    estimation_windows_df = pd.DataFrame(estimation_records)
    event_windows_df = pd.DataFrame(event_records)
    validation_df = pd.DataFrame(validation_records)

    return estimation_windows_df, event_windows_df, validation_df

def estimate_market_model(estimation_windows_df):
    """
    Uses linear regression (least squares) to find alpha, beta, and the residuals of each event.

    Input:
        estimation_windows_df: the dataframe consisting of all the estimation window data

    Output:
        a new dataframe consisting of the ticker, the event id, alpha, beta, and residuals
    """
    results = []

    for event_id, group in estimation_windows_df.groupby("event_id"):

        x = group["market_return"].to_numpy()
        y = group["stock_return"].to_numpy()

        # Market-model regression with an intercept
        X = np.column_stack([np.ones(len(x)), x])

        alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]

        predicted = alpha + beta * x
        residuals = y - predicted

        results.append({
            "event_id": event_id,
            "ticker": group["ticker"].iloc[0],
            "alpha": alpha,
            "beta": beta,
            "residual_std": np.std(residuals, ddof=2),
            "n_obs": len(group)
        })

    return pd.DataFrame(results)

def compute_abnormal_returns(event_windows_df, model_params_df):
    """
    Computes abnormal returns as a new column to event_windows_df.

    Input:
        event_windows_df: dataframe with all event window data
        model_params_df: dataframe with alpha & beta information to calculate normal returns
        
    Output:
        appended column to event_windows_df with expected (normal) return and abnormal return
    """

    df = event_windows_df.merge(
        model_params_df[
            ["event_id", "alpha", "beta"]
        ],
        on="event_id",
        how="inner",
        validate="many_to_one"
    )

    df["expected_return"] = (
        df["alpha"] + df["beta"] * df["market_return"]
    )

    df["abnormal_return"] = (
        df["stock_return"] - df["expected_return"]
    )

    return df

def compute_car(abnormal_returns_df):
    """
    Computes cumulative abnormal returns by adding the three abnormal returns per event together.

    Input:
        abnormal_returns_df: dataframe with event ids, relative days, and abnormal returns

    Output:
        old dataframe with appended column; dataframe with cumulative abnormal return
    """
    df = abnormal_returns_df.sort_values(
        ["event_id", "relative_day"]
    ).copy()

    # Running cumulative abnormal return
    df["cumulative_abnormal_return"] = (
        df.groupby("event_id")["abnormal_return"].cumsum()
    )

    # One CAR observation per earnings event
    car_df = (
        df.groupby(["event_id", "ticker", "event_date"], as_index=False)
        ["abnormal_return"]
        .sum()
        .rename(columns={
            "abnormal_return": "car"
        })
    )

    return df, car_df

def aggregate_abnormal_returns(abnormal_returns_df):
    """
    Computes AAR and CAAR using abnormal returns.

    Input:
        abnormal_returns_df: dataframe with abnormal returns

    Output:
        dataframe with relative day, AAR, and CAAR. CAAR column is cumulative so the last entry (bottom right) is the true CAAR value.
    """
    
    aggregate_df = (
        abnormal_returns_df
        .groupby("relative_day")
        .agg(
            aar=("abnormal_return", "mean"),
            std_ar=("abnormal_return", "std"),
            n_events=("event_id", "nunique")
        )
        .reset_index()
        .sort_values("relative_day")
    )

    aggregate_df["caar"] = aggregate_df["aar"].cumsum()

    return aggregate_df

def test_mean_significance(values, confidence=0.95):
    """
    This function tests if a dataset mean is significant from 0 via t-test.

    Input:
        values: list of values

    Output:
        series of metrics including dataset mean, standard deviation, t-statistic, p-value, and confidence intervals
    """

    values = pd.Series(values).dropna()

    n = len(values)

    if n < 2:
        raise ValueError("At least two observations required.")

    mean = values.mean()
    std = values.std(ddof=1)
    se = std / np.sqrt(n)

    if se == 0:
        raise ValueError("Zero standard error; t-test undefined.")

    t_stat = mean / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n-1)

    critical_value = stats.t.ppf(
        (1 + confidence) / 2,
        df=n-1
    )

    ci_lower = mean - critical_value * se
    ci_upper = mean + critical_value * se

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

def fetch_company_metadata(tickers):
    """
    Fetches company data from yfinance, including sector, market cap, and industry.
    Note that we don't use this market cap, but instead use the market cap at the time of each event by multiplying shares outstanding with price.
    
    Input:
        tickers: list of tickers to fetch data for
    Output:
        dataframe with the necessary information
    """

    records = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info

            records.append({
                "ticker": ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "current_market_cap": info.get("marketCap")
            })

        except Exception as e:
            print(f"Failed for {ticker}: {e}")

            records.append({
                "ticker": ticker,
                "sector": None,
                "industry": None,
                "current_market_cap": np.nan
            })

    return pd.DataFrame(records)

def classify_surprise(surprise_pct):
    """
    Simple function that classifies surprise level to
    """

    if pd.isna(surprise_pct):
        return "Unknown"

    if surprise_pct > 1: # Anything between -1% and 1% is pretty consistent with no surprise.
        return "Positive"

    elif surprise_pct < -1:
        return "Negative"

    else:
        return "Zero"

def summarize_car_by_group(df, group_col):
    """
    Provides a summary when separating CAR measurements by group

    Input:
        df: dataframe containing group and "car" information
        group_col: column indicating group

    Output:
        a summary of groupings and how they experience abnormal returns from earning announcements
    """

    return (
        df.groupby(group_col, observed=True)
        .agg(
            n_events=("car", "count"),
            mean_car=("car", "mean"),
            median_car=("car", "median"),
            std_car=("car", "std")
        )
        .reset_index()
    )

def test_group_significance(
    df,
    group_col,
    return_col="car",
    confidence=0.95
):
    """
    Tests whether or not the group mean is significant (distinguishable from 0).

    Input:
        df: master dataframe containing grouping information and car
        group_col: column with the groups being compared (str)
        return_col: column with car (str)
        confidence: confidence interval, usually 0.95 (fl)

    Output:
        dataframe of t-test results
    """
    results = []

    for group_name, group in df.groupby(
        group_col, observed=True, dropna=False
    ):
        values = group[return_col].dropna()

        n = len(values)

        if n < 2:
            continue

        mean = values.mean()
        median = values.median()
        std = values.std(ddof=1)
        se = std / np.sqrt(n)

        if se > 0:
            t_stat = mean / se
            p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)

            critical = stats.t.ppf(
                (1 + confidence) / 2, df=n - 1
            )

            ci_lower = mean - critical * se
            ci_upper = mean + critical * se
        else:
            t_stat = np.nan
            p_value = np.nan
            ci_lower = np.nan
            ci_upper = np.nan

        results.append({
            group_col: group_name,
            "n_events": n,
            "mean_car": mean,
            "median_car": median,
            "std_car": std,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant_5pct": (
                p_value < 0.05 if pd.notna(p_value) else False
            )
        })

    return pd.DataFrame(results)

def compare_groups(
    df,
    group_col,
    group_a,
    group_b,
    return_col="car",
    confidence=0.95
):
    """
    Performs Welch's t-test to compare two groups of data to test significance.

    Input:
        df: dataframe containing all the information
        group_col: the column indicating grouping (str)
        group_a: one of the specified compare groups (str)
        group_b: the other compare group (str)
        return_col: column of car, usually "car" (str)
        confidence: confidence interval, usually 0.95 (fl)

    Output:
        dataframe containing t-test results
    """
    a = df.loc[
        df[group_col] == group_a, return_col
    ].dropna()

    b = df.loc[
        df[group_col] == group_b, return_col
    ].dropna()

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least two observations.")

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(
        a, b, equal_var=False
    )

    mean_diff = a.mean() - b.mean()

    # Welch standard error
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    se = np.sqrt(
        var_a / len(a) + var_b / len(b)
    )

    # Welch-Satterthwaite degrees of freedom
    v1 = var_a / len(a)
    v2 = var_b / len(b)

    df_welch = (v1 + v2)**2 / (
        v1**2 / (len(a) - 1)
        + v2**2 / (len(b) - 1)
    )

    critical = stats.t.ppf(
        (1 + confidence) / 2,
        df=df_welch
    )

    return pd.DataFrame([{
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "mean_difference": mean_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": mean_diff - critical * se,
        "ci_upper": mean_diff + critical * se
    }])

def format_test_results(df):
    """
    Small function to help format t-test results and include effect sizes

    Input:
        df: dataframe of t-test results

    Output:
        dataframe with sorted, easy to read results
    """
    result = df.copy()

    return_cols = [
        "mean_car", "median_car",
        "mean_a", "mean_b", "mean_difference",
        "ci_lower", "ci_upper"
    ]

    for col in return_cols:
        if col in result.columns:
            result[col] = (result[col] * 10000).round(2)
            result = result.rename(
                columns={col: f"{col}_bps"}
            )

    for col in ["t_stat", "p_value"]:
        if col in result.columns:
            result[col] = result[col].round(4)

    return result

def plot_event_returns(
    df,
    title,
    group_col=None,
    groups=None,
    metric="caar",
    confidence=True,
    save_path=None,
    dpi=300
):
    """
    Plotting AAR/CAAR around earnings announcements, with possibility of groupings

    Input:
        df: dataframe with AAR results
        title: title of graph (str)
        group_col: grouping used to generate graph (str)
        groups: list of groups we use, meant to exclude "Unknown" categories (list)
        metric: type of metric, either "aar" or "caar"
        confidence: whether to display confidence intervals (bool)
        save_path: if to save figure. None to not save.
        dpi: quality of saved image (int)

    Output:
        plot of desired category comparing AAR/CAAR
    """
    if metric not in ["aar", "caar"]:
        raise ValueError("metric must be 'aar' or 'caar'")

    fig, ax = plt.subplots(figsize=(10, 6))

    # No grouping: plot the overall AAR or CAAR
    if group_col is None:
        plot_groups = [("Overall", df)]

    # Grouped: plot one curve per group
    else:
        if groups is None:
            groups = df[group_col].dropna().unique()

        plot_groups = [
            (name, df[df[group_col] == name])
            for name in groups
        ]

    for group_name, subset in plot_groups:

        subset = subset.sort_values("relative_day")

        if subset.empty:
            continue

        x = subset["relative_day"].to_numpy()
        mean = subset["mean"].to_numpy() * 100

        label = (
            metric.upper()
            if group_col is None
            else f"{group_name} (n={int(subset['n'].iloc[0])})"
        )

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label=label
        )

        if confidence:
            lower = subset["ci_lower"].to_numpy() * 100
            upper = subset["ci_upper"].to_numpy() * 100

            ax.fill_between(
                x,
                lower,
                upper,
                alpha=0.12
            )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Relative Trading Day")
    ax.set_ylabel(f"{metric.upper()} (%)")
    ax.set_xticks(sorted(df["relative_day"].unique()))

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight"
        )

    plt.show()