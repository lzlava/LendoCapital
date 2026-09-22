#File containing all past functions useful for future analysis.
import pandas as pd
import pyarrow
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def load_ohlcv(path):
    """
    This function reads a .parquet file into a the notebook as a dataset (df).
    Input: path (of file)
    Output: dataframe
    """
    
    df = pd.read_parquet(path)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    return df

def check_missing_data(df):
    """
    Check if any data is missing.

    Input: dataframe
    Output: dataframe of rows with missing values
    """

    missing = df.isna().sum()

    print("\nMissing values by column:")
    print(missing)

    problem_rows = df[df.isna().any(axis=1)]

    print("\nRows containing missing values: " + str(len(problem_rows)))
    print(problem_rows)

    return problem_rows
    
def check_duplicates(df):
    """
    Check if any duplicate rows exist.
    Note: This counts the number of rows that have a duplicate at all! so 2 rows of the same date --> 2 duplicate rows; 3 rows of the same date --> 3 duplicate rows

    Input: dataframe
    Output: dataframe of duplicate rows
    """

    duplicates = df[
        df.duplicated(
            subset=["ticker", "date"],
            keep=False
        )
    ]

    print("\nNumber of duplicate rows:")
    print(len(duplicates))

    print("\nDuplicate observations:")
    print(duplicates)

    return duplicates

def check_invalid_values(df):
    """
    Here we want to check if prices are invalid - notably, we will make sure that all prices >0, and volume =>0.
    We also must check that high is indeed high, and low is indeed low.

    Input: dataframe
    Output: dataframe of rows with invalid prices
    """

    price_cols = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close"
    ]
    
    invalid = df[
        (df[price_cols] <= 0).any(axis=1)
        |
        (df["volume"] < 0)
        |
        (df["high"] < df["open"])
        |
        (df["high"] < df["close"])
        |
        (df["high"] < df["low"])
        |
        (df["low"] > df["open"])
        |
        (df["low"] > df["close"])
    ]

    print("\nRows with invalid prices:")
    print(invalid)

    return invalid

def check_valid_date(df, benchmark = "SPY"):
    """
    This function checks that all the dates listed for each stock are actual days when trades are possible.
    I will use SPY as the benchmark (suggested by most online sources; plus, its S&P500) to see if other stocks dates match. But this is just the default and can be changed.

    Input: dataframe, str:ticker
    Output: dataframe of rows with erroneous dates
    """

    benchmark_dates = set(
        df[df["ticker"] == benchmark]["date"]
    )

    tickers = df["ticker"].unique()

    problems = {}

    for ticker in tickers:

        ticker_dates = set(
            df[df["ticker"] == ticker]["date"]
        )

        missing_dates = benchmark_dates - ticker_dates
        extra_dates = ticker_dates - benchmark_dates

        if missing_dates or extra_dates:
            problems[ticker] = {
                "missing_dates": missing_dates,
                "extra_dates": extra_dates
            }

    print("\nCalendar discrepancies:")
    if problems == {}:
        print("No discrepancies!")
    else:
        print(problems)

    return problems

def check_extreme_returns(df, threshold = 0.25):
    """
    This function checks for whether or not the simply daily return for any ticker at any moment is suspiciously high or low.

    Input: dataframe
    Output: dataframe of list of days with suspicious returns, for investigation.
    """
    extreme_returns = df[
        df["daily_simple_return"].abs() > threshold
    ]

    print("\nExtreme daily returns:")
    print(
        extreme_returns[
            ["date", "ticker", "adjusted_close", "daily_simple_return"]
        ]
    )

    return extreme_returns

def validate_ohlcv(df):
    """
    This function provides a series of checks for whether or not the data is clean & devoid of errors.

    Input: dataframe
    Output: relevant info on dataset, warnings for particular instances of values
    """

    #1. Overview: Check the big-scale information about the dataset, including duplicates
    # shape
    print("Shape")
    print(df.shape)
    
    # Overview: Column types
    print("\nColumn Types")
    print(df.dtypes)

    # basic numerical summary
    print("\nSummary")
    print(df.describe())

    # missing values in each column
    missing_rows = check_missing_data(df)
    
    # duplicate ticker-date observations
    duplicates = check_duplicates(df)

    # checking if values are valid
    invalid_values = check_invalid_values(df)

    # checking if dates are valid & complete
    check_valid_date(df)

    # check if any daily simple return is suspiciously high
    check_extreme_returns(df)

def compute_period_returns(df, frequency = "W-FRI"):
    """
    This is a subfunction for compute_returns that computes a certain frequency of returns based on input.

    Input: dataframe, str:frequency ("W-FRI", "ME", "YE")
    Output: dataframe cropped per period, with returns computed
    """
   
    period_prices = (
    df.set_index("date")
      .groupby("ticker")["adjusted_close"]
      .resample(frequency)
      .last()
      .dropna()
      .reset_index()
    )

    # Instead of compounding by day, AI suggested I compute simple returns through jumping from weekly endpoint to weekly endpoint. I also coded a simple test for if I used the compounding method and they do produce the same results (as we would intuitively expect)
    
    period_prices["simple_return"] = (
    period_prices.groupby("ticker")["adjusted_close"]
                 .pct_change(fill_method=None)
    )

    period_prices["log_return"] = (
    np.log(period_prices["adjusted_close"])
    - np.log(
        period_prices.groupby("ticker")["adjusted_close"].shift(1)
    )
    )

    return period_prices

def compute_returns(df):
    """
    This function computes the daily simple & log returns of each stock, putting them as new columns in a dataframe copy.
    Simple returns are the most intuitive as they calculate percent change in closing price from the previous day. These are generally used in discussion for clarity and ease of understanding.
    Log returns take the log-difference of the closing prices, which is less convenient to think about but is very useful in summing returns, as this metric has an sums well (whereas simple returns require a compound formula).

    Input: dataframe
    Output: dataframe with extra columns on daily returns, dataframe with all types of returns
    """

    df = df.copy()

    # Make sure observations are ordered correctly
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily simple return --> already done in this notebook but included here for future ones.
    df["daily_simple_return"] = (
        df.groupby("ticker")["adjusted_close"]
          .pct_change(fill_method=None)
    )

    # Daily log return
    df["daily_log_return"] = (
        np.log(df["adjusted_close"])
        - np.log(
            df.groupby("ticker")["adjusted_close"].shift(1)
        )
    )

    daily = df[
        ["date", "ticker", "daily_simple_return", "daily_log_return"]
    ].copy()
    
    daily = daily.rename(
        columns={
            "daily_simple_return": "simple_return",
            "daily_log_return": "log_return"
        }
    )

    daily["frequency"] = "daily"
    
    # Weekly
    weekly = compute_period_returns(df, "W-FRI")
    weekly["frequency"] = "weekly"

    # Monthly
    monthly = compute_period_returns(df, "ME")
    monthly["frequency"] = "monthly"

    # Annual (not required)
    annual = compute_period_returns(df, "YE")
    annual["frequency"] = "annual"

    # Check columns are consistent
    columns = [
    "date",
    "ticker",
    "simple_return",
    "log_return",
    "frequency"
    ]
    
    daily = daily[columns]
    weekly = weekly[columns]
    monthly = monthly[columns]
    annual = annual[columns]

    # Combine all these calculated values into one dataframe,
    returns_df = pd.concat(
        [daily, weekly, monthly, annual],
        ignore_index=True
    )

    return df, returns_df

def plot_return_distribution(
    returns_df,
    frequency,
    return_type="simple_return",
    bins=50,
    tickers=None
):
    """
    Plot the distribution of returns for a given frequency and return type.

    Input: returns_df (dataframe), frequency ("daily", "weekly", "monthly", "annual"),
           return_type ("simple_return" or "log_return"), bins,
           tickers (optional, list of tickers)
    Output: plot
    """

    data = returns_df[
        returns_df["frequency"] == frequency
    ].copy()

    # Filter by ticker(s) only when specified in argument (for selecting a few companies or a benchmark)
    if tickers is not None:
        if isinstance(tickers, str):
            tickers = [tickers]

        data = data[
            data["ticker"].isin(tickers)
        ]
    if tickers is None:
        ticker_title = "All Stocks"
    else:
        ticker_title = ", ".join(tickers)

    data = data[return_type].dropna()
    
    # So the range on the plot isn't very big
    lower = data.quantile(0.01)
    upper = data.quantile(0.99)

    plt.figure(figsize=(8, 5))

    plt.axvline(0, linestyle="--", color="k")

    plt.hist(
        data,
        bins=bins,
        range=(lower, upper)
    )

    plt.title(
        f"{ticker_title} - {frequency.capitalize()} "
        f"{return_type.replace('_', ' ').title()} Distribution"
    )

    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.show()