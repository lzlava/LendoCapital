#File containing all past functions useful for future analysis.
import pandas as pd
import pyarrow
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def load_ohlcv(path):
    """
    This function reads a .parquet file into a the notebook as a dataset (df).
    
    Input:
        path: file path on computer
        
    Output:
        dataframe: containing stock data
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

    Input:
        df: dataframe
        
    Output:
        dataframe of rows with missing values
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

    Input:
        df: dataframe
        
    Output:
        dataframe of duplicate rows
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

    Input:
        df: dataframe
        
    Output:
        dataframe of rows with invalid prices
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

    Input:
        df: dataframe
        benchmark: ticker (str)

    Output:
        dataframe of rows with erroneous dates
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

    Input:
        df: dataframe
        threshold: what counts as extreme (fl)
        
    Output:
        dataframe of list of days with suspicious returns, for investigation.
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

    Input:
        df: dataframe

    Output:
        relevant info on dataset, warnings for particular instances of values
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

    Input:
        df: dataframe
        frequency: timeframe from ["W-FRI", "ME", "YE"] (str)
        
    Output:
        dataframe cropped per period, with returns computed
    """
   
    period_prices = (
    df.set_index("date")
      .groupby("ticker")[["close", "adjusted_close"]]
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

    Input:
        df: dataframe
        
    Output:
        dataframe with extra columns on daily returns, dataframe with all types of returns
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
        [
            "date",
            "ticker",
            "close",
            "adjusted_close",
            "daily_simple_return",
            "daily_log_return"
        ]
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
        "close",
        "adjusted_close",
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

    Input:
        returns_df: dataframe
        frequency: how often the data was sampled, from ["daily", "weekly", "monthly", "annual"] (str)
        return_type: "simple_return" or "log_return" (str)
        bins: bin size (int)
        tickers: optional, list of tickers to track in specific (list or str)
           
    Output:
        plot
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

    plt.axvline(
        0,
        linestyle="--",
        color="black",
        label="Zero Return"
    )
    
    average_return = data.mean()

    plt.axvline(
        average_return,
        linestyle="--",
        color="red",
        label=f"Average Return ({average_return:.4f})"
    )

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

    #Add some extra features

    plt.legend()

    plt.show()

def rolling_statistics(
    returns_df,
    window=20,
    return_type="simple_return"
):
    """
    This function computes the relevant rolling statistics for the returns dataframe, given the rolling window.

    Input:
        returns_df: dataframe (returns)
        window: sample window (int)
        return_type: "simple_return" or "log_return" (str)
        
    Output:
        dataframe (returns) with added columns on rolling mean, volatility, skewness, kurtosis
    """

    if return_type == "simple_return":
        heading = "_simple"
    if return_type == "log_return":
        heading = "_log"
    
    df = returns_df.copy()
    df = df.sort_values(["ticker", "date"])
    grouped = df.groupby("ticker")[return_type]

    # Mean
    df["rolling_mean"+heading] = grouped.transform(
        lambda x: x.rolling(window).mean()
    )

    # Volatility (stddev)
    df["rolling_volatility"+heading] = grouped.transform(
        lambda x: x.rolling(window).std()
    )

    # Skewness
    df["rolling_skewness"+heading] = grouped.transform(
        lambda x: x.rolling(window).skew()
    )

    # Kurtosis
    df["rolling_kurtosis"+heading] = grouped.transform(
        lambda x: x.rolling(window).kurt()
    )

    return df

def build_equal_weight_portfolio(
    returns_df,
    return_type="simple_return",
    benchmark_ticker="SPY"
):
    """
    Simple function that calculates the returns of an equal-weight portfolio

    Input:
        returns_df: dataframe (returns_df type document, such as "weekly_returns")
        return_type: "simple_return" or "log_return" (str)
        benchmark_ticker: "SPY" (str)
        
    Output:
        dataframe (new dataframe, of equal-weight returns each date
    """

    portfolio_df = returns_df[
        returns_df["ticker"] != benchmark_ticker
    ].copy()
    
    portfolio = (
        returns_df
        .groupby("date")[return_type]
        .mean()
        .reset_index()
    )

    portfolio = portfolio.rename(
        columns={return_type: "ew_portfolio_return"}
    )

    return portfolio

def get_shares_outstanding(ticker, start_date=start_date):
    """
    This function pulls historical shares outstanding for a given ticker.

    Input:
        ticker: the name of the ticker, e.g., "AAPL" (str)

    Output:
        dataframe with ticker, date, and shares_outstanding
    """
    stock = yf.Ticker(ticker)

    shares = stock.get_shares_full(
        start=start_date
    )

    if shares is None:
        print(f"No shares outstanding data found for {ticker}")
        return None

    shares_df = shares.reset_index()

    shares_df.columns = [
        "date",
        "shares_outstanding"
    ]

    shares_df["ticker"] = ticker

    return shares_df

def get_all_shares_outstanding(tickers):
    """
    Once we have the individual shares outstanding data for each company, we need to combine these into one dataframe.

    Input:
        tickers: exhaustive list of tickers

    Output:
        dataframe complete with date, tickers and shares outstanding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    o        ooooooooooodoooooodoododooooodooododooooddooodooooooodoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooojij
    """

    shares_list = []

    # Combine into one big dataframe using above function
    for ticker in tickers:
        shares_df = get_shares_outstanding(ticker)
        shares_list.append(shares_df)

    all_shares_df = pd.concat(
        shares_list,
        ignore_index=True
    )

    all_shares_df = all_shares_df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    return all_shares_df

def merge_shares_with_returns(
    returns_df,
    shares_df,
    benchmark_ticker="SPY"
):
    """
    Merge historical shares outstanding data into the returns_df dataframe for calculating value-weight portfolio.
    The benchmark ticker is excluded from the portfolio.
    For each stock/date, the most recent available shares outstanding
    observation is used.

    Input:
        returns_df: dataframe of returns, with close price. Should be "daily_returns", "weekly_returns", etc.
                    Do not use the entire returns_df.
        shares_df: dataframe of shares outstanding for each stock
        benchmark_ticker: "SPY"

    Output:
        dataframe, merged
    """
    
    portfolio_df = returns_df[
        returns_df["ticker"] != benchmark_ticker
    ].copy()

    shares_df = shares_df.copy()

    # Check timezones/ date convention is consistent
    shares_df["date"] = (
        pd.to_datetime(shares_df["date"])
          .dt.tz_localize(None)
    )

    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])

    # Make sure dates are datetime
    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
    shares_df["date"] = pd.to_datetime(shares_df["date"])

    # Required for merge_asof
    portfolio_df = portfolio_df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    shares_df = shares_df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    combined_df = pd.merge_asof(
        portfolio_df,
        shares_df,
        on="date",
        by="ticker",
        direction="backward"
    )

    return combined_df

def build_value_weight_portfolio(
    combined_df,
    return_column="simple_return",
    benchmark_ticker="SPY"
):
    """
    Function that calculates the value-weight portfolio.
    Value-weights are selected by calculating the most-recently known market cap (fromshares outstanding & close) at that time.

    Inputs:
        combined_df: dataframe with relevant columns from previous functions. Shouldn't come from the whole returns_df
        return_type: column of return - "simple_return" or "log_return" (str)
        benchmark_ticker="SPY"
    """

    df = combined_df.copy()

    # Calculate market capitalization
    df["market_cap"] = (
        df["close"] * df["shares_outstanding"]
    )

    # Calculate each stock's weight within each date
    df["value_weight"] = (
        df["market_cap"]
        / df.groupby(
            ["date"]
        )["market_cap"].transform("sum")
    )

    # Use the previous period's weight
    df["portfolio_weight"] = (
        df.groupby(
            ["ticker"]
        )["value_weight"].shift(1)
    )

    # Calculate each stock's contribution to portfolio return
    df["weighted_return"] = (
        df["portfolio_weight"] * df[return_column]
    )

    # Sum contributions to get portfolio return
    val_portfolio_returns = (
        df.groupby(
            ["date"]
        )["weighted_return"]
        .sum()
        .reset_index(name="val_portfolio_return")
    )

    return df, val_portfolio_returns    

def portfolio_performance(
    portfolio,
    frequency="daily",
    return_type="simple_return",
    return_column="ew_portfolio_return",
    risk_free_rate=0
):
    """
    Compute portfolio performance statistics.

    Input:
        portfolio: dataframe containing portfolio returns (e.g., ew_daily_portfolio)
        frequency: "daily", "weekly", "monthly", or "annual"
        return_type: "simple_return" or "log_return"
        return_column: name of returns column, e.g. "ew_portfolio_return" or "val_portfolio_return"
        risk_free_rate: annual risk-free rate

    Output:
        annualized return, annualized volatility, Sharpe ratio, and maximum drawdown
    """

    returns = portfolio[return_column].dropna()

    # Because we might want to insert different portfolios samples (sampled daily vs weekly, for example), we need to calculate total periods per year for subsequent calcs.
    periods = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "annual": 1
    }

    periods_per_year = periods[frequency]

    if return_type == "simple_return":

        # Annualized return, summing up portfolio performance (avg per year) over the entire timeframe
        # Compounded if simple
        annualized_return = (
            (1 + returns).prod()
            ** (periods_per_year / len(returns))
            - 1
        )

        # Cumulative wealth
        cumulative = (1 + returns).cumprod()

    elif return_type == "log_return":

        # Log returns are easier since we just add the returns and then take the exp to get rid of the log
        annualized_return = (
            np.exp(returns.mean() * periods_per_year)
            - 1
        )

        # Cumulative wealth
        cumulative = np.exp(returns.cumsum())

    # Annualized volatility: a standard deviation measurement = stddev_daily * sqrt(timeframe)
    annualized_volatility = (
        returns.std() * np.sqrt(periods_per_year)
    )

    # Sharpe ratio: measures performance of an investment relative to a risk-free asset, after adjusting for risk.
    sharpe_ratio = (
        (annualized_return - risk_free_rate) # Essentially return difference in comparison to what you could've gotten with investing into something without risk
        / annualized_volatility # Accounting for volatility so more risky investments --> poorer Sharpe ratio
    )

    # Maximum drawdown: largest peak to trough loss in %
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

def plot_cumulative_returns(
    portfolio,
    return_type="simple_return",
    return_column="ew_portfolio_return",
    frequency="daily"
):
    """
    This function generates a plot of cumulative portfolio returns.

    Input:
        portfolio: dataframe containing date and portfolio returns, e.g., ew_simple_daily_portfolio
        return_type: "simple_return" or "log_return" (str)
        return_column: name of the column containing returns, e.g., "ew_portfolio_return" (str)
        frequency: "daily", "weekly", etc. Just for the axis label (str)

    Output:
        plot of cumulative returns
    """

    df = portfolio.copy()
    returns = df[return_column].fillna(0)

    if return_type == "simple_return":
        df["cumulative_return"] = (
            (1 + returns).cumprod() - 1
        )

    elif return_type == "log_return":
        df["cumulative_return"] = (
            np.exp(returns.cumsum()) - 1
        )

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["date"],
        df["cumulative_return"]
    )

    plt.axhline(
        0,
        linestyle="--",
        color="k",
        label="0% Return"
    )

    plt.title(f"'{return_column}' Portfolio Cumulative Return")
    plt.xlabel(f"Date (sampled {frequency})")
    plt.ylabel("Cumulative Return") # Units in decimal, or %/100. So 1.0 means +100% in returns

    plt.legend()
    plt.show()

def plot_rolling_volatility(
    portfolio,
    frequency="daily",
    return_column="ew_portfolio_return",
    window=20
):
    """
    This function plots annualized rolling portfolio volatility.
    I found this not too intuitive, but we are indeed calculating an annualized volatility using a sample window.
    We essentially determine a value for volatility expressed on an annual scale (sqrt(252)) given the behavior of the portfolio in the sample window

    Input:
        portfolio: dataframe containing date and portfolio returns
        frequency: "daily", "weekly", "monthly", or "annual" (str)
        return_column: name of the column containing returns, e.g., "ew_portfolio_return" (str)
        window: number of periods in rolling window (int)

    Output:
        plot of annualized rolling volatility
    """

    df = portfolio.copy()
    returns = df[return_column]

    periods = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "annual": 1
    }

    periods_per_year = periods[frequency]

    df["rolling_volatility"] = (
        returns
        .rolling(window)
        .std()
        * np.sqrt(periods_per_year)
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["date"],
        df["rolling_volatility"]
    )

    plt.title(
        f"{window}-Period, Rolling Annualized Volatility"
    )

    plt.xlabel(f"Date (sampled {frequency})")
    plt.ylabel("Annualized Volatility") # Units also in decimal, think normalized standard deviation. So 0.30 = 30% annualized volatility.

    plt.show()

# Above is all functions defined from week 1.