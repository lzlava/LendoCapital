import pandas as pd
import yfinance as yf
import requests
import time
import numpy as np
import matplotlib.pyplot as plt

# The first part is using SEC EDGAR to retrieve data, a couple of functions:
def get_cik(ticker):
    """
    Find SEC CIK for a ticker.
    """

    match = ticker_df[
        ticker_df["ticker"] == ticker
    ]

    if match.empty:
        return None

    return str(
        match.iloc[0]["cik_str"]
    ).zfill(10)

get_cik("AAPL")

def get_book_equity_sec(ticker, start_date):
    """
    Pull historical stockholders' equity from SEC EDGAR.

    Input:
        ticker: stock ticker (str)
        start_date: earliest fiscal period end to include (str)

    Output:
        dataFrame containing:
        ticker
        fiscal_date
        filed_date
        book_equity
        form
    """

    cik = get_cik(ticker)

    if cik is None:
        print(f"No CIK found for {ticker}")
        return None

    url = (
        f"https://data.sec.gov/api/xbrl/companyfacts/"
        f"CIK{cik}.json"
    )

    response = requests.get(
        url,
        headers=headers
    )

    data = response.json()

    facts = data["facts"]["us-gaap"]

    equity_concepts = [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
    ]
    
    equity_concept = None
    most_observations = 0
    
    for concept in equity_concepts:
    
        if concept in facts:
    
            units = facts[concept].get("units", {})
    
            if "USD" in units:
    
                observations = units["USD"]
    
                if len(observations) > most_observations:
                    most_observations = len(observations)
                    equity_concept = concept
    
    if equity_concept is None:
        print(f"No usable equity concept found for {ticker}")
        return None
    
    equity_data = facts[equity_concept]["units"]["USD"]
    
    # Turn the SEC equity observations into a dataframe
    df = pd.DataFrame(equity_data)
    
    # Add which SEC equity concept we used
    df["equity_concept"] = equity_concept
    
    df = df[
        df["form"].isin(["10-K", "10-Q"])
    ].copy()
    
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    
    df = df[
        df["end"] >= pd.to_datetime(start_date)
    ]
    
    df = df.rename(columns={
        "end": "fiscal_date",
        "filed": "filed_date",
        "val": "book_equity"
    })
    
    df["ticker"] = ticker
    
    df = df[
        [
            "ticker",
            "fiscal_date",
            "filed_date",
            "book_equity",
            "form",
            "equity_concept"
        ]
    ]
    
    # Sort by fiscal period first, then by when it was filed
    df = df.sort_values(
        ["fiscal_date", "filed_date"]
    )
    
    # If the same fiscal period appears in multiple later filings,
    # keep the first time that information was reported
    df = df.drop_duplicates(
        subset=["ticker", "fiscal_date"],
        keep="first"
    )
    
    df = df.reset_index(drop=True)
    
    return df

def get_all_book_equity_sec(tickers, start_date):
    """
    Gets all the book equity information from the list of tickers.

    Input:
        tickers: list of all tickers used (list)
        start_date: start date of data grab (str)
    """

    all_book_equity = []

    for ticker in tickers:

        try:
            ticker_df = get_book_equity_sec(
                ticker,
                start_date
            )

            if ticker_df is not None:
                all_book_equity.append(ticker_df)
                #print(f"Completed: {ticker}")

            else:
                print(f"Skipped: {ticker}")

        except Exception as e:
        #    print(f"Error for {ticker}: {e}")

        # To not exceed data grab limit
            time.sleep(0.15)

    if len(all_book_equity) == 0:
        return pd.DataFrame()

    return pd.concat(
        all_book_equity,
        ignore_index=True
    )

def build_value_df(daily_combined_df, book_equity_df):
    """
    Construct a dataframe for value-factor analysis.

    Inputs:
        daily_combined_df:
            Daily stock data containing at least:
            ticker, date, close, simple_return, log_return,
            shares_outstanding

        book_equity_df:
            SEC book equity data containing at least:
            ticker, filed_date, book_equity

    Output:
        value_df:
            DataFrame containing:
            ticker, date, close, simple_return, log_return,
            shares_outstanding, market_cap,
            filed_date, book_equity, book_to_market
    """

    # Keep only columns relevant for value-factor research
    market_df = daily_combined_df[
        [
            "ticker",
            "date",
            "close",
            "simple_return",
            "log_return",
            "shares_outstanding"
        ]
    ].copy()

    # Calculate historical market capitalization
    market_df["market_cap"] = (
        market_df["close"]
        * market_df["shares_outstanding"]
    )

    # Keep only the book-equity information we need
    book_df = book_equity_df[
        [
            "ticker",
            "filed_date",
            "book_equity"
        ]
    ].copy()

    # Make sure date columns are datetime
    market_df["date"] = pd.to_datetime(
        market_df["date"]
    )

    book_df["filed_date"] = pd.to_datetime(
        book_df["filed_date"]
    )

    # Sort before using merge_asof
    market_df = market_df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)
    
    book_df = book_df.sort_values(
        ["filed_date", "ticker"]
    ).reset_index(drop=True)

    # Match each market date with the latest book equity
    # that had already been publicly filed
    value_df = pd.merge_asof(
        market_df,
        book_df,
        left_on="date",
        right_on="filed_date",
        by="ticker",
        direction="backward"
    )

    # Construct book-to-market value signal
    value_df["book_to_market"] = (
        value_df["book_equity"]
        / value_df["market_cap"]
    )

    return value_df.reset_index(drop=True)

def rank_month_end(
    df,
    value_column,
    ticker_column="ticker",
    date_column="date",
    simple_return_column="simple_return",
    log_return_column="log_return",
    ascending=True,
    rank_column="rank"
):
    """
    Rank stocks at the end of each month based on a specified column.

    Inputs:
        df: dataframe
        value_column: Column to rank, e.g. "book_to_market", "momentum", "market_cap" (str)
        ticker_column: Column containing ticker names
        date_column: Column containing dates
        simple_return_column: name of column to compute monthly returns
        log_return_column: name of column to compute monthly returns
        ascending: True  -> smallest value gets rank 1; False -> largest value gets rank 1
        rank_column: Name of the new ranking column

    Output:
        dataFrame containing one observation per ticker per month,with an additional ranking column.
    """

    ranked_df = df.copy()

    # -----------------------------------
    # CASE 1: dataframe contains daily data
    # -----------------------------------
    if date_column in ranked_df.columns:

        ranked_df[date_column] = pd.to_datetime(
            ranked_df[date_column]
        )

        ranked_df["month"] = (
            ranked_df[date_column].dt.to_period("M")
        )

        # Monthly simple returns
        monthly_simple = (
            ranked_df
            .groupby([ticker_column, "month"])[simple_return_column]
            .apply(lambda x: (1 + x).prod() - 1)
            .rename("monthly_simple_return")
            .reset_index()
        )

        # Monthly log returns
        monthly_log = (
            ranked_df
            .groupby([ticker_column, "month"])[log_return_column]
            .sum()
            .rename("monthly_log_return")
            .reset_index()
        )

        # Keep month-end observation
        ranked_df = (
            ranked_df
            .sort_values([ticker_column, date_column])
            .groupby([ticker_column, "month"])
            .tail(1)
            .copy()
        )

        ranked_df = ranked_df.merge(
            monthly_simple,
            on=[ticker_column, "month"],
            how="left"
        )

        ranked_df = ranked_df.merge(
            monthly_log,
            on=[ticker_column, "month"],
            how="left"
        )

    # -----------------------------------
    # CASE 2: dataframe is already monthly
    # -----------------------------------
    elif "month" in ranked_df.columns:

        # Make sure month has the same Period format
        ranked_df["month"] = pd.PeriodIndex(
            ranked_df["month"],
            freq="M"
        )

    else:
        raise ValueError(
            "DataFrame must contain either a 'date' column or a 'month' column."
        )

    # -----------------------------------
    # Rank stocks within each month
    # -----------------------------------
    ranked_df[rank_column] = (
        ranked_df
        .groupby("month")[value_column]
        .rank(
            method="first",
            ascending=ascending
        )
    )

    return ranked_df.reset_index(drop=True)

def add_momentum(df, return_col="simple_return"):
    """
    Calculate 12-1 momentum for each stock.

    Momentum at month t = cumulative return over months t-12 through t-1.

    Input:
        df: dataframe, daily_combined_df and daily_returns_df both should work.
        return_col: daily simple return column; since I decided we'd use the simple return for this, I didn't code up the log return part yet.

    Output:
        monthly dataframe with momentum column
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Create month identifier
    df["month"] = df["date"].dt.to_period("M")

    # Convert daily simple returns into monthly simple returns
    monthly_df = (
        df.groupby(["ticker", "month"])[return_col]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index(name="monthly_simple_return")
    )

    monthly_df = monthly_df.sort_values(
        ["ticker", "month"]
    ).reset_index(drop=True)

    # Calculate t-12 through t-1 cumulative return
    monthly_df["momentum"] = (
        monthly_df.groupby("ticker")["monthly_simple_return"]
        .transform(
            lambda x: (1 + x.shift(1)) # Shift by 1
            .rolling(window=12, min_periods=12)
            .apply(np.prod, raw=True) - 1
        )
    )

    monthly_df = monthly_df.dropna(subset=["momentum"])
    
    return monthly_df

def plot_cross_sectional_distribution(
    df,
    month,
    value_column,
    bins=10
):
    """
    This function plots the cross-sectional distribution of a factor across stocks for a specified month.

    Inputs:
        df: dataframe (monthly, such as monthly_momentum_df or monthly_value_df)
        month: month to examine, e.g. "2025-12" (str)
        value_column: factor column to plot, "book_to_market" or "momentum"
        bins: number of histogram bins

    Output:
        histogram of factor values across stocks
    """

    # Select specified month
    month_data = df[
        df["month"] == month
    ][value_column].dropna()

    # Plot distribution
    plt.figure(figsize=(8, 5))

    plt.hist(
        month_data,
        bins=bins
    )

    # Add average factor value
    mean_value = month_data.mean()

    plt.axvline(
        mean_value,
        linestyle="--",
        color="black",
        label=f"Mean = {mean_value:.3f}"
    )

    plt.xlabel(
        value_column.replace("_", " ").title()
    )
    plt.ylabel("Number of Stocks")

    plt.title(
        f"{value_column.replace('_', ' ').title()} Distribution — {month}"
    )

    plt.legend()
    plt.show()

def assign_quintiles(
    df,
    factor_column,
    rank_column,
    return_column="monthly_simple_return"
):
    """
    Assign stocks to five quintiles within each month
    based on their existing factor rank.
    Here: rank 1 = strongest factor

    Inputs:
        df: monthly factor dataframe
        factor_column: e.g. "momentum" or "book_to_market"
        rank_column: e.g. "momentum_rank" or "value_rank"
        return_column: monthly return column

    Output:
        dataframe containing:
        ticker, month, monthly_return, factor, rank, quintile
    """
    quintile_df = df.copy()

    # Sort chronologically within each stock
    quintile_df = quintile_df.sort_values(
        ["ticker", "month"]
    )

    # Get each stock's return in the following month
    quintile_df["next_month_return"] = (
        quintile_df
        .groupby("ticker")[return_column]
        .shift(-1)
    )

    # Keep necessary columns
    quintile_df = quintile_df[
        [
            "ticker",
            "month",
            return_column,
            "next_month_return",
            factor_column,
            rank_column
        ]
    ].copy()

    # Assign quintiles within each month
    quintile_df["quintile"] = (
        quintile_df
        .groupby("month")[rank_column]
        .transform(
            lambda x: pd.qcut(
                x,
                q=5,
                labels=[1, 2, 3, 4, 5]
            )
        )
    )

    return quintile_df

def compute_factor_returns(
    quintile_df,
    return_column="monthly_return",
    factor_return_column="factor_return"
):
    """
    Compute monthly factor returns as Q5 average return minus Q1 average return.

    Inputs:
        quintile_df: dataframe containing ticker, month, monthly returns, and quintile at minimum
        return_column: column containing each stock's monthly return (str)
        factor_return_column: (decided here) chosen name of output factor return column (str)

    Output:
        dataframe containing two columns: month, factor_return
    """
    # Average return of each quintile each month
    quintile_returns = (
        quintile_df
        .groupby(
            ["month", "quintile"],
            observed=True
        )[return_column]
        .mean()
        .reset_index()
    )

    # Put Q1-Q5 into separate columns
    quintile_returns = quintile_returns.pivot(
        index="month",
        columns="quintile",
        values=return_column
    )

    # Factor return = Q5 - Q1
    quintile_returns[factor_return_column] = (
        quintile_returns[5] - quintile_returns[1]
    )

    # Keep only month and factor return
    factor_returns = (
        quintile_returns[[factor_return_column]]
        .reset_index()
    )

    return factor_returns

def add_cumulative_returns(
    df,
    return_column,
    cumulative_column="cumulative_return"
):
    """
    Simple function to add cumulative returns by compounding simple returns over time.

    Inputs:
        df: dataframe, such as
        return_column: column containing simple returns
        cumulative_column: name of new cumulative return column (str)

    Output:
        dataframe with cumulative return column added
    """

    result = df.copy()

    result = result.sort_values("month").reset_index(drop=True)

    result[cumulative_column] = (
        (1 + result[return_column]).cumprod() - 1
    )

    return result

def plot_cumulative_returns(
    df,
    cumulative_column,
    title="Cumulative Factor Returns"
):
    """
    This function plots cumulative returns over time.

    Inputs:
        df: dataframe, value_factor_returns or momentum_factor_returns
        cumulative_column: cumulative return column (str)
        title: plot title (str)
    """

    plot_df = df.copy()

    # Convert Period month into timestamp for plotting
    plot_df["date"] = plot_df["month"].dt.to_timestamp()

    plt.figure(figsize=(10, 6))

    plt.plot(
        plot_df["date"],
        plot_df[cumulative_column]
    )

    plt.axhline(
        0,
        linestyle="--",
        color="black"
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(title)

    plt.show()

def plot_quintile_cumulative_returns(
    quintile_df,
    return_column="next_month_return",
    title="Cumulative Returns by Quintile"
):
    """
    Calculate, then plot cumulative returns for Q1-Q5, assuming equal-weight portfolios

    Inputs:
        quintile_df: dataframe containing month, quintile, and next-month returns
        return_column: column containing holding-period returns (str)
        title: plot title (str)

    Output:
        dataframe containing monthly and cumulative returns for each quintile
        plots
    """

    # Average stock return within each quintile each month
    quintile_returns = (
        quintile_df
        .groupby(
            ["month", "quintile"],
            observed=True
        )[return_column]
        .mean()
        .reset_index(name="quintile_return")
    )

    # Sort chronologically
    quintile_returns = quintile_returns.sort_values(
        ["quintile", "month"]
    )

    # Compound returns separately for each quintile
    quintile_returns["cumulative_return"] = (
        quintile_returns
        .groupby(
            "quintile",
            observed=True
        )["quintile_return"]
        .transform(
            lambda x: (1 + x).cumprod() - 1
        )
    )

    # Plot
    plt.figure(figsize=(10, 6))

    colors = {
    1: "red",
    2: "orange",
    3: "gray",
    4: "lightblue",
    5: "blue"
    }

    for q in range(1, 6):

        data = quintile_returns[
            quintile_returns["quintile"] == q
        ]

        plt.plot(
        (data["month"] + 1).dt.to_timestamp(),
        data["cumulative_return"],
        label=f"Q{q}",
        color=colors[q]
        )

    plt.axhline(
        0,
        linestyle="--",
        color="black"
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(title)
    plt.legend()

    plt.show()

    return quintile_returns

def factor_statistics(
    df,
    return_column,
    risk_free_rate=0
):
    """
    Compute performance statistics for monthly factor returns.

    Inputs:
        df: dataframe containing monthly factor returns
        return_column: column containing factor returns
        risk_free_rate: annual risk-free rate, default = 0

    Output:
        annualized return
        annualized volatility
        Sharpe ratio
        maximum drawdown
        hit rate
    """

    returns = df[return_column].dropna()

    # Number of months
    n_months = len(returns)

    # Annualized return
    annualized_return = (
        (1 + returns).prod() ** (12 / n_months)
        - 1
    )

    # Annualized volatility
    annualized_volatility = (
        returns.std() * np.sqrt(12)
    )

    # Sharpe ratio
    sharpe_ratio = (
        (annualized_return - risk_free_rate)
        / annualized_volatility
    )

    # Cumulative wealth
    wealth = (1 + returns).cumprod()

    # Previous highest wealth level
    running_peak = wealth.cummax()

    # Drawdown at each point
    drawdown = (
        wealth / running_peak - 1
    )

    # Worst drawdown
    max_drawdown = drawdown.min()

    # Percentage of months where factor return > 0
    hit_rate = (
        (returns > 0).mean()
    )

    return pd.Series({
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Hit Rate": hit_rate
    })

def quintile_statistics(
    quintile_returns_df,
    risk_free_rate=0
):
    """
    Compute performance statistics separately for each quintile.

    Input:
        quintile_returns_df: dataframe containing
                             month, quintile, quintile_return
        risk_free_rate: annual risk-free rate

    Output:
        dataframe of statistics for Q1-Q5
    """

    results = []

    for q in range(1, 6):

        returns = (
            quintile_returns_df[
                quintile_returns_df["quintile"] == q
            ]["quintile_return"]
            .dropna()
        )

        n_months = len(returns)

        # Annualized return
        annualized_return = (
            (1 + returns).prod() ** (12 / n_months)
            - 1
        )

        # Annualized volatility
        annualized_volatility = (
            returns.std() * np.sqrt(12)
        )

        # Sharpe ratio
        sharpe_ratio = (
            (annualized_return - risk_free_rate)
            / annualized_volatility
        )

        # Wealth over time
        wealth = (1 + returns).cumprod()

        # Running peak
        running_peak = wealth.cummax()

        # Drawdowns
        drawdown = wealth / running_peak - 1

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Hit rate
        hit_rate = (returns > 0).mean()

        results.append({
            "quintile": f"Q{q}",
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "hit_rate": hit_rate
        })

    return pd.DataFrame(results)

