import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from matplotlib.ticker import PercentFormatter
from adjustText import adjust_text
from src.factor_utils import quintile_statistics
from src.factor_utils import factor_statistics
from src.data_utils import load_ohlcv
from src.data_utils import compute_returns
from src.factor_utils import add_cumulative_returns

def compute_risk_metrics(
    df,
    return_column,
    portfolio_type,
    risk_free_rate=0 
):
    """
    Computes relevant risk metrics for a given dataset with a list of monthly simple returns.
    Can be implemented on both per-quintile portfolios as well as long-short factor portfolios.
    The function has two modes:
        1) if portfolio is a quintile portfolio of stocks, it aggregates different stock data first before computing statistics
        2) if portfolio is the long-short factor return data it computes statistics directly.

    Input:
        df: dataframe - either momentum/value_quintiles_dfs[i] or momentum/value_factor_returns
        return_column: name of return column (str)
        portfolio_type: "quintile" or "factor" (str)
        risk_free_rate: Annual risk-free rate, default = 0 (fl)
    Output:
        pd.series of relevant metric results for the given dataframe
    """

    # --------------------------------------------------
    # 1. Get one monthly return series for the portfolio
    # --------------------------------------------------

    if portfolio_type == "quintile":

        # Assuming each row represents one stock in one month,
        # take the equal-weight average return across stocks.
        returns = (
            df.groupby("month")[return_column]
              .mean()
              .dropna()
        )

    elif portfolio_type == "factor":

        # Factor dataframe already has one return per month
        returns = df[return_column].dropna()

    else:
        raise ValueError(
            "portfolio_type must be 'quintile' or 'factor'"
        )

    # --------------------------------------------------
    # 2. Basic information
    # --------------------------------------------------

    num_months = len(returns)

    # --------------------------------------------------
    # 3. Annualized return
    # --------------------------------------------------

    total_growth = (1 + returns).prod()

    annualized_return = (
        total_growth ** (12 / num_months) - 1
    )

    # --------------------------------------------------
    # 4. Annualized volatility
    # --------------------------------------------------

    monthly_volatility = returns.std()

    annualized_volatility = (
        monthly_volatility * np.sqrt(12)
    )

    # Building on top of these, the other metrics:

    # Skew and Kurt
    skewness = returns.skew()
    kurtosis = returns.kurt()

    # Sharpe, Sortino, Calmar ratios, max drawdown, hit rate,

    sharpe_ratio = (
    (annualized_return - risk_free_rate) / annualized_volatility
    )

    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(12)
    sortino_ratio = (
        (annualized_return - risk_free_rate)
        / downside_deviation
    )

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    calmar_ratio = (
        annualized_return / abs(max_drawdown)
    )

    hit_rate = (returns > 0).mean()

    # And return everything:
    metrics = pd.Series({
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "hit_rate": hit_rate
    })

    return metrics

def stock_correlation_matrix(
    df,
    return_column="monthly_simple_return",
    ticker_column="ticker",
    date_column="month",
    tickers=None
):
    """
    Computes a stock-level correlation matrix using returns over time (months).

    Input:
        df: dataframe containing stock returns, needs ticker, date, returns (e.g., momentum_quintiles actually suffices)
        return_column: column containing returns used to calculate correlation.
        ticker_column: column identifying each stock.
        date_column: column identifying the return period.

    Output:
        dataframe containing the correlation matrix.
    """

    # Optionally select certain stocks
    if tickers is not None:
        df = df[df[ticker_column].isin(tickers)]
    
    # Reshape from long format:
    # one row per month, one column per stock
    returns_wide = df.pivot(
        index=date_column,
        columns=ticker_column,
        values=return_column
    )

    # Calculate correlations between every pair of stocks
    correlation_matrix = returns_wide.corr()

    returns_wide = df.pivot(
        index=date_column,
        columns=ticker_column,
        values=return_column
    )

    correlation_matrix = returns_wide.corr()

    return correlation_matrix

def portfolio_correlation_matrix(
    df,
    return_column="monthly_simple_return",
    date_column="month",
    portfolio_column="quintile"
):
    """
    Computes a portfolio-level correlation matrix.

    For each month, the function first calculates the equal-weight
    return of each portfolio/quintile. It then computes correlations
    between the portfolio return series over the full sample period.

    Input:
        df: dataframe containing individual stock returns and portfolio/quintile
        return_column: column containing returns used to calculate correlation.
        date_column: column identifying the return period.
        portfolio_column: column identifying portfolio membership, e.g. "quintile".

    Output:
        dataframe containing portfolio correlations.
    """

    # 1. Calculate one portfolio return for each quintile each month
    portfolio_returns = (
        df.groupby([date_column, portfolio_column])[return_column]
          .mean()
          .reset_index()
    )

    # 2. Convert to wide format:
    # one row per month, one column per quintile
    portfolio_returns_wide = portfolio_returns.pivot(
        index=date_column,
        columns=portfolio_column,
        values=return_column
    )

    # 3. Correlation between quintile portfolios
    correlation_matrix = portfolio_returns_wide.corr()

    return correlation_matrix

def calculate_benchmark_metrics(
    portfolio_df,
    benchmark_df,
    portfolio_return_column,
    benchmark_return_column="benchmark_return",
    date_column="month",
    periods_per_year=12
):
    """
    Calculates benchmark-relative portfolio statistics.

    Input:
        portfolio_df: dataframe of portfolio
        benchmark_df: dataframe of benchmark stock, usually SPY
        portfolio_return_column: return column name (str)
        benchmark_return_column: benchmark return column name (str)
        date_column: date column name (str)
        periods_per_year: for monthly, 12 (int)
    
    Output:
        beta
        annualized tracking error
        information ratio
    """

    # Align portfolio and benchmark by period
    merged = pd.merge(
        portfolio_df[[date_column, portfolio_return_column]],
        benchmark_df[[date_column, benchmark_return_column]],
        on=date_column,
        how="inner"
    ).dropna()

    portfolio_returns = merged[portfolio_return_column]
    benchmark_returns = merged[benchmark_return_column]

    # Beta
    beta = (
        portfolio_returns.cov(benchmark_returns)
        / benchmark_returns.var()
    )

    # Active returns
    active_returns = portfolio_returns - benchmark_returns

    # Annualized tracking error
    tracking_error = (
        active_returns.std()
        * np.sqrt(periods_per_year)
    )

    # Annualized information ratio
    information_ratio = (
        active_returns.mean()
        / active_returns.std()
        * np.sqrt(periods_per_year)
    )

    return pd.Series({
        "beta": beta,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio
    })

def diversification_curve(
    df,
    return_column="monthly_simple_return",
    ticker_column="ticker",
    date_column="month",
    n_simulations=50,
    periods_per_year=12
):
    """
    Computes a diversification curve showing how average portfolio volatility changes as the number of stocks in an equal-weight portfolio increases. Doesn't include the graphing part.

    For each portfolio size:
        1. Randomly sample stocks
        2. Form an equal-weight portfolio
        3. Compute annualized volatility
        4. Repeat many times
        5. Average the resulting volatilities

    Inputs:
        df: dataframe containing stock returns; momentum_quintiles works
        return_column: column containing periodic stock returns (str)
        ticker_column: column identifying each stock (str)
        date_column: column identifying each return period (str)
        n_simulations: number of random portfolios generated for each portfolio size (int)
        periods_per_year: 12 for monthly returns, 252 for daily returns (int)

    Output:
        dataframe with number_of_stocks vs average_volatility
    """

    # Convert to wide format:
    # rows = months, columns = stocks
    returns_wide = df.pivot(
        index=date_column,
        columns=ticker_column,
        values=return_column
    )

    # Number of available stocks
    n_stocks = len(returns_wide.columns)

    results = []

    # Try portfolio sizes from 1 stock up to all stocks
    for n in range(1, n_stocks + 1):

        simulated_volatilities = []

        for _ in range(n_simulations):

            # Randomly choose n stocks
            selected_stocks = returns_wide.sample(
                n=n,
                axis=1
            )
            
            # Only use months where all selected stocks have returns
            selected_stocks = selected_stocks.dropna()
            
            # Equal-weight portfolio return each month
            portfolio_returns = selected_stocks.mean(axis=1)
            
            # Annualized volatility
            volatility = (
                portfolio_returns.std()
                * np.sqrt(periods_per_year)
            )

            simulated_volatilities.append(volatility)

        # Average volatility across all random portfolios
        average_volatility = np.mean(simulated_volatilities)

        results.append({
            "number_of_stocks": n,
            "average_volatility": average_volatility
        })

    return pd.DataFrame(results)

def plot_correlation_heatmap(
    correlation_matrix,
    tickers=None,
    figsize=(10, 8),
    values = True
):
    """
    Plot a correlation heatmap for selected stocks.

    Input:
        correlation_matrix: Stock-level correlation matrix.
        tickers: optional list of stocks to display. If None, displays the full matrix.
        figsize: figure size.
        values: True to display numbers, False to just show colors
    Output:
        heatmap showing correlation between stocks
    """

    if tickers is not None:
        corr = correlation_matrix.loc[tickers, tickers]
    else:
        corr = correlation_matrix

    labels = corr.columns

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        corr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # Add correlation values
    if values == True:
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center"
                )

    ax.set_title("Stock Return Correlation Heatmap")

    plt.tight_layout()
    plt.show()

def prepare_monthly_returns(
    ohlcv,
    benchmark_ticker="SPY"
):
    """
    Prepare monthly stock returns for portfolio construction.

    Inputs:
        ohlcv: daily OHLCV dataframe
        benchmark_ticker: ticker to exclude from the portfolio

    Output:
        dataframe containing month, ticker, monthly_simple_return
    """

    df = ohlcv.copy()

    # Exclude benchmark
    df = df[df["ticker"] != benchmark_ticker].copy()

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Create monthly period
    df["month"] = df["date"].dt.to_period("M")

    # Extract last adjusted close of each month
    monthly_prices = (
        df.sort_values("date")
          .groupby(["ticker", "month"])["adjusted_close"]
          .last()
          .reset_index()
    )

    # Calculate monthly simple returns
    monthly_prices["monthly_simple_return"] = (
        monthly_prices.groupby("ticker")["adjusted_close"]
                      .pct_change(fill_method=None)
    )

    return monthly_prices[
        ["month", "ticker", "monthly_simple_return"]
    ]

def add_portfolio_eligibility(
    df,
    window=12,
    return_column="monthly_simple_return"
):
    """
    Mark stocks eligible for portfolio construction.
    A stock is eligible in month t if it has valid returns
    for all window months immediately preceding t.

    Input:
        df: dataframe containing portfolio weights
        window: rolling window for statistics such as volatility (int)
        return_column: column name for returns (str)
    Output:
        appended column indicating if month is eligible or not
    """

    df = df.copy()

    # Convert to wide format: months x stocks
    returns_wide = df.pivot(
        index="month",
        columns="ticker",
        values=return_column
    ).sort_index()

    # Require consecutive calendar months
    returns_wide = returns_wide.reindex(
        pd.period_range(
            returns_wide.index.min(),
            returns_wide.index.max(),
            freq="M"
        )
    )

    # Count valid returns over the previous window months
    historical_count = (
        returns_wide.notna()
                    .shift(1)
                    .rolling(window, min_periods=window)
                    .sum()
    )

    eligible = historical_count.eq(window)

    # Convert eligibility back to long format
    eligible_long = (
        eligible.stack()
                .rename("eligible")
                .reset_index()
    )

    eligible_long.columns = ["month", "ticker", "eligible"]

    df = df.merge(
        eligible_long,
        on=["month", "ticker"],
        how="left"
    )

    df["eligible"] = df["eligible"].fillna(False)

    return df

def inverse_volatility_weights(
    df,
    return_column="monthly_simple_return",
    date_column="month",
    ticker_column="ticker",
    window=12
):
    """
    Compute inverse-volatility portfolio weights using
    trailing historical returns.

    Inputs:
        df: dataframe containing monthly stock returns
        return_column: monthly return column (str)
        date_column: month column (str)
        ticker_column: stock identifier (str)
        window: number of historical months used for volatility (rolling)

    Output:
        dataframe with 1)rolling_volatility 2)inv_vol_weight
    """

    df = df.copy()

    # Sort by stock and month
    df = df.sort_values([ticker_column, date_column])

    # Calculate historical rolling volatility for each stock
    df["rolling_volatility"] = (
        df.groupby(ticker_column)[return_column]
          .transform(
              lambda x: x.shift(1).rolling(
                  window=window,
                  min_periods=window
              ).std()
          )
    )

    # Inverse volatility
    df["inverse_volatility"] = (
        1 / df["rolling_volatility"]
    )

    # Normalize weights within each month
    df["inv_vol_weight"] = (
        df["inverse_volatility"]
        /
        df.groupby(date_column)["inverse_volatility"].transform("sum")
    )

    return df

def calculate_risk_parity_weights(cov_matrix):
    """
    Calculate long-only equal-risk-contribution portfolio weights.

    Input:
        cov_matrix: covariance matrix of stock returns

    Output:
        pd.Series containing risk-parity weights
    """

    n = len(cov_matrix)
    cov = cov_matrix.to_numpy(dtype=float)

    # Symmetrize and regularize covariance
    cov = (cov + cov.T) / 2

    # Small diagonal regularization for numerical stability
    scale = np.trace(cov) / n
    cov = cov + 1e-4 * scale * np.eye(n)

    # Optimize in log-weights to guarantee positivity
    def objective(y):
        x = np.exp(y)
        return 0.5 * x @ cov @ x - np.sum(y)

    def gradient(y):
        x = np.exp(y)
        return x * (cov @ x) - 1

    # Initial guess based on individual volatilities
    vol = np.sqrt(np.diag(cov))
    x0 = 1 / vol

    result = minimize(
        objective,
        np.log(x0),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12}
    )

    if not result.success:
        raise ValueError(result.message)

    x = np.exp(result.x)
    weights = x / x.sum()

    return pd.Series(
        weights,
        index=cov_matrix.index,
        name="rp_weight"
    )

def add_risk_parity_weights(
    df,
    window=12,
    return_column="monthly_simple_return"
):
    """
    Add monthly risk-parity weights using historical covariance.

    Input:
        df: dataframe with month, ticker, monthly_simple_return, eligible
        window: rolling window, usually 12
        return_column: name of return column (str)

    Output:
        dataframe with rp_weight added.
    """

    df = df.copy()
    df["rp_weight"] = np.nan

    # Wide matrix: months x stocks
    returns_wide = df.pivot(
        index="month",
        columns="ticker",
        values=return_column
    ).sort_index()

    # Include missing calendar months explicitly
    returns_wide = returns_wide.reindex(
        pd.period_range(
            returns_wide.index.min(),
            returns_wide.index.max(),
            freq="M"
        )
    )

    # Go through each month
    for month in returns_wide.index:

        # Find eligible stocks for this month
        eligible_stocks = df.loc[
            (df["month"] == month) & df["eligible"],
            "ticker"
        ].tolist()

        if len(eligible_stocks) == 0:
            continue

        # Use only the previous window months
        month_position = returns_wide.index.get_loc(month)

        if month_position < window:
            continue

        historical_returns = returns_wide.iloc[
            month_position - window:month_position
        ][eligible_stocks]

        # Require complete historical observations
        if historical_returns.isna().any().any():
            continue

        # Estimate covariance matrix
        cov_matrix = historical_returns.cov()

        # Calculate risk parity weights
        weights = calculate_risk_parity_weights(cov_matrix)

        # Add weights to the original DataFrame
        mask = df["month"] == month

        df.loc[mask, "rp_weight"] = (
            df.loc[mask, "ticker"].map(weights)
        )

    return df

def calculate_portfolio_returns(
    df,
    weight_methods,
    return_column="monthly_simple_return"
):
    """
    Calculate monthly portfolio returns for multiple
    allocation methods.

    Input:
        df: dataframe containing portfolio returns
        weight_methods: dictionary containing column names for multiple portfolios
        return_column: column of monthly simple return
    """

    df = df.copy()

    for weight_col, portfolio_return_col in weight_methods.items():
        df[portfolio_return_col] = (
            df[weight_col] * df[return_column]
        )

    portfolio_returns = (
        df.groupby("month")[list(weight_methods.values())]
        .sum()
        .reset_index()
    )

    return portfolio_returns

def calculate_turnover(
    df,
    weight_column,
    return_column="monthly_simple_return"
):
    """
    Calculate monthly portfolio turnover.

    Assumes:
        - Weights are target weights held during each month.
        - Monthly rebalancing.
        - Fully invested, long-only portfolio.

    Returns:
        DataFrame containing month and turnover.
    """

    df = df.sort_values(["month", "ticker"]).copy()

    # Create wide matrices: months x stocks
    weights = df.pivot(
        index="month",
        columns="ticker",
        values=weight_column
    ).fillna(0)

    returns = df.pivot(
        index="month",
        columns="ticker",
        values=return_column
    ).reindex(
        index=weights.index,
        columns=weights.columns
    )

    # Previous month's target weights
    previous_weights = weights.shift(1)

    # Previous month's stock returns
    previous_returns = returns.shift(1)

    # Value of each position after market movement
    drifted_values = previous_weights * (1 + previous_returns)

    # Normalize to obtain pre-rebalancing weights
    pretrade_weights = drifted_values.div(
        drifted_values.sum(axis=1),
        axis=0
    )

    # Compare target and pretrade weights
    turnover = (
        0.5 * (weights - pretrade_weights).abs().sum(axis=1)
    )

    # First month has no previous holdings to compare
    turnover.iloc[0] = np.nan

    return turnover.reset_index(name="turnover")

def calculate_min_variance_weights(
    cov_matrix,
    max_weight=0.10
):
    """
    Calculate long-only minimum-variance portfolio weights.

    Input:
        cov_matrix: covariance matrix of stock returns
        max_weight: maximum allocation to any individual stock

    Output:
        pd.Series of optimal portfolio weights
    """

    n = len(cov_matrix)
    cov = cov_matrix.to_numpy(dtype=float)

    # Initial guess: equal weights
    initial_weights = np.ones(n) / n

    # Objective: minimize portfolio variance
    def objective(weights):
        return weights @ cov @ weights

    # Gradient of portfolio variance
    def gradient(weights):
        return 2 * cov @ weights

    # Constraint: weights sum to 1
    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    # Long-only and maximum-weight constraints
    bounds = [(0, max_weight)] * n

    # Check feasibility
    if n * max_weight < 1 - 1e-10:
        raise ValueError(
            "Maximum weight too small for the number of stocks."
        )

    result = minimize(
        objective,
        initial_weights,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12
        }
    )

    if not result.success:
        raise ValueError(
            f"Optimization failed: {result.message}"
        )

    return pd.Series(
        result.x,
        index=cov_matrix.index,
        name="min_var_weight"
    )

def add_min_variance_weights(
    df,
    window=12,
    max_weight=0.10,
    shrinkage=0.20,
    return_column="monthly_simple_return"
):
    """
    Add rolling minimum-variance portfolio weights. Uses historical returns only, with monthly rebalancing.
    Also includes shrinkage term for reducing the covariance matrix as an option, due to the large null space of the covariance matrix causing exploits with low # of stocks and very low volatility.

    Outputs:
        min_var_weight
        shrunk_min_var_weight
    """

    df = df.copy()

    df["min_var_weight"] = np.nan
    df["shrunk_min_var_weight"] = np.nan

    # Monthly return matrix: months x stocks
    returns_wide = (
        df.pivot(
            index="month",
            columns="ticker",
            values=return_column
        )
        .sort_index()
    )

    # Ensure the timeline includes all calendar months
    returns_wide = returns_wide.reindex(
        pd.period_range(
            returns_wide.index.min(),
            returns_wide.index.max(),
            freq="M"
        )
    )

    # Iterate through each month
    for month_position in range(window, len(returns_wide)):

        month = returns_wide.index[month_position]

        # Select eligible stocks for this month
        eligible_stocks = df.loc[
            (df["month"] == month) & df["eligible"],
            "ticker"
        ].tolist()

        if len(eligible_stocks) == 0:
            continue

        # Historical returns: previous 12 months
        history = returns_wide.iloc[
            month_position - window:month_position
        ][eligible_stocks]

        # Require complete historical observations
        if history.isna().any().any():
            continue

        # Original sample covariance
        cov_matrix = history.cov()

        # Diagonal shrinkage
        cov = cov_matrix.to_numpy()

        diagonal = np.diag(np.diag(cov))

        shrunk_cov = (
            (1 - shrinkage) * cov
            + shrinkage * diagonal
        )

        shrunk_cov_matrix = pd.DataFrame(
            shrunk_cov,
            index=cov_matrix.index,
            columns=cov_matrix.columns
        )

        # Original minimum-variance weights
        original_weights = calculate_min_variance_weights(
            cov_matrix,
            max_weight=max_weight
        )

        # Shrunk minimum-variance weights
        shrunk_weights = calculate_min_variance_weights(
            shrunk_cov_matrix,
            max_weight=max_weight
        )

        # Append weights to the dataframe
        mask = df["month"] == month

        df.loc[mask, "min_var_weight"] = (
            df.loc[mask, "ticker"].map(original_weights)
        )

        df.loc[mask, "shrunk_min_var_weight"] = (
            df.loc[mask, "ticker"].map(shrunk_weights)
        )

    return df

def optimize_target_return(
    expected_returns,
    cov_matrix,
    target_return,
    max_weight=0.10
):
    """
    Find minimum-variance weights for a target monthly return.

    Returns:
        pd.Series of optimal portfolio weights
    """

    mu = expected_returns.to_numpy()
    cov = cov_matrix.loc[
        expected_returns.index,
        expected_returns.index
    ].to_numpy()

    n = len(mu)

    initial_weights = np.ones(n) / n

    def objective(weights):
        return weights @ cov @ weights

    def gradient(weights):
        return 2 * cov @ weights

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        },
        {
            "type": "eq",
            "fun": lambda w: w @ mu - target_return
        }
    ]

    bounds = [(0, max_weight)] * n

    result = minimize(
        objective,
        initial_weights,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12
        }
    )

    if not result.success:
        raise ValueError(result.message)

    return pd.Series(
        result.x,
        index=expected_returns.index,
        name="optimized_weight"
    )

def calculate_factor_weights(df):
    """
    Short function that calculates turnover for momentum & value portfolios

    Input:
        df: dataframe of momentum/value portfolio
    Output:
        dataframe with added factor weight column
    """
    df = df.copy()

    df["factor_weight"] = 0.0

    long_count = df.groupby("month")["quintile"].transform(
        lambda x: (x == 5).sum()
    )

    short_count = df.groupby("month")["quintile"].transform(
        lambda x: (x == 1).sum()
    )

    df.loc[df["quintile"] == 5, "factor_weight"] = (
        1 / long_count[df["quintile"] == 5]
    )

    df.loc[df["quintile"] == 1, "factor_weight"] = (
        -1 / short_count[df["quintile"] == 1]
    )

    return df

def calculate_factor_target_turnover(df):
    """
    Short function that calculates turnover given a dataframe with factor weights

    Input:
        df: dataframe with factor weights
    Output:
        turnover value
    """
    weights_wide = (
        df.pivot(
            index="month",
            columns="ticker",
            values="factor_weight"
        )
        .sort_index()
        .fillna(0)
    )

    turnover = (
        weights_wide.diff()
        .abs()
        .sum(axis=1)
        / 2
    )

    turnover.iloc[0] = np.nan

    return turnover