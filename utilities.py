import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def fetch_data(tickers, start_date, end_date, market='India'):
    """
    Fetch historical prices (Adj Close or Close) for the given tickers.
    """
    valid_tickers = []
    for t in tickers:
        t = t.strip()
        if market == 'India' and not t.endswith(".NS") and not t.startswith("^"):
            valid_tickers.append(f"{t}.NS")
        else:
            valid_tickers.append(t)
            
    data = yf.download(valid_tickers, start=start_date, end=end_date)
    
    if hasattr(data.columns, 'get_level_values'):
        # Multi-index columns in newer yfinance versions
        if 'Adj Close' in data.columns.get_level_values(0):
            data = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            data = data['Close']
    else:
        # Fallback for older versions or single-level index
        if 'Adj Close' in data.columns:
            data = data['Adj Close']
        elif 'Close' in data.columns:
            data = data['Close']

    # If only one ticker is passed, make sure it remains a DataFrame with the ticker name as column
    if isinstance(data, pd.Series):
        if len(valid_tickers) == 1:
            data = data.to_frame(name=valid_tickers[0])
        else:
            data = data.to_frame()
            
    return data

def calculate_returns_and_cov(data):
    """
    Calculate annualized expected returns and covariance matrix.
    Assumes 252 trading days in a year.
    """
    daily_returns = data.pct_change().dropna()
    expected_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    return expected_returns, cov_matrix

def portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate=0.07):
    """
    Calculate portfolio return, volatility, and Sharpe Ratio.
    Using 7% as a default risk-free rate typical for Indian context (approx 10-yr G-Sec).
    """
    returns = np.sum(expected_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe_ratio

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.07):
    return -portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate)[2]

def portfolio_volatility(weights, expected_returns, cov_matrix, risk_free_rate=0.07):
    return portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate)[1]

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.07, target='Sharpe'):
    """
    Optimize portfolio for Max Sharpe or Min Volatility.
    """
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess (equal weights)
    init_guess = num_assets * [1. / num_assets,]
    
    if target == 'Sharpe':
        result = minimize(negative_sharpe_ratio, init_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    elif target == 'Volatility':
        result = minimize(portfolio_volatility, init_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        raise ValueError("Target must be 'Sharpe' or 'Volatility'")
        
    return result

def calculate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate=0.07, points=50):
    """
    Calculate points on the efficient frontier.
    """
    num_assets = len(expected_returns)
    
    # Get Min and Max returns bounds
    min_vol_result = optimize_portfolio(expected_returns, cov_matrix, risk_free_rate, target='Volatility')
    min_vol_ret, _, _ = portfolio_performance(min_vol_result.x, expected_returns, cov_matrix, risk_free_rate)
    max_ret = expected_returns.max()
    
    target_returns = np.linspace(min_vol_ret, max_ret, points)
    efficient_portfolios = []
    
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x, expected_returns, cov_matrix, risk_free_rate)[0] - target}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
        
        res = minimize(portfolio_volatility, init_guess, args=(expected_returns, cov_matrix, risk_free_rate),
                       method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            efficient_portfolios.append(res)
            
    return efficient_portfolios

def calculate_advanced_metrics(daily_returns, risk_free_rate=0.07):
    """
    Calculate advanced portfolio metrics given a Series of daily returns.
    """
    # Number of years
    days = len(daily_returns)
    years = days / 252
    
    if years <= 0:
        return {}

    # Cumulative Return
    cum_returns = (1 + daily_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    
    # CAGR
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Volatility
    annual_vol = daily_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    annual_return = daily_returns.mean() * 252
    sharpe = (annual_return - risk_free_rate) / (annual_vol if annual_vol > 0 else 1)
    
    # Sortino Ratio (using downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / (downside_vol if downside_vol > 0 else 1)
    
    # Max Drawdown
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Daily, Monthly returns (averages)
    avg_daily = daily_returns.mean()
    # Approx avg monthly
    avg_monthly = (1 + avg_daily) ** 21 - 1
    avg_yearly = (1 + avg_daily) ** 252 - 1
    
    return {
        "Cumulative Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Avg Daily Return": avg_daily,
        "Avg Monthly Return": avg_monthly,
        "Avg Yearly Return": avg_yearly
    }