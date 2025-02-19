import numpy as np
import pandas as pd

def cagr(portfolio_values: pd.Series) -> float:
    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    total_return = final_value / initial_value - 1
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    return (1 + total_return) ** (1 / years) - 1

def volatility(portfolio_returns: pd.Series, annual_factor=252) -> float:
    return portfolio_returns.std() * np.sqrt(annual_factor)

def sharpe_ratio(portfolio_returns: pd.Series, risk_free=0.13, annual_factor=252) -> float:
    mean_return_daily = portfolio_returns.mean()
    annual_return = mean_return_daily * annual_factor
    vol = volatility(portfolio_returns, annual_factor)
    return (annual_return - risk_free) / vol if vol != 0 else 0

def max_drawdown(portfolio_values: pd.Series) -> float:
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()
