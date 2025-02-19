import pandas as pd
import numpy as np
from typing import List, Dict
from src.metrics import cagr, volatility, sharpe_ratio, max_drawdown

def run_backtest(
    df_prices: pd.DataFrame,
    buy_tickers: List[str],
    buy_weights: List[float],
    start_date: str = "2012-01-01",
    end_date: str = "2025-01-01",
    risk_free_annual: float = 0.13
) -> Dict:
    df_period = df_prices.loc[start_date:end_date, buy_tickers].copy()
    df_period.dropna(how='all', inplace=True)

    daily_returns = df_period.pct_change().fillna(0)
    portfolio_returns = (daily_returns * buy_weights).sum(axis=1)
    portfolio_curve = (1 + portfolio_returns).cumprod()

    metrics_dict = {
        'cagr': cagr(portfolio_curve),
        'volatility': volatility(portfolio_returns),
        'sharpe': sharpe_ratio(portfolio_returns, risk_free=risk_free_annual),
        'max_drawdown': max_drawdown(portfolio_curve)
    }

    return {
        'portfolio_curve': portfolio_curve,
        'metrics': metrics_dict
    }
