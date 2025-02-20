# src/backtesting.py
import pandas as pd
from typing import List, Dict
import numpy as np

from src.metrics import cagr, volatility, sharpe_ratio, max_drawdown

def run_backtest(
    df_prices: pd.DataFrame,
    buy_tickers: List[str],
    buy_weights: List[float],
    sell_tickers: List[str] = None,
    sell_weights: List[float] = None,
    start_date: str = "2012-01-01",
    end_date: str = "2025-01-01",
    risk_free_annual: float = 0.13
) -> Dict:
    """
    Executa um backtest simples (buy & hold) em um DataFrame pivotado 
    contendo preços de 'Close'.

    Parâmetros
    ----------
    df_prices : pd.DataFrame
        colunas = tickers, index = datas, valores = Close
    buy_tickers, buy_weights : ativos e pesos (ex.: [0.5, 0.5])
    sell_tickers, sell_weights : short, transformados em pesos negativos
    start_date, end_date : período de simulação
    risk_free_annual : taxa livre de risco anual, ex.: 0.13 = 13%

    Retorna
    -------
    dict {
       'portfolio_curve': pd.Series,
       'metrics': {
         'final_return': float,
         'cagr': float,
         'volatility': float,
         'sharpe': float,
         'max_drawdown': float
       }
    }
    """
    if sell_tickers is None:
        sell_tickers = []
    if sell_weights is None:
        sell_weights = []

    # Todos os tickers relevantes
    all_tickers = buy_tickers + sell_tickers
    # Pesos (positivos de compra, negativos de venda)
    all_weights = buy_weights + [-w for w in sell_weights]

    # Filtra período e colunas
    df_period = df_prices.loc[start_date:end_date, all_tickers].copy()
    # Remove linhas que sejam totalmente NaN
    df_period.dropna(how='all', inplace=True)

    if df_period.empty:
        return {
            'portfolio_curve': pd.Series([], dtype=float),
            'metrics': {}
        }

    # Retornos diários
    daily_returns = df_period.pct_change().fillna(0)

    # Soma ponderada dos retornos
    portfolio_returns = (daily_returns * all_weights).sum(axis=1)

    # Curva do portfólio (iniciando em 1.0)
    portfolio_curve = (1 + portfolio_returns).cumprod()

    # Métricas
    final_return = portfolio_curve.iloc[-1] - 1
    cagr_val = cagr(portfolio_curve)
    vol_val = volatility(portfolio_returns)
    sharpe_val = sharpe_ratio(portfolio_returns, risk_free=risk_free_annual)
    mdd = max_drawdown(portfolio_curve)

    metrics_dict = {
        'final_return': final_return,
        'cagr': cagr_val,
        'volatility': vol_val,
        'sharpe': sharpe_val,
        'max_drawdown': mdd
    }

    return {
        'portfolio_curve': portfolio_curve,
        'metrics': metrics_dict
    }
