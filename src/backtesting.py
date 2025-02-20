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
        (ex.: df["PETR4.SA"], df["VALE3.SA"], etc.)
    buy_tickers, buy_weights : lista de ativos e pesos (ex.: [0.5, 0.5])
    sell_tickers, sell_weights : lista de ativos para short e seus pesos
                                 (transformados internamente em pesos negativos)
    start_date, end_date : período de simulação (strings 'YYYY-MM-DD')
    risk_free_annual : taxa livre de risco anual (ex.: 0.13 = 13%)

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

    Observações
    -----------
    - Se algum ticker não existir nas colunas de df_prices, ele será ignorado.
    - Se não sobrar nenhum ticker válido ou o DataFrame ficar vazio no período,
      o retorno será um dicionário com 'portfolio_curve' vazio e 'metrics' vazio.
    """
    if sell_tickers is None:
        sell_tickers = []
    if sell_weights is None:
        sell_weights = []

    # Combina todos os tickers (compra e venda)
    all_tickers = buy_tickers + sell_tickers
    # Pesos: compra (positivos), venda (negativos)
    all_weights = buy_weights + [-w for w in sell_weights]

    # 1) Garantir que só usamos tickers existentes no df_prices
    valid_tickers = [t for t in all_tickers if t in df_prices.columns]
    if not valid_tickers:
        # Nenhum ticker encontrado na base
        return {
            'portfolio_curve': pd.Series([], dtype=float),
            'metrics': {}
        }

    # 2) Filtra datas e colunas válidas
    df_period = df_prices.loc[start_date:end_date, valid_tickers].copy()

    # Remove linhas que sejam totalmente NaN
    df_period.dropna(how='all', inplace=True)

    # Se ainda assim estiver vazio (sem linhas ou sem colunas)
    if df_period.empty:
        return {
            'portfolio_curve': pd.Series([], dtype=float),
            'metrics': {}
        }

    # 3) Calcula retornos diários
    daily_returns = df_period.pct_change().fillna(0)

    # 4) Retorno do portfólio = soma ponderada dos retornos
    portfolio_returns = (daily_returns * all_weights).sum(axis=1)

    # 5) Curva do portfólio (iniciando em 1.0)
    portfolio_curve = (1 + portfolio_returns).cumprod()

    # 6) Métricas
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
