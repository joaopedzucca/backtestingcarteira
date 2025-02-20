import pandas as pd
import numpy as np
from typing import List, Dict

# Importe suas funções de métricas personalizadas
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
    Executa um backtest simples (buy & hold) em um DataFrame pivotado, 
    onde cada coluna corresponde a um ticker, e o índice são as datas.

    Parâmetros
    ----------
    df_prices : pd.DataFrame
        DataFrame com as cotações (ex.: 'Adj Close'), pivotado:
        - index: datas (datetime)
        - columns: tickers (str)
    buy_tickers : List[str]
        Lista de tickers que serão comprados.
    buy_weights : List[float]
        Lista de pesos para cada ticker da buy_tickers (ex.: [0.10, 0.20]).
        A soma dos pesos de compra pode ou não ser 1, dependendo da estratégia.
    sell_tickers : List[str], opcional
        Lista de tickers que serão vendidos (short). Se não houver, pode ser None.
    sell_weights : List[float], opcional
        Pesos para short, correspondentes aos tickers de sell_tickers (ex.: [0.10, 0.10]).
        Se não houver, pode ser None.
    start_date : str
        Data de início do período (formato 'YYYY-MM-DD').
    end_date : str
        Data de término do período.
    risk_free_annual : float
        Taxa livre de risco anual para cálculo do Sharpe. Ex.: 0.13 (13% a.a.).

    Retorno
    -------
    Dict
        Um dicionário com as chaves:
        - 'portfolio_curve': pd.Series com a evolução do valor do portfólio (inicia em 1.0).
        - 'metrics': dict com as métricas (cagr, volatility, sharpe, max_drawdown, final_return).

    Exemplo de uso
    --------------
    >>> result = run_backtest(
    ...     df_prices=df, 
    ...     buy_tickers=["PETR4.SA", "VALE3.SA"],
    ...     buy_weights=[0.5, 0.5],
    ...     sell_tickers=["OIBR3.SA"],
    ...     sell_weights=[0.2],
    ...     start_date="2012-01-01",
    ...     end_date="2023-01-01",
    ...     risk_free_annual=0.13
    ... )
    >>> portfolio = result['portfolio_curve']
    >>> metrics = result['metrics']
    >>> print(metrics)
    """
    if sell_tickers is None:
        sell_tickers = []
    if sell_weights is None:
        sell_weights = []

    # Lista total de tickers
    all_tickers = buy_tickers + sell_tickers
    # Pesos: compras são positivos, shorts são negativos
    all_weights = buy_weights + [-w for w in sell_weights]

    # Filtra o DataFrame no período e nos tickers relevantes
    df_period = df_prices.loc[start_date:end_date, all_tickers].copy()
    # Remove linhas que sejam totalmente NaN
    df_period.dropna(how='all', inplace=True)

    # Calcula retornos diários
    daily_returns = df_period.pct_change().fillna(0)

    # Retorno do portfólio = soma ponderada dos retornos
    # Obs.: isto assume buy & hold com pesos fixos (proporcional ao capital inicial).
    portfolio_returns = (daily_returns * all_weights).sum(axis=1)

    # Evolução do portfólio, iniciando em 1.0
    portfolio_curve = (1 + portfolio_returns).cumprod()

    # Cálculo das métricas
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
