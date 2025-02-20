# src/metrics.py
import numpy as np
import pandas as pd

def cagr(portfolio_values: pd.Series) -> float:
    """
    Taxa de Crescimento Anual Composta.
    portfolio_values: série com valor do portfólio ao longo do tempo (index=date).
    """
    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    total_return = final_value / initial_value - 1
    
    # Nº de dias totais
    days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    years = days / 365.25  # aproximado
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1

def volatility(portfolio_returns: pd.Series, annual_factor=252) -> float:
    """
    Desvio padrão anualizado dos retornos diários (ou semanais).
    portfolio_returns: série de retornos (ex.: daily).
    """
    return portfolio_returns.std() * np.sqrt(annual_factor)

def sharpe_ratio(portfolio_returns: pd.Series, risk_free=0.13, annual_factor=252) -> float:
    """
    Sharpe = (Retorno_esperado - risk_free) / Volatilidade
    Aqui simplificamos usando a média de retorno diário * annual_factor.
    risk_free = 0.13 (13%) como taxa livre de risco anual.
    """
    mean_return_daily = portfolio_returns.mean()
    annual_return = mean_return_daily * annual_factor
    vol = portfolio_returns.std() * np.sqrt(annual_factor)
    if vol == 0:
        return 0.0
    return (annual_return - risk_free) / vol

def max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Retorna o maior drawdown (valor negativo) da série de valor do portfólio.
    """
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()  # valor negativo
