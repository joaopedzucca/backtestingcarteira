# src/data_loader.py

import pandas as pd
from typing import List

def get_all_tickers(parquet_path: str) -> List[str]:
    """
    Lê somente a coluna 'Ticker' de um Parquet e retorna a lista de tickers únicos.
    """
    df_tmp = pd.read_parquet(parquet_path, columns=['Ticker'])
    tickers_unicos = df_tmp['Ticker'].unique().tolist()
    return sorted(tickers_unicos)

def load_filtered_data(
    parquet_path: str,
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Lê o Parquet (com colunas Date, Ticker, Close, etc.), filtra pelos `tickers` e
    datas [start_date, end_date], e faz pivot em 'Close'.

    Retorna:
      DataFrame pivotado (index=Date, columns=Ticker, values=Close).
      Se não houver dados, retorna DataFrame vazio.
    """
    df_raw = pd.read_parquet(parquet_path)
    # Converte a coluna Date para datetime
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

    # Filtra tickers
    df_raw = df_raw[df_raw['Ticker'].isin(tickers)]

    # Converte parâmetros para datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Filtra datas
    mask = (df_raw['Date'] >= start_dt) & (df_raw['Date'] <= end_dt)
    df_raw = df_raw[mask]
    if df_raw.empty:
        return pd.DataFrame()

    # Pivotar
    df_pivot = df_raw.pivot(index='Date', columns='Ticker', values='Close')
    df_pivot.sort_index(inplace=True)

    return df_pivot

def load_cdi(parquet_path: str, start_date: str, end_date: str) -> pd.Series:
    """
    Lê cdi.parquet (colunas: Date, valor), filtra pelo período e retorna
    a Série 'CDI_acumulado' indexada por Date.

    - 'valor' deve ser a taxa diária, ex.: 0.0003 (0,03%/dia).
    - O resultado CDI_acumulado inicia no primeiro dia do período (>0).
    """
    df_cdi = pd.read_parquet(parquet_path)
    df_cdi['Date'] = pd.to_datetime(df_cdi['Date'], errors='coerce')

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (df_cdi['Date'] >= start_dt) & (df_cdi['Date'] <= end_dt)
    df_cdi = df_cdi[mask].copy()

    if df_cdi.empty:
        return pd.Series([], dtype=float)

    df_cdi.sort_values('Date', inplace=True)
    df_cdi.set_index('Date', inplace=True)

    # Cria fator diário e acumula
    df_cdi['fator'] = 1 + df_cdi['valor']
    df_cdi['CDI_acumulado'] = df_cdi['fator'].cumprod()

    return df_cdi['CDI_acumulado']

def load_ibov(parquet_path: str, start_date: str, end_date: str) -> pd.Series:
    """
    Lê ibov.parquet (Date, Close), filtra o período e retorna
    uma Série rebaseada 'IBOV_acumulado' iniciando em 1.0 no primeiro dia.
    """
    df_ibov = pd.read_parquet(parquet_path)
    df_ibov['Date'] = pd.to_datetime(df_ibov['Date'], errors='coerce')

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (df_ibov['Date'] >= start_dt) & (df_ibov['Date'] <= end_dt)
    df_ibov = df_ibov[mask].copy()

    if df_ibov.empty:
        return pd.Series([], dtype=float)

    df_ibov.sort_values('Date', inplace=True)
    df_ibov.set_index('Date', inplace=True)

    first_close = df_ibov['Close'].iloc[0]
    df_ibov['IBOV_acumulado'] = df_ibov['Close'] / first_close

    return df_ibov['IBOV_acumulado']
