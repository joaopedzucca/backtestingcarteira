# src/data_loader.py
import pandas as pd
from typing import List

def get_all_tickers(parquet_path: str) -> List[str]:
    """
    Lê o arquivo Parquet para descobrir todos os tickers disponíveis.
    Retorna uma lista de strings ordenada.
    """
    # Carrega apenas a coluna 'Ticker' para economizar memória
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
    Lê o Parquet completo, mas filtra SOMENTE as linhas dos tickers 
    e datas selecionados, depois pivota em 'Close'.

    Retorna um DataFrame com:
      - index = Date (datetime)
      - columns = Ticker
      - values = Close
    """
    df_raw = pd.read_parquet(parquet_path)
    
    # Converte Date para datetime se necessário
    if not pd.api.types.is_datetime64_any_dtype(df_raw['Date']):
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    # Filtro de tickers
    df_raw = df_raw[df_raw['Ticker'].isin(tickers)]
    
    # Filtro de datas
    mask = (df_raw['Date'] >= pd.to_datetime(start_date)) & (df_raw['Date'] <= pd.to_datetime(end_date))
    df_raw = df_raw[mask]
    
    if df_raw.empty:
        return pd.DataFrame()
    
    # Pivotar para index=Date, columns=Ticker, values=Close
    df_pivot = df_raw.pivot(index='Date', columns='Ticker', values='Close')
    df_pivot.sort_index(inplace=True)
    
    return df_pivot
def load_cdi(path: str, start_date: str, end_date: str) -> pd.Series:
    """
    Lê o arquivo cdi.parquet, filtra datas e retorna uma série com o 
    fator acumulado do CDI (index = datas, values = 'CDI acumulado').
    """
    df_cdi = pd.read_parquet(path)
    # Garante que Date seja datetime
    df_cdi['Date'] = pd.to_datetime(df_cdi['Date'])
    # Filtro datas
    mask = (df_cdi['Date'] >= start_date) & (df_cdi['Date'] <= end_date)
    df_cdi = df_cdi[mask].copy()
    if df_cdi.empty:
        return pd.Series([], dtype=float)

    df_cdi.sort_values('Date', inplace=True)
    df_cdi.set_index('Date', inplace=True)

    # Supondo que 'valor' é a taxa diária (ex.: 0.0003)
    # Criar fator diário = (1 + taxa)
    df_cdi['fator'] = 1 + df_cdi['valor']
    # Fator acumulado = cumprod
    df_cdi['CDI_acumulado'] = df_cdi['fator'].cumprod()

    return df_cdi['CDI_acumulado']


def load_ibov(path: str, start_date: str, end_date: str) -> pd.Series:
    """
    Lê ibov.parquet (colunas: Date, Close), filtra datas,
    e retorna uma série rebaseada iniciando em 1.0.
    """
    df_ibov = pd.read_parquet(path)
    df_ibov['Date'] = pd.to_datetime(df_ibov['Date'])

    mask = (df_ibov['Date'] >= start_date) & (df_ibov['Date'] <= end_date)
    df_ibov = df_ibov[mask].copy()
    if df_ibov.empty:
        return pd.Series([], dtype=float)

    df_ibov.sort_values('Date', inplace=True)
    df_ibov.set_index('Date', inplace=True)

    # Rebase: divide todos os valores pela 1ª cotação
    first_close = df_ibov['Close'].iloc[0]
    df_ibov['IBOV_acumulado'] = df_ibov['Close'] / first_close

    return df_ibov['IBOV_acumulado']
