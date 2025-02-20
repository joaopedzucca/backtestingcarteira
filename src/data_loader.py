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
