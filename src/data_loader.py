# data_loader.py
import pandas as pd
import datetime
from typing import List

#############################################
# 1) Lista de todos os tickers no CSV
#############################################
def get_all_tickers(csv_path: str) -> List[str]:
    """
    Lê apenas as colunas Date e Ticker do CSV para descobrir 
    todos os tickers disponíveis, sem carregar toda a base.
    
    Retorna a lista de tickers únicos.
    """
    # Use 'usecols' para ler só as colunas necessárias
    df_tickers = pd.read_excel(csv_path, usecols=["Date", "Ticker"])
    tickers_unicos = df_tickers["Ticker"].unique().tolist()
    return sorted(tickers_unicos)

#############################################
# 2) Carrega apenas dados filtrados
#############################################
def load_filtered_data(
    csv_path: str,
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Lê o CSV inteiro, mas filtra somente as linhas dos tickers 
    e datas escolhidos, então faz pivot (Date como index, 
    colunas = Tickers, valores = AdjClose).

    Retorna um DataFrame pronto para cálculos (cada coluna = um ticker).
    """
    # Primeiro lemos o CSV com todas as colunas relevantes:
    #  (Date, Ticker, AdjClose, etc.)
    df_raw = pd.read_excel(csv_path, parse_dates=["Date"])
    
    # Filtro de tickers
    df_raw = df_raw[df_raw["Ticker"].isin(tickers)]
    
    # Filtro de data
    mask = (df_raw["Date"] >= pd.to_datetime(start_date)) & (df_raw["Date"] <= pd.to_datetime(end_date))
    df_raw = df_raw[mask]
    
    # Se ficar vazio, retorna um DF vazio (ou trate conforme necessidade)
    if df_raw.empty:
        return pd.DataFrame()
    
    # Pivotar: linhas = datas, colunas = tickers, valores = AdjClose
    df_pivot = df_raw.pivot(index="Date", columns="Ticker", values="AdjClose")
    
    # Ordena por data
    df_pivot.sort_index(inplace=True)
    
    return df_pivot
