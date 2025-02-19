import pandas as pd

def load_price_data(csv_path: str) -> pd.DataFrame:
    """
    Lê um CSV com colunas [Date, Ticker, Adj Close, ...]
    e retorna um DataFrame formatado para backtest.
    """
    df_raw = pd.read_excel(csv_path)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%Y-%m-%d')
    df_pivot = df_raw.pivot(index='Date', columns='Ticker', values='Adj Close')
    df_pivot.sort_index(inplace=True)
    df_pivot.dropna(how='all', inplace=True)
    return df_pivot
