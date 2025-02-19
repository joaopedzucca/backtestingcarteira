import streamlit as st
import pandas as pd
from src.data_loader import load_price_data
from src.backtesting import run_backtest

def main():
    st.title("Backtesting de Carteiras")

    df_prices = load_price_data("data/dados_ajustados_price_all.xlsx")

    all_tickers = df_prices.columns.tolist()

    st.sidebar.write("### Parâmetros do Backtest")
    start_date = st.sidebar.date_input("Data Início", pd.to_datetime("2012-12-31"))
    end_date = st.sidebar.date_input("Data Fim", pd.to_datetime("2023-01-01"))

    buy_selection = st.sidebar.multiselect("Tickers para Comprar", all_tickers)
    buy_weights = [st.sidebar.number_input(f"Peso de {ticker}", 0.0, 1.0, 0.1) for ticker in buy_selection]

    if st.sidebar.button("Rodar Backtest"):
        st.write("## Resultados do Backtest")
        result = run_backtest(df_prices, buy_selection, buy_weights, str(start_date), str(end_date))

        st.write("### Métricas")
        st.write(result['metrics'])
        st.line_chart(result['portfolio_curve'])

if __name__ == "__main__":
    main()
