# app.py
import pandas as pd
import streamlit as st

from src.data_loader import get_all_tickers, load_filtered_data
from src.backtesting import run_backtest

PARQUET_PATH = "data/dados_ajustados_price_all.parquet"

def main():
    st.title("Backtesting com Parquet e Close")

    # Lista de tickers
    st.write("Carregando lista de tickers disponíveis...")
    tickers_all = get_all_tickers(PARQUET_PATH)
    st.write(f"Total de tickers: {len(tickers_all)}")

    # Escolha de tickers
    buy_selection = st.multiselect("Tickers para COMPRAR", tickers_all)
    sell_selection = st.multiselect("Tickers para VENDER (short)", tickers_all)

    # Pesos
    buy_weights = []
    for tk in buy_selection:
        w = st.number_input(f"Peso de compra {tk}", 0.0, 1.0, 0.10, 0.01)
        buy_weights.append(w)

    sell_weights = []
    for tk in sell_selection:
        w = st.number_input(f"Peso de short {tk}", 0.0, 1.0, 0.10, 0.01)
        sell_weights.append(w)

    # Datas
    start_date = st.date_input("Data Início", pd.to_datetime("2012-01-01"))
    end_date = st.date_input("Data Fim", pd.to_datetime("2020-01-01"))

    # Taxa livre de risco
    risk_free = st.number_input("Taxa livre de risco (ex.: 0.13 = 13%)", 0.0, 1.0, 0.13, 0.01)

    if st.button("Rodar Backtest"):
        # Carrega DF filtrado
        all_tickers_needed = list(set(buy_selection + sell_selection))
        if not all_tickers_needed:
            st.error("Nenhum ticker selecionado!")
            return

        df_filtered = load_filtered_data(
            parquet_path=PARQUET_PATH,
            tickers=all_tickers_needed,
            start_date=str(start_date),
            end_date=str(end_date)
        )

        if df_filtered.empty:
            st.warning("Não há dados para esse filtro.")
            return

        # Roda backtest
        result = run_backtest(
            df_prices=df_filtered,
            buy_tickers=buy_selection,
            buy_weights=buy_weights,
            sell_tickers=sell_selection,
            sell_weights=sell_weights,
            start_date=str(start_date),
            end_date=str(end_date),
            risk_free_annual=risk_free
        )

        curve = result["portfolio_curve"]
        metrics = result["metrics"]

        if curve.empty:
            st.warning("Curva vazia (sem dados).")
        else:
            st.line_chart(curve, height=400)

        st.write("### Métricas do Portfólio")
        st.write(metrics)


if __name__ == "__main__":
    main()
