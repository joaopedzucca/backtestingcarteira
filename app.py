# app.py

import streamlit as st
import pandas as pd
from typing import List

from src.data_loader import get_all_tickers, load_filtered_data
from src.backtesting import run_backtest

CSV_PATH = "data/dados_ajustados_price_all.parquet"  # Ajuste para seu caminho

def main():
    st.title("Backtesting - Carregar só o necessário")

    # 1) Descobrir todos os tickers
    st.write("Carregando lista de tickers disponíveis...")
    tickers_all = get_all_tickers(CSV_PATH)
    st.write(f"Total de tickers disponíveis: {len(tickers_all)}")

    # 2) Seletor de tickers (compra e venda)
    buy_selection = st.multiselect("Tickers para COMPRAR", tickers_all)
    sell_selection = st.multiselect("Tickers para VENDER/SHORT", tickers_all)

    # 3) Pesos para cada lista
    buy_weights = []
    for tk in buy_selection:
        w = st.number_input(f"Peso de {tk} (compra)", 0.0, 1.0, 0.1, 0.01)
        buy_weights.append(w)
    
    sell_weights = []
    for tk in sell_selection:
        w = st.number_input(f"Peso de {tk} (venda)", 0.0, 1.0, 0.1, 0.01)
        sell_weights.append(w)

    # 4) Datas
    start_date = st.date_input("Data Início", pd.to_datetime("2012-01-01"))
    end_date = st.date_input("Data Fim", pd.to_datetime("2023-01-01"))

    # 5) Rodar backtest só quando clicar no botão
    if st.button("Rodar Backtest"):
        # Junta todos os tickers que vamos precisar
        tickers_needed = list(set(buy_selection + sell_selection))
        
        if not tickers_needed:
            st.warning("Nenhum ticker selecionado!")
            return
        
        # 5.1) Carrega SOMENTE esses tickers e essas datas
        df_filtered = load_filtered_data(
            CSV_PATH,
            tickers=tickers_needed,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        
        if df_filtered.empty:
            st.warning("Não há dados para esse filtro!")
            return
        
        # 5.2) Executa o backtest
        portfolio_curve = run_backtest(
            df_prices=df_filtered,
            buy_tickers=buy_selection,
            buy_weights=buy_weights,
            sell_tickers=sell_selection,
            sell_weights=sell_weights
        )
        
        # 5.3) Plotar resultados
        st.write("### Curva do Portfólio")
        st.line_chart(portfolio_curve)

        # Exemplo: valor final
        final_val = portfolio_curve.iloc[-1]
        st.write(f"Retorno final: {final_val - 1:.2%}")


if __name__ == "__main__":
    main()
