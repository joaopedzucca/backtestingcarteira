import pandas as pd
import streamlit as st

from src.data_loader import get_all_tickers, load_filtered_data
from src.backtesting import run_backtest

PARQUET_PATH = "data/dados_ajustados_price_all.parquet"

def main():
    st.set_page_config(page_title="O Melhor Backtest do Mundo", layout="wide")
    st.title("O Melhor Backtest do Mundo - Ousadia e Alegria")

    # 1) Sidebar com parâmetros
    st.sidebar.header("Parâmetros do Backtest")

    # Carregar tickers disponíveis
    tickers_all = get_all_tickers(PARQUET_PATH)
    st.sidebar.write(f"Total de tickers disponíveis: {len(tickers_all)}")

    buy_selection = st.sidebar.multiselect("Tickers para COMPRAR", tickers_all)
    sell_selection = st.sidebar.multiselect("Tickers para VENDER (short)", tickers_all)

    # Pesos de compra
    buy_weights = []
    for tk in buy_selection:
        w = st.sidebar.number_input(f"Peso para {tk} (compra)", 0.0, 1.0, 0.10, 0.01)
        buy_weights.append(w)

    # Pesos de venda (short)
    sell_weights = []
    for tk in sell_selection:
        w = st.sidebar.number_input(f"Peso para {tk} (short)", 0.0, 1.0, 0.10, 0.01)
        sell_weights.append(w)

    # Datas
    start_date = st.sidebar.date_input("Data Início", pd.to_datetime("2012-01-02"))
    end_date = st.sidebar.date_input("Data Fim", pd.to_datetime("2020-01-02"))

    # Taxa livre de risco e capital inicial
    risk_free = st.sidebar.number_input("Taxa Livre de Risco (ex: 0.13 = 13%)", 0.0, 1.0, 0.13, 0.01)
    initial_capital = st.sidebar.number_input("Capital Inicial (R$)", min_value=1000, value=100000, step=1000)

    # Botão de rodar o backtest
    if st.sidebar.button("Executar Backtest"):
        st.subheader("Resultados do Backtest")

        # Carrega dados para só os tickers escolhidos, no período
        all_tickers_needed = list(set(buy_selection + sell_selection))
        if not all_tickers_needed:
            st.error("Nenhum ticker selecionado.")
            return

        df_filtered = load_filtered_data(
            parquet_path=PARQUET_PATH,
            tickers=all_tickers_needed,
            start_date=str(start_date),
            end_date=str(end_date)
        )

        if df_filtered.empty:
            st.warning("Não há dados para esse período e/ou para esses tickers.")
            return

        # Executa o backtest
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

        portfolio_curve = result["portfolio_curve"]
        metrics = result["metrics"]

        if portfolio_curve.empty:
            st.warning("Curva vazia (sem dados).")
            return

        # 2) Converte a curva relativa (inicia em 1.0) para R$ (capital inicial)
        portfolio_value = portfolio_curve * initial_capital

        # 3) Plotar gráfico
        st.line_chart(portfolio_value, height=400, use_container_width=True)

        # 4) Exibir métricas em colunas bonitinhas
        # final_return, cagr, volatility, sharpe, max_drawdown
        final_return_pct = metrics.get('final_return', 0) * 100
        cagr_pct = metrics.get('cagr', 0) * 100
        vol_pct = metrics.get('volatility', 0) * 100
        sharpe_val = metrics.get('sharpe', 0)
        max_dd_pct = metrics.get('max_drawdown', 0) * 100

        st.write("---")
        st.markdown("### Indicadores de Desempenho")

        # Usar st.columns e st.metric para formatar
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retorno Acumulado", f"{final_return_pct:.2f}%")
            st.metric("CAGR", f"{cagr_pct:.2f}%")
        with col2:
            st.metric("Volatilidade (a.a.)", f"{vol_pct:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe_val:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{max_dd_pct:.2f}%")

        # 5) Exibir valor final do portfólio
        valor_final = portfolio_value.iloc[-1]
        lucro = valor_final - initial_capital
        retorno_pc = final_return_pct

        st.write("---")
        st.markdown(f"### Valor Final da Carteira: **R$ {valor_final:,.2f}**")
        st.write(f"Lucro no período: R$ {lucro:,.2f} ({retorno_pc:.2f}%)")

        st.success("Backtest executado com sucesso! Ousadia e Alegria!")


if __name__ == "__main__":
    main()
