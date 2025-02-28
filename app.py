import streamlit as st
import pandas as pd

from src.data_loader import (
    get_all_tickers,
    load_filtered_data,
    load_cdi,
    load_ibov
)
from src.backtesting import run_backtest

# Ajuste os caminhos para seus arquivos .parquet
PARQUET_PATH_PRECOS = "data/precos.parquet"
PARQUET_PATH_CDI    = "data/cdi.parquet"
PARQUET_PATH_IBOV   = "data/ibov.parquet"

def main():
    st.set_page_config(page_title="Backtest Dinâmico", layout="wide")
    st.title("Backtest com Comparação CDI e IBOV")

    # Sidebar: datas, capital
    st.sidebar.header("Parâmetros de Data e Capital")
    start_date = st.sidebar.date_input("Data Início", pd.to_datetime("2012-01-02"))
    end_date   = st.sidebar.date_input("Data Fim",    pd.to_datetime("2020-01-02"))
    initial_capital = st.sidebar.number_input("Capital Inicial (R$)", min_value=1000, value=100000, step=1000)
    risk_free = st.sidebar.number_input("Taxa Livre de Risco (Ex.: 0.13 = 13%)", 0.0, 1.0, 0.13, 0.01)

    # Tabs: Seleção e Resultados
    tab1, tab2 = st.tabs(["Seleção de Ativos", "Resultados do Backtest"])

    with tab1:
        st.subheader("Selecione Tickers de Compra/Venda")

        # Carregar todos os tickers
        tickers_all = get_all_tickers(PARQUET_PATH_PRECOS)
        st.write(f"Total de tickers disponíveis: {len(tickers_all)}")

        # Compra
        st.markdown("#### Tickers para Comprar")
        buy_selection = st.multiselect("Selecione ativos para COMPRA", tickers_all, key="buy_select")
        equal_weights_buy = st.checkbox("Pesos Iguais (Compra)", value=False, key="chk_buy")

        buy_weights = []
        if buy_selection:
            if equal_weights_buy:
                peso = 1.0 / len(buy_selection)
                st.info(f"Pesos iguais de {peso:.2f} para cada ativo de COMPRA.")
                buy_weights = [peso] * len(buy_selection)
            else:
                st.write("Defina os pesos manualmente:")
                for tk in buy_selection:
                    w = st.number_input(f"Peso de {tk} (compra)", 0.0, 1.0, 0.10, 0.01, key=f"buy_{tk}")
                    buy_weights.append(w)

        # Venda
        st.markdown("#### Tickers para Vender (Short)")
        sell_selection = st.multiselect("Selecione ativos para VENDA (short)", tickers_all, key="sell_select")
        equal_weights_sell = st.checkbox("Pesos Iguais (Venda)", value=False, key="chk_sell")

        sell_weights = []
        if sell_selection:
            if equal_weights_sell:
                peso = 1.0 / len(sell_selection)
                st.info(f"Pesos iguais de {peso:.2f} para cada ativo de VENDA.")
                sell_weights = [peso] * len(sell_selection)
            else:
                st.write("Defina os pesos manualmente:")
                for tk in sell_selection:
                    w = st.number_input(f"Peso de {tk} (short)", 0.0, 1.0, 0.10, 0.01, key=f"sell_{tk}")
                    sell_weights.append(w)

        st.write("---")
        if st.button("Executar Backtest!"):
            st.session_state["execute_backtest"] = True
        else:
            st.session_state["execute_backtest"] = False

    # Resultado
    with tab2:
        st.subheader("Resultados do Backtest")

        if not st.session_state.get("execute_backtest"):
            st.info("Selecione os ativos e clique em 'Executar Backtest' na aba anterior.")
            return

        all_tickers_needed = list(set(buy_selection + sell_selection))
        if not all_tickers_needed:
            st.error("Nenhum ticker foi selecionado (compra ou venda).")
            return

        # Carrega DF filtrado
        df_filtered = load_filtered_data(
            parquet_path=PARQUET_PATH_PRECOS,
            tickers=all_tickers_needed,
            start_date=str(start_date),
            end_date=str(end_date),
        )
        if df_filtered.empty:
            st.warning("Não há dados para esse período/tickers.")
            return

        # Roda o backtest
        from src.backtesting import run_backtest  # se já importado, ok
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
            st.warning("Portfólio sem dados (vazio).")
            return

        # Converte para R$
        portfolio_value = portfolio_curve * initial_capital

        # Carregar CDI / IBOV
        cdi_series = load_cdi(PARQUET_PATH_CDI, str(start_date), str(end_date))
        ibov_series = load_ibov(PARQUET_PATH_IBOV, str(start_date), str(end_date))

        # Rebase CDI e IBOV e multiplica por capital
        if not cdi_series.empty:
            cdi_series /= cdi_series.iloc[0]
            cdi_value = cdi_series * initial_capital
        else:
            cdi_value = pd.Series([], dtype=float)

        if not ibov_series.empty:
            ibov_series /= ibov_series.iloc[0]
            ibov_value = ibov_series * initial_capital
        else:
            ibov_value = pd.Series([], dtype=float)

        # Plot
        df_plot = pd.DataFrame({"Carteira": portfolio_value})
        if not cdi_value.empty:
            df_plot["CDI"] = cdi_value
        if not ibov_value.empty:
            df_plot["IBOV"] = ibov_value

        st.line_chart(df_plot, height=450, use_container_width=True)

        # Exibe métricas
        final_return_pct = metrics.get('final_return', 0) * 100
        cagr_pct = metrics.get('cagr', 0) * 100
        vol_pct = metrics.get('volatility', 0) * 100
        sharpe_val = metrics.get('sharpe', 0)
        max_dd_pct = metrics.get('max_drawdown', 0) * 100

        valor_final = portfolio_value.iloc[-1]
        lucro = valor_final - initial_capital

        st.write("---")
        st.markdown("### Indicadores da Carteira")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retorno Acumulado", f"{final_return_pct:.2f}%")
            st.metric("CAGR", f"{cagr_pct:.2f}%")
        with col2:
            st.metric("Volatilidade (a.a.)", f"{vol_pct:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe_val:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{max_dd_pct:.2f}%")
            st.metric("Valor Final (R$)", f"{valor_final:,.2f}")

        st.write(f"Lucro: **R$ {lucro:,.2f}**")

        # Se quiser métricas para CDI e IBOV
        if not cdi_value.empty:
            cdi_ret = (cdi_value.iloc[-1] / cdi_value.iloc[0]) - 1
            cdi_ret_pct = cdi_ret * 100
            days_cdi = (cdi_value.index[-1] - cdi_value.index[0]).days
            years_cdi = days_cdi / 365.25
            cdi_cagr = ((1 + cdi_ret) ** (1 / years_cdi) - 1) * 100 if years_cdi > 0 else 0
            st.markdown(f"**CDI Retorno:** {cdi_ret_pct:.2f}% | **CAGR:** {cdi_cagr:.2f}%")

        if not ibov_value.empty:
            ibov_ret = (ibov_value.iloc[-1] / ibov_value.iloc[0]) - 1
            ibov_ret_pct = ibov_ret * 100
            days_ibov = (ibov_value.index[-1] - ibov_value.index[0]).days
            years_ibov = days_ibov / 365.25
            ibov_cagr = ((1 + ibov_ret) ** (1 / years_ibov) - 1) * 100 if years_ibov > 0 else 0
            st.markdown(f"**IBOV Retorno:** {ibov_ret_pct:.2f}% | **CAGR:** {ibov_cagr:.2f}%")

        st.success("Backtest finalizado com sucesso!")


if __name__ == "__main__":
    main()
