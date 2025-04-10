import streamlit as st
import pandas as pd
import numpy as np
import io

# Importações de módulos internos do projeto
from src.data_loader import get_all_tickers, load_filtered_data, load_cdi, load_ibov
from src.backtesting import run_backtest

# Caminhos dos arquivos de dados
PARQUET_PATH_PRECOS = "data/precos.parquet"
PARQUET_PATH_CDI    = "data/cdi.parquet"
PARQUET_PATH_IBOV   = "data/ibov.parquet"

# Aplicativo Streamlit de Backtesting (Ações BR) - Versão Aprimorada
# Funcionalidades adicionadas:
# - Gráficos interativos (Plotly) de desempenho acumulado, volatilidade móvel e drawdown.
# - Entrada de estratégia personalizada via código Python sandbox (seguro), gerando alocação por ativo.
# - Tratamento automático de intervalos de dados parciais (alinhar datas de início/término comuns).
# - Tabela comparativa de performance (Retorno, CAGR, Volatilidade, Sharpe, Drawdown) para carteira vs. ativos.
# - Exportação de dados e gráficos (download de CSV/Excel e imagens PNG dos gráficos).

# Configuração da página Streamlit (título e layout)
st.set_page_config(page_title="Backtest Dinâmico - Ações BR", layout="wide")

# Título principal
st.title("Backtest com Comparação CDI e IBOV")

# Barra lateral - Parâmetros gerais
st.sidebar.header("Parâmetros de Data e Capital")
start_date = st.sidebar.date_input("Data Início", pd.to_datetime("2012-01-02"))
end_date   = st.sidebar.date_input("Data Fim", pd.to_datetime("2020-01-02"))
initial_capital = st.sidebar.number_input("Capital Inicial (R$)", min_value=1000, value=100000, step=1000)
risk_free = st.sidebar.number_input("Taxa Livre de Risco (Ex.: 0.13 = 13%)", 0.0, 1.0, 0.13, 0.01)

# Criação das abas de seleção e resultados
tab_select, tab_result = st.tabs(["Seleção de Ativos", "Resultados do Backtest"])

# ---------------------------------------------
# Aba 1: Seleção de Ativos ou Código Personalizado
# ---------------------------------------------
with tab_select:
    st.subheader("Configuração da Estratégia")
    # Opção de seleção de estratégia: Manual ou Código Personalizado
    strategy_mode = st.radio("Modo de Seleção da Estratégia:", options=["Manual", "Código Personalizado"], index=0)
    
    # Inicializa listas de tickers e pesos (para garantir existência das variáveis)
    buy_selection = []
    buy_weights = []
    sell_selection = []
    sell_weights = []
    
    if strategy_mode == "Manual":
        # Carregar todos os tickers disponíveis
        tickers_all = get_all_tickers(PARQUET_PATH_PRECOS)
        st.write(f"Total de tickers disponíveis: {len(tickers_all)}")
        
        # Seleção de ativos para posição comprada (long)
        st.markdown("#### Tickers para Comprar (Long)")
        buy_selection = st.multiselect("Selecione ativos para **COMPRA**", tickers_all, key="buy_select")
        equal_weights_buy = st.checkbox("Usar pesos iguais para compra", value=False, key="chk_buy")
        
        buy_weights = []
        if buy_selection:
            if equal_weights_buy:
                peso = 1.0 / len(buy_selection)
                st.info(f"Pesos iguais de {peso:.2f} para cada ativo de COMPRA.")
                buy_weights = [peso] * len(buy_selection)
            else:
                st.write("Defina os pesos manualmente para cada ativo de **compra**:")
                for tk in buy_selection:
                    w = st.number_input(f"Peso de {tk} (compra)", 0.0, 1.0, 0.10, 0.01, key=f"buy_{tk}")
                    buy_weights.append(w)
        
        # Seleção de ativos para posição vendida (short)
        st.markdown("#### Tickers para Vender (Short)")
        sell_selection = st.multiselect("Selecione ativos para **VENDA**", tickers_all, key="sell_select")
        equal_weights_sell = st.checkbox("Usar pesos iguais para venda", value=False, key="chk_sell")
        
        sell_weights = []
        if sell_selection:
            if equal_weights_sell:
                peso = 1.0 / len(sell_selection)
                st.info(f"Pesos iguais de {peso:.2f} para cada ativo de VENDA.")
                sell_weights = [peso] * len(sell_selection)
            else:
                st.write("Defina os pesos manualmente para cada ativo de **venda**:")
                for tk in sell_selection:
                    w = st.number_input(f"Peso de {tk} (venda)", 0.0, 1.0, 0.10, 0.01, key=f"sell_{tk}")
                    sell_weights.append(w)
    
    else:
        # Entrada de código Python personalizado
        st.markdown("#### Código da Estratégia Personalizada")
        st.write("Insira o código Python que retorna um DataFrame com colunas **'Ticker'** e **'Weight'** (peso).")
        code_input = st.text_area("Código da estratégia:", 
            value="""import pandas as pd

# Exemplo: 50% PETR4 e 50% VALE3
df = pd.DataFrame({
    'Ticker': ['PETR4.SA', 'VALE3.SA'],
    'Weight': [0.5, 0.5]
})
df""", height=200, key="custom_code")
    
    # Botão para executar o backtest (com base no modo selecionado)
    if st.button("Executar Backtest!"):
        if strategy_mode == "Manual":
            # Modo manual: usar seleções feitas acima
            all_tickers_needed = list(set(buy_selection + sell_selection))
        else:
            # Modo código personalizado: executar o código do usuário de forma segura
            safe_globals = {"pd": pd, "np": np, "__builtins__": None}
            local_vars = {}
            try:
                exec(code_input, safe_globals, local_vars)
            except Exception as e:
                st.error(f"Erro ao executar o código personalizado: {e}")
                st.stop()
            # Verifica se 'df' foi definido no código do usuário
            df_weights = None
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, pd.DataFrame):
                    df_weights = var_value
                    break
            if df_weights is None:
                st.error("O código personalizado não definiu um DataFrame de resultado.")
                st.stop()
            # Verifica se colunas requeridas existem
            if not {'Ticker', 'Weight'}.issubset(df_weights.columns):
                st.error("O DataFrame retornado deve conter colunas 'Ticker' e 'Weight'.")
                st.stop()
            # Separa em listas de compra (peso positivo) e venda (peso negativo)
            buy_selection = df_weights[df_weights['Weight'] > 0]['Ticker'].tolist()
            buy_weights = df_weights[df_weights['Weight'] > 0]['Weight'].tolist()
            sell_selection = df_weights[df_weights['Weight'] < 0]['Ticker'].tolist()
            sell_weights = (-df_weights[df_weights['Weight'] < 0]['Weight']).tolist()  # torna peso positivo para short
            # Guarda o DataFrame de estratégia customizada no estado (para exportação/exibição posterior)
            st.session_state['custom_strategy_df'] = df_weights.copy()
            all_tickers_needed = list(set(buy_selection + sell_selection))
        
        # Caso nenhum ticker selecionado (ou DataFrame vazio)
        if not all_tickers_needed:
            st.error("Nenhum ticker válido foi especificado na estratégia.")
        else:
            # Carrega dados de preços filtrados para os tickers e período selecionado
            df_prices = load_filtered_data(PARQUET_PATH_PRECOS, all_tickers_needed, str(start_date), str(end_date))
            if df_prices.empty:
                st.warning("Não há dados disponíveis para os tickers/período selecionados.")
            else:
                # Tratamento de dados parciais: determinar período comum entre os ativos
                first_dates = {}
                last_dates = {}
                for tk in df_prices.columns:
                    series = df_prices[tk].dropna()
                    if series.empty:
                        continue
                    first_dates[tk] = series.index[0]
                    last_dates[tk] = series.index[-1]
                if not first_dates:
                    st.warning("Dados dos ativos selecionados não encontrados ou incompletos.")
                else:
                    common_start = max(first_dates.values())
                    common_end = min(last_dates.values())
                    # Filtra o DataFrame de preços para o período comum
                    df_prices = df_prices[(df_prices.index >= common_start) & (df_prices.index <= common_end)].copy()
                    if df_prices.empty or common_start > common_end:
                        st.warning("Os ativos selecionados não possuem período de dados em comum.")
                    else:
                        # Execução do backtest usando o motor existente
                        result = run_backtest(
                            df_prices=df_prices,
                            buy_tickers=buy_selection,
                            buy_weights=buy_weights,
                            sell_tickers=sell_selection,
                            sell_weights=sell_weights,
                            start_date=str(common_start.date()),
                            end_date=str(common_end.date()),
                            risk_free_annual=risk_free
                        )
                        portfolio_curve = result.get("portfolio_curve", pd.Series(dtype=float))
                        metrics = result.get("metrics", {})
                        # Verifica resultado do backtest
                        if portfolio_curve.empty or not metrics:
                            st.warning("O backtest não retornou resultados (portfólio vazio).")
                        else:
                            # Calcula valor da carteira ao longo do tempo (multiplica pelo capital inicial)
                            portfolio_value = portfolio_curve * float(initial_capital)
                            # Carrega séries de referência (CDI e IBOV) para o mesmo período
                            cdi_series = load_cdi(PARQUET_PATH_CDI, str(common_start.date()), str(common_end.date()))
                            ibov_series = load_ibov(PARQUET_PATH_IBOV, str(common_start.date()), str(common_end.date()))
                            # Ajusta CDI e IBOV para capital inicial
                            cdi_value = cdi_series * float(initial_capital) if not cdi_series.empty else pd.Series(dtype=float)
                            ibov_value = ibov_series * float(initial_capital) if not ibov_series.empty else pd.Series(dtype=float)
                            # Prepara DataFrame para gráfico de performance acumulada
                            df_performance = pd.DataFrame({"Carteira": portfolio_value})
                            if not cdi_value.empty:
                                df_performance["CDI"] = cdi_value
                            if not ibov_value.empty:
                                df_performance["IBOV"] = ibov_value
                            # Calcula métricas por ativo individual
                            metrics_rows = []
                            # Adiciona linha da carteira (consolidada)
                            port_final_ret = metrics.get('final_return', 0.0)
                            port_cagr = metrics.get('cagr', 0.0)
                            port_vol = metrics.get('volatility', 0.0)
                            port_sharpe = metrics.get('sharpe', 0.0)
                            port_max_dd = -metrics.get('max_drawdown', 0.0)
                            metrics_rows.append({
                                "Ativo": "Carteira",
                                "Retorno": port_final_ret,
                                "CAGR": port_cagr,
                                "Volatilidade": port_vol,
                                "Sharpe": port_sharpe,
                                "Max Drawdown": port_max_dd
                            })
                            # Calcula métricas para cada ativo selecionado
                            for tk in all_tickers_needed:
                                price = df_prices[tk].dropna()
                                if price.empty:
                                    continue
                                initial = price.iloc[0]
                                final = price.iloc[-1]
                                ret = final / initial - 1
                                # período em anos para CAGR
                                days = (price.index[-1] - price.index[0]).days
                                years = days / 365.25
                                cagr_val = ((1+ret) ** (1/years) - 1) if years > 0 else 0.0
                                # Retornos diários para volatilidade e Sharpe
                                daily_ret = price.pct_change().fillna(0)
                                vol_val = daily_ret.std() * np.sqrt(252)
                                sharpe_val = 0.0
                                if vol_val != 0:
                                    sharpe_val = (daily_ret.mean() * 252 - risk_free) / vol_val
                                # Max drawdown (positivo, em termos de proporção do pico)
                                running_max = price.cummax()
                                drawdown_series = price / running_max - 1
                                max_dd_val = -drawdown_series.min()
                                metrics_rows.append({
                                    "Ativo": tk,
                                    "Retorno": ret,
                                    "CAGR": cagr_val,
                                    "Volatilidade": vol_val,
                                    "Sharpe": sharpe_val,
                                    "Max Drawdown": max_dd_val
                                })
                            metrics_df = pd.DataFrame(metrics_rows)
                            metrics_df.set_index("Ativo", inplace=True)
                            # Ordena ativos (exceto carteira) por retorno decrescente
                            if "Carteira" in metrics_df.index:
                                assets_df = metrics_df.drop(index="Carteira")
                            else:
                                assets_df = metrics_df
                            assets_df = assets_df.sort_values(by="Retorno", ascending=False)
                            if "Carteira" in metrics_df.index:
                                metrics_df = pd.concat([metrics_df.loc[["Carteira"]], assets_df])
                            else:
                                metrics_df = assets_df
                            # Armazena resultados no estado para uso na aba de Resultados
                            st.session_state["last_run_data"] = {
                                "df_performance": df_performance,
                                "metrics_df": metrics_df,
                                "portfolio_metrics": metrics,
                                "portfolio_value": portfolio_value,
                                "cdi_value": cdi_value,
                                "ibov_value": ibov_value,
                                "common_start": common_start,
                                "common_end": common_end
                            }
                            st.session_state["last_strategy_mode"] = strategy_mode
                            st.session_state["execute_backtest"] = True
                            st.success("Backtest executado com sucesso! Vá para a aba 'Resultados do Backtest' para visualizar.")
    else:
        # Se o botão não foi clicado, garante que não executaremos o backtest automaticamente
        st.session_state["execute_backtest"] = False

# ---------------------------------------------
# Aba 2: Resultados do Backtest e Visualizações
# ---------------------------------------------
with tab_result:
    st.subheader("Resultados do Backtest")
    # Verifica se há resultados armazenados da última execução
    if not st.session_state.get("execute_backtest"):
        st.info("Configure os ativos/estratégia e clique em **Executar Backtest** na aba anterior.")
    else:
        data = st.session_state.get("last_run_data", None)
        if data is None:
            st.info("Nenhum resultado para exibir. Execute o backtest na aba anterior.")
        else:
            df_performance = data["df_performance"]
            metrics_df = data["metrics_df"]
            portfolio_metrics = data["portfolio_metrics"]
            portfolio_value = data["portfolio_value"]
            cdi_value = data["cdi_value"]
            ibov_value = data["ibov_value"]
            common_start = data["common_start"]
            common_end = data["common_end"]
            
            # Gráfico interativo de performance acumulada (Carteira vs CDI vs IBOV)
            import plotly.graph_objects as go
            fig_perf = go.Figure()
            for col in df_performance.columns:
                fig_perf.add_trace(go.Scatter(x=df_performance.index, y=df_performance[col], mode='lines', name=col))
            fig_perf.update_layout(title="Desempenho Acumulado da Carteira vs CDI vs IBOV",
                                   xaxis_title="Data", yaxis_title="Valor (R$)")
            st.plotly_chart(fig_perf, use_container_width=True)
            # Botão para baixar imagem do gráfico de performance
            try:
                img_bytes = fig_perf.to_image(format="png")
                st.download_button("Baixar Gráfico de Performance", img_bytes, "performance.png", "image/png")
            except Exception:
                # Se falhar (ex.: kaleido não instalado), não mostra botão
                pass
            
            # Indicadores resumidos da carteira
            final_return_pct = portfolio_metrics.get('final_return', 0) * 100
            cagr_pct = portfolio_metrics.get('cagr', 0) * 100
            vol_pct = portfolio_metrics.get('volatility', 0) * 100
            sharpe_ratio = portfolio_metrics.get('sharpe', 0)
            max_dd_pct = -portfolio_metrics.get('max_drawdown', 0) * 100
            final_value = float(portfolio_value.iloc[-1])
            profit_value = final_value - float(initial_capital)
            st.write('---')
            st.markdown('### Indicadores da Carteira')
            col1, col2, col3 = st.columns(3)
            col1.metric("Retorno Acumulado", f"{final_return_pct:.2f}%")
            col1.metric("CAGR", f"{cagr_pct:.2f}%")
            col2.metric("Volatilidade (a.a.)", f"{vol_pct:.2f}%")
            col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            col3.metric("Max Drawdown", f"{max_dd_pct:.2f}%")
            col3.metric("Valor Final (R$)", f"{final_value:,.2f}")
            st.write(f"Lucro: **R$ {profit_value:,.2f}**")
            
            # Cálculo e gráficos de volatilidade móvel e drawdown da carteira
            # Volatilidade móvel (janela de 60 dias)
            port_returns = portfolio_value.pct_change().fillna(0)
            rolling_vol = port_returns.rolling(window=60).std() * np.sqrt(252)
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name='Volatilidade (60d)'))
            fig_vol.update_layout(title="Volatilidade Histórica (janela 60 dias)", xaxis_title="Data", yaxis_title="Volatilidade Anualizada")
            # Drawdown ao longo do tempo
            running_max = portfolio_value.cummax()
            drawdown = portfolio_value / running_max - 1
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown'))
            fig_dd.update_layout(title="Drawdown (queda acumulada desde o pico)", xaxis_title="Data", yaxis_title="Drawdown (%)", yaxis_tickformat='%')
            # Exibe gráficos lado a lado
            col_a, col_b = st.columns(2)
            col_a.plotly_chart(fig_vol, use_container_width=True)
            col_b.plotly_chart(fig_dd, use_container_width=True)
            # Botões de download das imagens desses gráficos
            try:
                img_vol = fig_vol.to_image(format="png")
                col_a.download_button("Baixar Volatilidade (PNG)", img_vol, "volatilidade.png", "image/png")
                img_dd = fig_dd.to_image(format="png")
                col_b.download_button("Baixar Drawdown (PNG)", img_dd, "drawdown.png", "image/png")
            except Exception:
                pass
            
            # Gráficos de barras para melhores e piores desempenhos dos ativos
            asset_returns = metrics_df.drop(index="Carteira", errors='ignore')["Retorno"]
            asset_returns = asset_returns.sort_values(ascending=False)
            top_n = 5
            if len(asset_returns) <= top_n:
                top_assets = asset_returns
                bottom_assets = pd.Series(dtype=float)
            else:
                top_assets = asset_returns.head(top_n)
                bottom_assets = asset_returns.tail(top_n)
            if not top_assets.empty:
                fig_top = go.Figure([go.Bar(x=top_assets.index, y=top_assets.values, marker_color='green')])
                fig_top.update_layout(title=f"Top {len(top_assets)} Ativos - Retorno Acumulado", xaxis_title="Ativo", yaxis_title="Retorno (%)", yaxis_tickformat='%')
                st.plotly_chart(fig_top, use_container_width=True)
                try:
                    img_top = fig_top.to_image(format="png")
                    st.download_button(f"Baixar Top {len(top_assets)} (PNG)", img_top, "top_assets.png", "image/png")
                except Exception:
                    pass
            if not bottom_assets.empty:
                fig_bottom = go.Figure([go.Bar(x=bottom_assets.index, y=bottom_assets.values, marker_color='red')])
                fig_bottom.update_layout(title=f"Piores {len(bottom_assets)} Ativos - Retorno Acumulado", xaxis_title="Ativo", yaxis_title="Retorno (%)", yaxis_tickformat='%')
                st.plotly_chart(fig_bottom, use_container_width=True)
                try:
                    img_bottom = fig_bottom.to_image(format="png")
                    st.download_button(f"Baixar Piores {len(bottom_assets)} (PNG)", img_bottom, "bottom_assets.png", "image/png")
                except Exception:
                    pass
            
            # Tabela de desempenho comparativo (Carteira vs Ativos)
            styled_table = metrics_df.style.format({
                "Retorno": "{:.2%}",
                "CAGR": "{:.2%}",
                "Volatilidade": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}"
            }).background_gradient(cmap="Greens", subset=["Retorno", "CAGR", "Sharpe"]) \
              .background_gradient(cmap="Reds", subset=["Volatilidade", "Max Drawdown"])
            st.write("#### Desempenho da Carteira e Ativos Selecionados")
            st.dataframe(styled_table, use_container_width=True)
            
            # Se estratégia customizada, exibe também os pesos definidos
            if st.session_state.get("last_strategy_mode") == "Código Personalizado" and st.session_state.get("custom_strategy_df") is not None:
                st.write("#### Pesos da Estratégia Personalizada")
                st.dataframe(st.session_state["custom_strategy_df"], use_container_width=True)
            
            # Botões de download de dados e métricas
            perf_csv = df_performance.to_csv().encode('utf-8')
            st.download_button("Baixar Dados de Performance (CSV)", perf_csv, "desempenho_portfolio.csv", "text/csv")
            try:
                metrics_excel = io.BytesIO()
                with pd.ExcelWriter(metrics_excel, engine='openpyxl') as writer:
                    metrics_df.to_excel(writer, sheet_name="Desempenho")
                st.download_button("Baixar Tabela de Métricas (Excel)", metrics_excel.getvalue(), "metricas.xlsx", "application/vnd.ms-excel")
            except Exception:
                metrics_csv = metrics_df.to_csv().encode('utf-8')
                st.download_button("Baixar Tabela de Métricas (CSV)", metrics_csv, "metricas.csv", "text/csv")
            if st.session_state.get("custom_strategy_df") is not None:
                strat_csv = st.session_state["custom_strategy_df"].to_csv(index=False).encode('utf-8')
                st.download_button("Baixar Estratégia (CSV)", strat_csv, "estrategia_custom.csv", "text/csv")
