import streamlit as st

# Título do app
st.title("Minha Aplicação de Backtesting")

# Mensagem inicial
st.write("Olá, mundo! Este é um aplicativo de backtesting usando Streamlit.")

# Entrada de dados
ticker = st.text_input("Digite o ticker da ação (exemplo: PETR4)")

data_inicial = st.date_input("Data inicial")
data_final = st.date_input("Data final")

if st.button("Executar Backtest"):
    st.write(f"Executando backtest para {ticker} de {data_inicial} até {data_final}")
    # Aqui você pode incluir sua lógica de backtesting
