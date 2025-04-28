import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

PROBS = [0.4, 0.5, 0.6]
N_ARMS = len(PROBS)
MULTIPLICADOR_VISITANTES = 1000  # Fixo, sem sidebar

class ThompsonSamplingMAB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        samples = [np.random.beta(self.successes[i] + 1, self.failures[i] + 1) for i in range(self.n_arms)]
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.counts[arm] += 1
        if reward:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
        self.values[arm] = self.successes[arm] / self.counts[arm] if self.counts[arm] > 0 else 0

st.set_page_config(page_title="MAB Simulador", layout="centered")
st.title("🎰 Simulador Multi-Armed Bandit (MAB) - Thompson Sampling")

# Introdução didática em um expander
with st.expander("Como funciona este simulador? (clique para ver/ocultar)"):
    st.markdown("""
    Imagine que você é responsável por um site e quer descobrir qual de três opções (por exemplo, três banners, layouts ou promoções) gera mais cliques ou conversões.

    Cada botão abaixo representa uma dessas opções. A cada clique, você simula a escolha de um visitante do site. O algoritmo **Multi-Armed Bandit** (MAB) vai aprendendo, com base nos resultados dos cliques, qual opção parece ser a mais eficiente.

    O modelo equilibra **exploração** (testar todas as opções para aprender) e **aproveitamento** (focar na opção que está performando melhor até agora).

    Os gráficos mostram, de forma visual, como o modelo distribui o tráfego entre as opções e como ele aprende ao longo do tempo.
    **Cada clique simula 1000 visitantes para deixar a visualização mais interessante!**

    _Observação:_ Conforme o número de cliques aumenta, a **taxa de sucesso estimada** de cada opção tende a se estabilizar. Isso significa que, quanto mais dados o modelo coleta, mais difícil fica mudar a estimativa — afinal, ele está mais confiante sobre o desempenho de cada opção. Por isso, mudanças bruscas nas taxas são mais comuns no início, quando há poucos dados, e vão ficando raras à medida que o experimento avança.
    """)

if 'mab' not in st.session_state:
    st.session_state.mab = ThompsonSamplingMAB(N_ARMS)
    st.session_state.history = []  # (botao, reward)
    st.session_state.cumulative_rewards = []

if st.button("Resetar experimento"):
    st.session_state.mab = ThompsonSamplingMAB(N_ARMS)
    st.session_state.history = []
    st.session_state.cumulative_rewards = []

# Sugestão do modelo
suggested = st.session_state.mab.select_arm()
taxa = st.session_state.mab.values[suggested]
st.markdown(f"<span style='color:lime;font-weight:bold'>Sugestão do modelo (Thompson): Botão {suggested+1} (taxa estimada: {taxa:.2f})</span>", unsafe_allow_html=True)

st.markdown("Clique em qualquer botão abaixo para simular um visitante:")

cols = st.columns(N_ARMS)
clicked = None
for i, col in enumerate(cols):
    if col.button(f"Botão {i+1}"):
        clicked = i

if clicked is not None:
    reward = int(np.random.rand() < PROBS[clicked])
    st.session_state.mab.update(clicked, reward)
    st.session_state.history.append((clicked, reward))
    total_rewards = sum([h[1] for h in st.session_state.history])
    st.session_state.cumulative_rewards.append(
        total_rewards / len(st.session_state.history)
    )

# Gráfico de área empilhada: escolhas por rodada (visual multiplicado)
st.markdown("#### Distribuição de tráfego (escolhas do modelo) ao longo do tempo")
if len(st.session_state.history) > 0:
    df = pd.DataFrame(0, index=np.arange(len(st.session_state.history)), columns=[f"Botão {i+1}" for i in range(N_ARMS)])
    for idx, (botao, _) in enumerate(st.session_state.history):
        df.iloc[idx, botao] = 1
    df_cumsum = df.cumsum()
    x_vis = (df_cumsum.index + 1) * MULTIPLICADOR_VISITANTES
    fig_area = go.Figure()
    for col in df.columns[::-1]:
        fig_area.add_trace(go.Scatter(
            x=x_vis,
            y=df_cumsum[col] * MULTIPLICADOR_VISITANTES,
            mode='lines',
            stackgroup='one',
            name=col
        ))
    fig_area.update_layout(
        xaxis_title="Visitantes simulados",
        yaxis_title="Tráfego acumulado",
        legend_title="Botão",
        height=350,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_area, use_container_width=True)

# Gráfico de linha: evolução da taxa de sucesso média (visual multiplicado)
if st.session_state.cumulative_rewards:
    st.markdown("#### Evolução da taxa de sucesso média")
    x_cum = np.arange(1, len(st.session_state.cumulative_rewards) + 1) * MULTIPLICADOR_VISITANTES
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=x_cum,
        y=st.session_state.cumulative_rewards,
        mode='lines+markers',
        name='Taxa média'
    ))
    fig3.update_layout(
        xaxis_title="Visitantes simulados",
        yaxis_title="Taxa de sucesso média",
        yaxis=dict(range=[0, 1]),
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)


# Gráfico de barras: taxa de sucesso estimada
st.markdown("#### Taxa de sucesso estimada por botão")
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=[f"Botão {i+1}" for i in range(N_ARMS)],
    y=st.session_state.mab.values,
    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
))
fig2.update_layout(
    yaxis=dict(range=[0, 1]),
    height=300,
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig2, use_container_width=True)

# Tabela de dados simulados por botão
st.markdown("#### Dados simulados por botão")
dados = {
    "Botão": [f"Botão {i+1}" for i in range(N_ARMS)],
    "Visitantes simulados": [int(st.session_state.mab.counts[i] * MULTIPLICADOR_VISITANTES) for i in range(N_ARMS)],
    "Sucessos simulados": [int(st.session_state.mab.successes[i] * MULTIPLICADOR_VISITANTES) for i in range(N_ARMS)],
    "Taxa de sucesso estimada": [f"{st.session_state.mab.values[i]:.2%}" for i in range(N_ARMS)]
}
st.table(pd.DataFrame(dados))

with st.expander("Como funciona o Thompson Sampling?"):
    st.markdown("""
    **O que é este simulador?**

    - Cada clique simula a escolha de um botão por 1000 visitantes fictícios.
    - O objetivo do modelo é aprender, ao longo do tempo, qual botão tem maior chance de sucesso.
    - O algoritmo Thompson Sampling mantém uma distribuição de probabilidade (Beta) para cada botão, baseada nos sucessos e fracassos observados.
    - A cada rodada, ele sorteia uma probabilidade para cada botão e escolhe aquele com maior valor sorteado, equilibrando exploração e aproveitamento.
    - Isso faz com que o modelo aprenda de forma mais estável e realista, mesmo com poucos dados.
    - Os gráficos mostram como o tráfego e as taxas de sucesso evoluem conforme o modelo aprende.
    - Os números exibidos são multiplicados por 1000 apenas para visualização, mas o aprendizado do modelo é feito clique a clique.

    **Sobre a estabilização das taxas:**
    Conforme o número de cliques aumenta, a taxa de sucesso estimada de cada opção tende a se estabilizar. Isso acontece porque, com mais dados, o modelo fica mais confiante sobre o desempenho de cada botão. Mudanças bruscas nas taxas são comuns no início, mas vão ficando raras à medida que o experimento avança.
    """)

# Rodapé com LinkedIn e autoria (ajuste o link abaixo para o seu repositório)
st.markdown("""
<br>
<hr style="border:1px solid #444;">
<div style="display:flex; align-items:center; justify-content:center;">
    <img src="https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/LinkedIn_logo_initials.png" width="32" style="margin-right:10px; vertical-align:middle;">
    <a href="https://www.linkedin.com/in/lucas-barbosa-a00302167/" target="_blank" style="font-size:17px; text-decoration:none; color:#0e76a8; font-weight:bold;">
        Desenvolvido por Lucas Barbosa
    </a>
</div>
""", unsafe_allow_html=True)