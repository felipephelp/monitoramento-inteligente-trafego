import streamlit as st

from ui import EXPORTS, header, teaching_prompt


header("08", "Aplicações do monitoramento inteligente", "Cada aplicação combina evidências diferentes. A mesma caixa de detecção não responde sozinha a perguntas sobre fluxo, identidade ou comportamento.")

view = st.segmented_control(
    "Aplicação",
    ["Detecção", "Tracking", "Contagem", "Re-ID", "Comportamento", "Anomalias"],
    default="Detecção",
    key="application_view",
)

content = {
    "Detecção": ("detection_before_after.png", "Localiza veículos, pedestres e outros objetos por quadro. Serve como entrada para as etapas temporais."),
    "Tracking": ("tracking_trajectories.png", "Associa detecções ao longo do vídeo e produz IDs locais, trajetórias e tempos de permanência."),
    "Contagem": ("tracking_preview.png", "Registra cruzamentos de linha ou transições entre zonas por classe e direção."),
    "Re-ID": ("reid_gallery.png", "Ordena candidatos por aparência entre observações. Tempo, rota e câmera ajudam a rejeitar pares impossíveis."),
    "Comportamento": ("tracking_comparison.png", "Interpreta trajetórias para reconhecer parada, retorno, conversão, permanência ou circulação na contramão."),
    "Anomalias": ("tracking_comparison.png", "Compara eventos com regras ou padrões esperados. Exemplos: veículo parado, trajeto incomum, invasão de área e fluxo fora do padrão."),
}
filename, explanation = content[view]
path = EXPORTS / filename
if path.exists():
    st.image(str(path), width="stretch")
st.markdown(f"### {view}")
st.markdown(explanation)

if view in {"Comportamento", "Anomalias"}:
    a, b = st.columns(2)
    with a.container(border=True):
        st.subheader("Regras sobre trajetórias")
        st.markdown("Zonas, linhas, direção, duração e velocidade estimada produzem eventos explicáveis. A geometria precisa estar calibrada quando a regra usa metros ou km/h.")
    with b.container(border=True):
        st.subheader("Modelos de padrão")
        st.markdown("Um modelo aprende sequências frequentes e sinaliza desvios. O operador ainda precisa distinguir risco real, mudança da via e erro de percepção.")

st.subheader("Da observação à aplicação")
st.table({
    "Detecção": "Objeto presente neste quadro.",
    "Tracking": "Movimento e continuidade dentro do vídeo.",
    "Contagem": "Evento de passagem por linha ou zona.",
    "Re-ID": "Candidatos semelhantes entre observações.",
    "Comportamento": "Regra ou classe construída sobre a trajetória.",
    "Anomalia": "Desvio de uma regra ou de um padrão validado.",
})
teaching_prompt("Uma trajetória incomum representa uma infração, uma emergência, uma obra na via ou uma falha do tracker? Que evidência permite separar essas hipóteses?")
