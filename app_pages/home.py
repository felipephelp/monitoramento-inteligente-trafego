import streamlit as st
from ui import header, EXAMPLES, EXPORTS, teaching_prompt

header("01", "Da imagem à decisão urbana", "Explore **detecção, tracking, contagem, reidentificação e oclusão** com imagens reais, modelos locais e parâmetros que você pode alterar.")
left, right = st.columns([1.45, 1], gap="large", vertical_alignment="center")
with left:
    img = EXPORTS / "detection_urban_traffic_frame.png"
    st.image(str(img if img.exists() else EXAMPLES / "urban_traffic_frame.jpg"), caption="Tráfego urbano · exemplo público Ultralytics. As caixas, quando presentes, são inferência do YOLO11n.", width="stretch")
with right:
    st.markdown("## Uma câmera. Quatro perguntas.")
    st.markdown("**O que está aqui?**\n\nO detector localiza veículos e pedestres em cada quadro.")
    st.markdown("**Como se move?**\n\nO tracker associa observações e mantém trajetórias temporárias.")
    st.markdown("**Quantos cruzaram?**\n\nUma linha virtual converte trajetórias em eventos de contagem.")
    st.markdown("**É o mesmo objeto?**\n\nO Re-ID ordena candidatos pela aparência e exige contexto para decidir.")
    if st.button("Começar pela detecção", type="primary", icon=":material/play_arrow:"):
        st.switch_page("app_pages/detection.py")
st.mermaid_chart("""flowchart LR
    A[Vídeo / imagem] --> B[Detecção<br/>classe + caixa]
    B --> C[Tracking<br/>ID + trajetória]
    C --> D[Contagem<br/>linha + direção]
    C --> E[Re-ID<br/>candidatos por aparência]
    D --> F[Operação<br/>validar + agir]
    E --> F
    style A fill:#E8F2FA,stroke:#005CA9
    style F fill:#D9F2EE,stroke:#007A6C
""")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.subheader("Experimente")
    st.markdown("Compare limiares, acompanhe IDs em vídeo e introduza oclusão controlada.")
with c2.container(border=True):
    st.subheader("Explique")
    st.markdown("Figuras da apresentação IPT e da defesa conectam a demonstração à metodologia.")
with c3.container(border=True):
    st.subheader("Exporte")
    st.markdown("Baixe imagens anotadas, rankings, trajetórias e tabelas para discutir evidências.")
teaching_prompt("Se um veículo desaparece atrás de uma árvore, que evidência permite afirmar que voltou com a mesma identidade?")
