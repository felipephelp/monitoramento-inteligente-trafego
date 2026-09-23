import streamlit as st
from ui import header, figure, teaching_prompt

header("06", "Da pesquisa à operação", "Conecte os modelos às figuras dos materiais originais, à geometria da cena e aos sistemas de vídeo analítico.")
view=st.segmented_control("Explorar",["Modelos","Geometria","Arquitetura","Produtos"],default="Modelos",key="research_view")
if view=="Modelos":
    a,b=st.columns(2)
    with a:
        st.subheader("CNN: padrões locais e hierarquia")
        figure("ipt_cnn.png","Figura do material IPT · arquitetura CNN, slide 12.")
        st.markdown("Convoluções aprendem padrões visuais em várias escalas. YOLO11 combina extração de características, fusão de escalas e uma cabeça que estima caixas e classes.")
    with b:
        st.subheader("Transformers: relações por atenção")
        figure("ipt_transformer.png","Figura do material IPT · Vision Transformer, slide 13.")
        st.markdown("ViT representa patches; DETR usa atenção e consultas para predizer conjuntos de objetos. Atenção pode modelar relações, mas eventos de trânsito exigem dados e objetivos próprios.")
    st.table({"Detecção":"Caixas + classes por quadro; demonstrada com YOLO11n.","Tracking":"Associação temporal; demonstrada com ByteTrack e BoT-SORT.","Re-ID":"Descritores de aparência e ranking; demonstrado com ResNet18 genérico.","Eventos":"Regras e/ou modelos sobre trajetórias, contexto, tempo e geometria."})
elif view=="Geometria":
    a,b=st.columns([1.5,1])
    with a:
        figure("tese_homografia.png","Homografia e planos · defesa de doutorado, slide 38.")
    with b:
        st.subheader("Pixels precisam de escala")
        st.markdown("Velocidade e distância demandam calibração. Uma homografia aproxima a relação entre a imagem e o plano do pavimento a partir de pontos correspondentes.")
        st.latex(r"s\begin{bmatrix}u\\v\\1\end{bmatrix}=H\begin{bmatrix}X\\Y\\1\end{bmatrix}")
        st.markdown("A projeção assume uma superfície aproximadamente plana. Use o ponto de contato com o solo; relevo, altura do objeto e vibração da câmera introduzem erro.")
    teaching_prompt("Dois carros percorrem o mesmo número de pixels perto e longe da câmera. Eles percorreram a mesma distância em metros?")
elif view=="Arquitetura":
    st.mermaid_chart("""flowchart LR
        A[Câmeras / drones] --> B[Borda<br/>decodificação + YOLO]
        B --> C[Tracking + regras]
        C --> D[Metadados<br/>trajetórias + eventos]
        D --> E[Centro de operação]
        C --> F[Vídeo de evidência]
        F --> E
        E --> G[Revisão humana<br/>priorização + resposta]
        G --> H[Dados anotados<br/>validação + melhoria]
        H --> B
    """)
    figure("tese_fluxo_sistema.png","Arquitetura original de sistema integrado · defesa de doutorado, slides 42–43.")
    st.markdown("**Do protótipo ao piloto:** defina pontos de observação, relógios sincronizados, qualidade do vídeo, critérios de alerta e registro das decisões. A revisão humana também produz exemplos de falha para melhorar a validação.")
else:
    a,b=st.columns(2)
    with a.container(border=True):
        st.badge("Plataforma de desenvolvimento",color="blue")
        st.subheader("NVIDIA Metropolis / DeepStream")
        st.markdown("Blocos para construir e integrar pipelines de vídeo analítico em borda e servidores. O desempenho depende de hardware, codecs, modelos e número de fluxos.")
        st.link_button("Documentação oficial","https://developer.nvidia.com/metropolis")
    with b.container(border=True):
        st.badge("Análise embarcada em câmeras",color="green")
        st.subheader("AXIS Object Analytics")
        st.markdown("Detecção, classificação, tracking e contagem de pessoas e veículos em dispositivos compatíveis, com integração a sistemas de gestão de vídeo.")
        st.link_button("Produto e capacidades","https://www.axis.com/products/axis-object-analytics")
    st.caption("Descrições resumidas de páginas oficiais consultadas em setembro de 2026. São referências de mercado; o laboratório não integra nem executa esses produtos comerciais.")
    st.subheader("Critérios para comparar uma solução")
    st.table({"Cenário":"Câmera fixa ou móvel, altura, classes, noite e chuva.","Qualidade":"Erro de contagem, falsos alarmes por câmera/hora, atraso do alerta.","Integração":"APIs, metadados, retenção de evidência, exportação e auditoria.","Custo operacional":"Licenças, hardware, armazenamento, manutenção e recalibração."})
