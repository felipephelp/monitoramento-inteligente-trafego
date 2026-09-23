import platform
import pandas as pd
import streamlit as st
from lessons import SOURCES, LESSON_PLAN, GLOSSARY
from ui import header, ROOT, ASSETS

header("07", "Guia de aula e fontes", "Um percurso de 60 minutos com momentos de previsão, execução e interpretação dos resultados.")
st.dataframe(pd.DataFrame(LESSON_PLAN,columns=["Tempo","Bloco","Atividade","Página"]),hide_index=True)
with st.expander("Glossário para acompanhar a demonstração",expanded=True):
    st.table(GLOSSARY)
st.subheader("Literatura, documentação e produtos")
for title,url,description in SOURCES:
    st.markdown(f"[{title}]({url}) — {description}")
st.subheader("Materiais do apresentador")
st.markdown("As figuras identificadas como IPT e defesa foram extraídas dos arquivos fornecidos por Felipe Euphrasio. O conteúdo didático usa essas figuras como referência, mas as inferências do aplicativo vêm dos pesos públicos explicitamente identificados.")
attribution=ASSETS/"ATRIBUICOES.md"
if attribution.exists():
    with st.expander("Origem e licenças dos exemplos"):
        st.markdown(attribution.read_text(encoding="utf-8"))
st.subheader("Protocolo mínimo para um piloto")
st.markdown("Selecione um corredor e 3 eventos prioritários. Anote amostras independentes de dia, noite e chuva; separe dias e câmeras entre treino e teste. Compare erros de contagem, eventos perdidos, falsos alarmes por câmera/hora e latência. Identifique quem revisa o alerta e qual ação ele desencadeia.")
with st.expander("Ambiente local e reprodução"):
    import torch, ultralytics
    st.table({"Python":platform.python_version(),"Streamlit":st.__version__,"Ultralytics":ultralytics.__version__,"PyTorch":torch.__version__,"CUDA disponível":str(torch.cuda.is_available()),"Pasta":str(ROOT)})
    st.code(".venv\\Scripts\\python.exe prepare_examples.py\n.venv\\Scripts\\python.exe -m streamlit run app.py",language="powershell")
    st.markdown("O servidor usa apenas 127.0.0.1. Imagens e vídeos enviados são processados localmente. Uploads de vídeo são mantidos em uploads/; execuções ficam em results/. Pesos e exemplos preparados permitem repetir a aula sem internet.")
