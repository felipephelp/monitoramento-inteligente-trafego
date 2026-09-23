import altair as alt
from PIL import ImageDraw
import streamlit as st
from ui import header, source_image, infer, occlude, download_csv, teaching_prompt, png_bytes

header("05", "O que muda quando o objeto fica oculto?", "Cubra uma fração crescente da caixa e execute **novas inferências reais**. A curva revela a estabilidade desta detecção sob uma perturbação controlada.")
image,data=source_image("occ",default="Cena urbana")
if image is None:
    st.stop()
conf=st.slider("Confiança mínima para considerar detectado",.05,.75,.25,.05,key="occ_conf")
with st.spinner("Localizando objetos para o experimento…"):
    baseline=infer(data,conf,640)
df=baseline["detections"]
if df.empty:
    st.image(image,width="stretch")
    st.warning("Nenhum objeto detectado neste limiar. Diminua a confiança ou escolha outra imagem.")
    st.stop()
options=[f"{i+1} · {r.class_name} · confiança {r.confidence:.2f}" for i,r in df.iterrows()]
index=st.selectbox("Objeto a encobrir",range(len(df)),format_func=lambda i:options[i],key="occ_target")
box=tuple(float(df.iloc[index][k]) for k in ["x1","y1","x2","y2"])
preview=image.copy()
draw=ImageDraw.Draw(preview)
draw.rectangle(box,outline="#E07516",width=5)
left,right=st.columns([1.6,1])
with left:
    st.image(preview,caption="Caixa de referência selecionada",width="stretch")
with right:
    st.subheader("Hipótese antes de executar")
    st.markdown("A detecção desaparecerá gradualmente? Ou o score cairá e voltará a subir?")
    st.markdown("**0 → 20 → 40 → 60 → 80%**\n\nA máscara cobre a caixa da esquerda para a direita. O cenário fora dela permanece igual.")
    execute=st.button("Executar curva de oclusão",type="primary",icon=":material/visibility_off:",key="occ_execute")
if execute:
    with st.spinner("Executando os cinco níveis de oclusão…"):
        result=occlude(data,box,conf)
        st.session_state.occlusion_result={"result":result,"data":data,"box":box,"conf":conf}
saved=st.session_state.occlusion_result
if saved and saved["data"]==data and saved["box"]==box and saved["conf"]==conf:
    result=saved["result"]
    metrics=result["metrics"].copy()
    metrics["Oclusão (%)"]=metrics.level*100
    st.subheader("Resposta observada do modelo")
    chart=alt.Chart(metrics).mark_line(point=alt.OverlayMarkDef(size=90),strokeWidth=3).encode(
        x=alt.X("Oclusão (%):Q",scale=alt.Scale(domain=[0,80])),
        y=alt.Y("confidence:Q",title="Confiança do objeto correspondente",scale=alt.Scale(domain=[0,1])),
        tooltip=["Oclusão (%)","confidence","matched","iou"]).properties(height=300)
    st.altair_chart(chart,width="stretch")
    selected=st.select_slider("Inspecionar nível",options=[0,20,40,60,80],value=40,key="occ_level")
    i=[0,20,40,60,80].index(selected)
    st.image(result["images"][i],caption=f"Saída real com {selected}% da largura da caixa ocultada",width="stretch")
    st.dataframe(metrics,hide_index=True)
    with st.container(horizontal=True):
        download_csv(metrics,"experimento_oclusao.csv","occ_csv")
        st.download_button("Baixar quadro",png_bytes(result["images"][i]),f"oclusao_{selected}.png","image/png",icon=":material/download:")
    st.caption("Correspondência: mesma classe e IoU ≥ 0,30 em relação à caixa original. Zero indica ausência de detecção correspondente acima do limiar, não score bruto zero da rede.")
teaching_prompt("A máscara representa fielmente chuva, árvore ou outro veículo? Que outros testes são necessários para afirmar robustez na rua?")
with st.expander("Limites e interpretação"):
    st.markdown("A referência é a detecção original do próprio modelo, não uma anotação humana. Portanto a curva mede retenção e score sob máscara, não recall ou mAP. Ela pode não ser monotônica: o contexto e a caixa predita mudam. O experimento de tracking permite estudar a continuidade dos IDs através de uma obstrução em vídeo.")
