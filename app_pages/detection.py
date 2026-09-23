import altair as alt
import streamlit as st
from ui import header, source_image, infer, png_bytes, download_csv, teaching_prompt, runtime_choice

header("02", "Detectar é localizar e classificar", "**YOLO11n · pesos COCO.** Ajuste o limiar e observe o compromisso entre recuperar objetos e admitir detecções frágeis.")
image, data = source_image("det")
if image is None:
    st.stop()
with st.form("detection_controls"):
    a, b, c = st.columns(3)
    conf = a.slider("Confiança mínima", .05, .95, .25, .05)
    size = b.select_slider("Resolução de inferência", options=[320, 480, 640, 960, 1280, 1920], value=640)
    with c:
        device = runtime_choice("det_device")
    submitted = st.form_submit_button("Executar YOLO", type="primary", icon=":material/play_arrow:")
output = st.container()
if submitted:
    with st.spinner("Executando o detector localmente…"):
        result = infer(data, conf, size, device)
        st.session_state.detection_result = {"result": result, "data": data, "conf": conf, "size": size}
saved = st.session_state.detection_result
with output:
    if saved and saved["data"] == data:
        result = saved["result"]
        st.caption(f"Resultado calculado: confiança ≥ {saved['conf']:.2f} · imagem de entrada {saved['size']} px. Execute novamente após alterar parâmetros.")
        a, b = st.columns(2)
        a.image(image, caption="Entrada", width="stretch")
        b.image(result["image"], caption="Saída do modelo", width="stretch")
        df = result["detections"]
        with st.container(horizontal=True):
            st.metric("Objetos detectados", len(df), border=True)
            st.metric("Classes", df.class_name.nunique() if len(df) else 0, border=True)
            st.metric("Tempo da chamada", f"{result['inference_ms']:.0f} ms", border=True)
        if len(df):
            counts = df.groupby("class_name").size().reset_index(name="Quantidade")
            chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=5).encode(x=alt.X("Quantidade:Q"), y=alt.Y("class_name:N", title="Classe", sort="-x"), tooltip=["class_name", "Quantidade"]).properties(height=220)
            st.altair_chart(chart, width="stretch")
        st.dataframe(df, hide_index=True)
        with st.container(horizontal=True):
            st.download_button("Baixar imagem anotada", png_bytes(result["image"]), "deteccao_yolo11n.png", "image/png", icon=":material/download:")
            download_csv(df, "deteccoes.csv", "det_csv")
    else:
        st.image(image, caption="Exemplo pronto para inferência. Clique em Executar YOLO.", width="stretch")
teaching_prompt("Uma confiança de 0,80 significa 80% de chance de acerto? Por que contar caixas não mede a precisão do sistema?")
with st.expander("Como o resultado foi calculado"):
    st.code("model.predict(image, conf=limiar, imgsz=resolucao, device=device)\n# Caixas, classe e score são saídas reais do modelo.", language="python")
    st.markdown("Sem caixas de referência anotadas, esta página não calcula mAP, precisão ou recall. O tempo exibido mede a chamada predict, incluindo pré/pós-processamento e eventual inicialização na primeira execução; não inclui renderização da interface ou escrita em disco. Na vista aérea, tente 1920 px para discutir objetos pequenos e custo computacional.")
