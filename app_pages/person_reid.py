from io import BytesIO
import altair as alt
from PIL import Image
import streamlit as st

from ui import EXAMPLES, download_csv, header, infer, png_bytes, rank, runtime_choice, teaching_prompt
from vision import crop_class_detections, person_reid_challenges


header("06", "Re-ID de pessoas sob mudanças visuais", "Um baseline neural ordena candidatos por aparência. O experimento controla oclusão, rotação e iluminação sem afirmar identidade biométrica.")

source = Image.open(EXAMPLES / "bus.jpg").convert("RGB")
with st.spinner("Localizando pessoas na cena de exemplo…"):
    detected = infer(png_bytes(source), .20, 640, "cpu")
people = crop_class_detections(source, detected["detections"], "person")
if len(people) < 3:
    st.error("O exemplo precisa de ao menos três recortes de pessoas detectadas.")
    st.stop()

st.caption("Fonte pública Ultralytics bus.jpg. Os candidatos diferentes vêm da mesma imagem; as variações positivas derivam artificialmente do recorte de consulta.")
with st.form("person_reid_controls"):
    a, b, c = st.columns(3)
    rotation = a.slider("Rotação", -45, 45, 25, 5, key="person_rotation")
    occlusion = b.slider("Oclusão", 0., .80, .35, .05, key="person_occlusion")
    brightness = c.slider("Iluminação", .30, 1.20, .65, .05, key="person_brightness")
    device = runtime_choice("person_reid_device")
    execute = st.form_submit_button("Calcular ranking de pessoas", type="primary", icon=":material/group:")

challenge = person_reid_challenges(people[0], people[1:4], rotation, occlusion, brightness)
query, gallery, names, relations = challenge["query"], challenge["gallery"], challenge["names"], challenge["relations"]

left, right = st.columns([1, 3], vertical_alignment="top")
with left:
    st.subheader("Consulta")
    st.image(query, caption="Pessoa A · recorte de referência", width="stretch")
with right:
    st.subheader("Galeria e desafios")
    cols = st.columns(4)
    for i, (image, name) in enumerate(zip(gallery, names)):
        with cols[i % 4]:
            st.image(image, caption=name, width="stretch")

signature = (rotation, occlusion, brightness, device)
if execute:
    with st.spinner("Extraindo embeddings de 512 dimensões…"):
        table = rank(png_bytes(query), tuple(png_bytes(x) for x in gallery), tuple(names), device)
        table["relation"] = [relations[int(i)] for i in table["index"]]
        st.session_state.person_reid_result = {"signature": signature, "table": table}

saved = st.session_state.person_reid_result
if saved and saved["signature"] == signature:
    table = saved["table"]
    st.subheader("Ranking calculado")
    chart = alt.Chart(table).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("cosine_similarity:Q", title="Similaridade do cosseno", scale=alt.Scale(domain=[-1, 1])),
        y=alt.Y("name:N", title=None, sort="-x"),
        color=alt.Color("relation:N", title="Origem"),
        tooltip=["name", "cosine_similarity", "relation"],
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")
    shown = table.rename(columns={"name": "Candidato", "cosine_similarity": "Similaridade", "relation": "Origem"}).drop(columns="index")
    st.dataframe(shown, hide_index=True)
    download_csv(table, "ranking_reid_pessoas.csv", "person_reid_csv")

st.subheader("Tipos de Re-ID de pessoas")
st.table({
    "Baseado em imagem": "Uma consulta e uma galeria de recortes independentes.",
    "Baseado em vídeo": "Agrega vários quadros de uma tracklet para reduzir ruído.",
    "Entre câmeras": "Busca a mesma pessoa em câmeras com pontos de vista distintos.",
    "Visível e infravermelho": "Relaciona modalidades diferentes, comum em transições dia/noite.",
    "Longo prazo ou troca de roupa": "Reduz a dependência da vestimenta, com maior risco e exigência de governança.",
})
teaching_prompt("Quando duas pessoas usam roupas semelhantes, que evidências de tempo, câmera e trajetória podem reduzir uma associação errada?")
with st.expander("Limites e governança"):
    st.markdown("Este baseline usa ResNet18 treinada para classificação ImageNet, não uma rede especializada em Re-ID de pessoas. A similaridade não identifica uma pessoa, não usa rosto e não produz identidade civil. Sistemas reais exigem conjunto de teste por câmera, métricas de recuperação, avaliação de vieses, finalidade definida, retenção limitada e revisão humana.")
