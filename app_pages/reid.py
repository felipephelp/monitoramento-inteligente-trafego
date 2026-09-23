from io import BytesIO
import altair as alt
from PIL import Image, ImageEnhance
import streamlit as st
from ui import header, EXAMPLES, read_json, png_bytes, rank, download_csv, teaching_prompt, figure, visual_ranking

header("04", "Aparência gera candidatos, não identidade", "**ResNet18 · ImageNet · embedding de 512 dimensões.** Um baseline neural real para ensinar similaridade; não é um modelo especializado em Re-ID de veículos.")
source = st.segmented_control("Conjunto", ["Exemplo controlado", "Minhas imagens"], default="Exemplo controlado", key="reid_source")
gallery=[]
names=[]
relations=[]
if source == "Minhas imagens":
    a,b=st.columns(2)
    query_file=a.file_uploader("Consulta: um recorte de veículo",type=["jpg","jpeg","png"],key="reid_query_upload")
    files=b.file_uploader("Galeria: 2 ou mais recortes",type=["jpg","jpeg","png"],accept_multiple_files=True,key="reid_gallery_upload")
    if not query_file or len(files)<2:
        st.info("Envie a consulta e ao menos dois candidatos. Prefira recortes centrados no objeto.")
        st.stop()
    query=Image.open(BytesIO(query_file.getvalue())).convert("RGB")
    gallery=[Image.open(BytesIO(f.getvalue())).convert("RGB") for f in files[:24]]
    names=[f.name for f in files[:24]]
    relations=["Identidade não verificada"]*len(names)
else:
    manifest=EXAMPLES/"reid_manifest.json"
    if not manifest.exists():
        st.info("Exemplos de Re-ID ainda não preparados. Execute prepare_examples.py ou envie suas imagens.")
        st.stop()
    data=read_json(manifest)
    query=Image.open(EXAMPLES/data["query"]).convert("RGB")
    for entry in data["gallery"]:
        gallery.append(Image.open(EXAMPLES/entry["path"]).convert("RGB"))
        names.append(entry["name"])
        relations.append(entry.get("relation",""))
    st.caption("Experimento controlado: consulta e variantes derivadas de um mesmo recorte, junto de outros objetos. Não é benchmark de câmeras diferentes.")
left,right=st.columns([1,2.5])
with left:
    brightness=st.slider("Iluminação da consulta",.3,1.7,1.,.1,key="reid_brightness")
    modified=ImageEnhance.Brightness(query).enhance(brightness)
    st.image(modified,caption="Consulta",width="stretch")
    execute=st.button("Calcular ranking",type="primary",icon=":material/compare:",key="reid_execute")
with right:
    st.subheader("Galeria de candidatos")
    cols=st.columns(min(4,len(gallery)))
    for i,(img,name) in enumerate(zip(gallery,names)):
        with cols[i%len(cols)]:
            st.image(img,caption=name,width="stretch")
result_slot=st.container()
if execute:
    with st.spinner("Extraindo embeddings e calculando similaridade do cosseno…"):
        df=rank(png_bytes(modified),tuple(png_bytes(x) for x in gallery),tuple(names))
        df["relation"]=[relations[int(i)] for i in df["index"]]
        st.session_state.reid_result={"df":df,"query":png_bytes(modified),"names":names}
saved=st.session_state.reid_result
with result_slot:
    if saved and saved["query"]==png_bytes(modified) and saved["names"]==names:
        df=saved["df"]
        st.subheader("Ranking calculado pelo modelo")
        visual_ranking(df, gallery, "Ranking visual dos veículos")
        st.altair_chart(alt.Chart(df).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("cosine_similarity:Q",title="Similaridade do cosseno",scale=alt.Scale(domain=[-1,1])),
            y=alt.Y("name:N",title=None,sort="-x"),color=alt.Color("relation:N",title="Origem"),
            tooltip=["name","cosine_similarity","relation"]).properties(height=240),width="stretch")
        st.dataframe(df.rename(columns={"name":"Candidato","cosine_similarity":"Similaridade","relation":"Origem"}).drop(columns="index"),hide_index=True)
        download_csv(df,"ranking_reid.csv","reid_csv")
st.mermaid_chart("""flowchart LR
    A[Consulta] --> B[ResNet18<br/>sem classificador]
    C[Galeria] --> B
    B --> D[Vetores 512D<br/>normalização L2]
    D --> E[Similaridade cosseno]
    E --> F[Ranking]
    F --> G[Contexto temporal<br/>rota + validação]
""")
teaching_prompt("Se dois carros brancos têm embeddings próximos, quais restrições de tempo e rota ajudam a descartar uma associação impossível?")
with st.expander("Conexão com sua pesquisa"):
    figure("tese_fluxo_reid.png","Fluxo original de Re-ID · defesa de doutorado de Felipe Euphrasio, slide 45.")
    st.markdown("O baseline desta demonstração não reproduz o treinamento nem os resultados da tese. Similaridade não é probabilidade de mesma identidade; avalie modelos especializados em conjuntos independentes e cenários multicâmera.")
