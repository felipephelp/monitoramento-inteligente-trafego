from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st
from ui import header, EXAMPLES, EXPORTS, save_upload, new_result_dir, download_csv, teaching_prompt, runtime_choice, monitoring_scenario
from vision import track_video, video_info

header("03", "Trajetórias dão memória ao vídeo", "**YOLO11n + ByteTrack ou BoT-SORT.** Cada execução começa com um tracker novo; IDs são hipóteses locais da sequência.")
source = st.segmented_control(
    "Cenário e fonte",
    ["Exemplo · câmera fixa", "Meu vídeo · câmera fixa", "Meu vídeo · drone"],
    default="Exemplo · câmera fixa",
    key="track_source",
)
scenario = "drone" if source.endswith("drone") else "fixed"
monitoring_scenario(scenario, "tracking")
path = EXAMPLES / "traffic.mp4"
if source != "Exemplo · câmera fixa":
    label = "Vídeo de drone" if scenario == "drone" else "Vídeo de câmera fixa"
    uploaded = st.file_uploader(f"{label} · MP4, AVI ou MOV · até 200 MB", type=["mp4", "avi", "mov"], key="track_upload")
    if uploaded is None:
        st.info("A imagem acima ilustra o cenário. Envie o vídeo correspondente para calcular IDs e trajetórias.")
        st.stop()
    path = save_upload(uploaded)
try:
    info = video_info(path)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
st.caption(f"Fonte: {info['width']} × {info['height']} · {info['fps']:.1f} fps · {info['duration_s']:.1f} s")
if scenario == "drone":
    st.warning("Em vídeo de drone, movimento da plataforma pode parecer movimento dos veículos. Estabilize ou estime o movimento global antes de interpretar trajetórias.", icon=":material/warning:")
with st.form("tracking_controls"):
    a, b, c = st.columns(3)
    tracker = a.selectbox("Algoritmo", ["ByteTrack", "BoT-SORT"], key="track_algorithm")
    conf = b.slider("Confiança de entrada", .05, .75, .15, .05, key="track_conf")
    max_frames = c.slider("Limite de quadros", 30, 600, 180, 30, key="track_frames")
    a, b, c = st.columns(3)
    line = a.slider("Linha de contagem · altura relativa", .1, .9, .6, .05, key="track_line")
    mask = b.checkbox("Inserir obstrução central", key="track_mask", help="Oculta uma região fixa antes de passar cada quadro ao modelo.")
    with c:
        device = runtime_choice("track_device")
    execute = st.form_submit_button("Processar vídeo", type="primary", icon=":material/play_arrow:")
st.caption("BoT-SORT usa o arquivo padrão da versão instalada, com Re-ID desativado. A página Re-ID separada ensina descritores de aparência.")
if execute:
    progress = st.progress(0, text="Carregando modelo e associando quadros…")
    try:
        result = track_video("yolo11n.pt", path, new_result_dir(), conf=conf,
            tracker="bytetrack.yaml" if tracker == "ByteTrack" else "botsort.yaml",
            max_frames=max_frames, device=device, line_y=line,
            occlusion_rect=(.35,.32,.62,.66) if mask else None,
            progress=lambda n, total: progress.progress(min(1., n/total), text=f"Quadro {n}/{total} · {tracker}"))
        st.session_state.tracking_result = {"result":result,"source":str(path),"masked":mask}
        progress.empty()
    except Exception as exc:
        progress.empty()
        st.error(f"Não foi possível processar: {exc}")
saved = st.session_state.tracking_result
if saved and saved["source"] == str(path):
    result = saved["result"]
    stats = result["stats"]
    st.caption(f"Resultado: {stats['tracker']} · conf={stats['conf']:.2f} · obstrução {'ativada' if saved['masked'] else 'desativada'}. Parâmetros novos requerem outra execução.")
    a, b = st.columns(2)
    with a:
        st.subheader("Vídeo original")
        st.video(str(path))
    with b:
        st.subheader("IDs, trajetórias e cruzamentos")
        st.video(result["video_path"])
    with st.container(horizontal=True):
        st.metric("IDs estimados", stats["unique_ids"], border=True)
        st.metric("Cruzamentos de linha", stats["crossings"], border=True)
        st.metric("Quadros processados", stats["frames"], border=True)
        st.metric("Processamento médio", f"{stats['processing_fps']:.1f} fps", border=True)
    tracks = result["tracks"]
    if len(tracks):
        a, b = st.columns([1.3,1])
        with a:
            st.subheader("Mapa de trajetórias em pixels")
            st.altair_chart(alt.Chart(tracks).mark_line(opacity=.8).encode(
                x=alt.X("cx:Q",title="x (pixels)",scale=alt.Scale(domain=[0,info["width"]])),
                y=alt.Y("cy:Q",title="y (pixels)",scale=alt.Scale(domain=[info["height"],0])),
                color=alt.Color("track_id:N",title="ID",legend=None),order="frame:Q",
                tooltip=["track_id","class_name","frame","time_s"]).properties(height=320), width="stretch")
        with b:
            st.subheader("Objetos ativos por quadro")
            active=tracks.groupby("frame").track_id.nunique().reset_index(name="Objetos")
            st.line_chart(active,x="frame",y="Objetos",height=320)
    tabs = st.tabs(["Cruzamentos", "Trajetórias", "Parâmetros"])
    with tabs[0]:
        st.dataframe(result["crossings"],hide_index=True)
        download_csv(result["crossings"],"cruzamentos.csv","cross_csv")
    with tabs[1]:
        st.dataframe(tracks,hide_index=True,height=250)
        download_csv(tracks,"trajetorias.csv","tracks_csv")
    with tabs[2]:
        st.json(stats)
    st.download_button("Baixar vídeo anotado",Path(result["video_path"]).read_bytes(),"tracking.mp4","video/mp4",icon=":material/download:")
else:
    st.video(str(path))
    st.caption("O vídeo acima é a fonte original. Clique em Processar vídeo para calcular IDs e trajetórias.")
teaching_prompt("Ao inserir uma obstrução, o número de IDs aumenta? Isso prova que apareceram mais veículos ou pode indicar fragmentação?")
with st.expander("Como interpretar a contagem"):
    st.markdown("A linha conta cada ID no máximo uma vez por direção, com histerese de 1% da altura. Trocas de ID, detecções perdidas ou posição da linha podem alterar o resultado. O fluxo é estimado em pixels; velocidade em km/h exige calibração. HOTA e IDF1 exigem trajetórias de referência anotadas.")
