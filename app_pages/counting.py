from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ui import EXAMPLES, EXPORTS, download_csv, header, monitoring_scenario, new_result_dir, runtime_choice, save_upload, teaching_prompt
from vision import track_video, video_info


header("04", "Contagem de veículos por linha virtual", "O detector localiza, o tracker mantém o ID e a regra registra um evento quando a trajetória cruza a linha.")

source = st.segmented_control(
    "Cenário e fonte",
    ["Exemplo · câmera fixa", "Meu vídeo · câmera fixa", "Meu vídeo · drone"],
    default="Exemplo · câmera fixa",
    key="count_source",
)
scenario = "drone" if source.endswith("drone") else "fixed"
monitoring_scenario(scenario, "contagem")
path = EXAMPLES / "traffic.mp4"
if source != "Exemplo · câmera fixa":
    label = "Vídeo de drone" if scenario == "drone" else "Vídeo de câmera fixa"
    uploaded = st.file_uploader(f"{label} · MP4, AVI ou MOV · até 200 MB", type=["mp4", "avi", "mov"], key="count_upload")
    if uploaded is None:
        st.info("A imagem acima ilustra o cenário. Envie o vídeo correspondente para executar a contagem.")
        st.stop()
    path = save_upload(uploaded)

info = video_info(path)
if scenario == "drone":
    st.warning("Uma linha fixa no quadro só é válida se o vídeo aéreo estiver estabilizado. Caso contrário, a linha se desloca sobre a via e produz cruzamentos falsos.", icon=":material/warning:")
with st.form("counting_controls"):
    a, b, c = st.columns(3)
    line = a.slider("Altura da linha", .15, .85, .55, .05, key="count_line")
    conf = b.slider("Confiança de entrada", .05, .70, .15, .05, key="count_conf")
    frames = c.slider("Quadros", 30, 600, 180, 30, key="count_frames")
    a, b = st.columns(2)
    tracker = a.selectbox("Associação", ["ByteTrack", "BoT-SORT"], key="count_tracker")
    with b:
        device = runtime_choice("count_device")
    execute = st.form_submit_button("Executar contagem", type="primary", icon=":material/calculate:")

if execute:
    progress = st.progress(0, text="Detectando, associando e avaliando cruzamentos…")
    try:
        result = track_video(
            "yolo11n.pt", path, new_result_dir(), conf=conf,
            tracker="bytetrack.yaml" if tracker == "ByteTrack" else "botsort.yaml",
            max_frames=frames, device=device, line_y=line, classes=[2, 3, 5, 7],
            progress=lambda n, total: progress.progress(min(1., n/total), text=f"Quadro {n}/{total}"),
        )
        st.session_state.counting_result = {"result": result, "source": str(path)}
    except Exception as exc:
        st.error(f"Não foi possível executar a contagem: {exc}")
    finally:
        progress.empty()

saved = st.session_state.counting_result
if saved and saved["source"] == str(path):
    result = saved["result"]
    events = result["crossings"]
    stats = result["stats"]
    video_path = Path(result["video_path"])
    caption = f"Execução atual · linha={stats['line_y']:.2f} · {stats['tracker']} · {stats['frames']} quadros"
elif source == "Exemplo · câmera fixa":
    events_path = EXPORTS / "tracking_crossings.csv"
    tracks_path = EXPORTS / "tracking_tracks.csv"
    video_path = EXPORTS / "tracking" / "tracking.mp4"
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(columns=["frame", "time_s", "track_id", "class_name", "direction"])
    tracks = pd.read_csv(tracks_path) if tracks_path.exists() else pd.DataFrame()
    stats = {"crossings": len(events), "unique_ids": int(tracks.track_id.nunique()) if len(tracks) else 0, "frames": int(tracks.frame.max()+1) if len(tracks) else 0}
    caption = "Resultado pré-calculado com YOLO11n + ByteTrack · linha em 55% da altura"
else:
    video_path = path
    events = pd.DataFrame(columns=["frame", "time_s", "track_id", "class_name", "direction"])
    stats = {"crossings": 0, "unique_ids": 0, "frames": 0}
    caption = "Vídeo enviado · execute a contagem para gerar os eventos deste cenário"

left, right = st.columns([1.45, 1], vertical_alignment="top")
with left:
    st.subheader("Evidência em vídeo")
    st.video(str(video_path if video_path.exists() else path))
    st.caption(caption)
with right:
    st.subheader("Eventos registrados")
    up = int((events.direction == "subindo").sum()) if len(events) else 0
    down = int((events.direction == "descendo").sum()) if len(events) else 0
    with st.container(horizontal=True):
        st.metric("Total", len(events), border=True)
        st.metric("Subindo", up, border=True)
        st.metric("Descendo", down, border=True)
    st.metric("IDs observados", stats.get("unique_ids", 0), border=True)

if len(events):
    summary = events.groupby(["class_name", "direction"], as_index=False).size().rename(columns={"size": "veículos"})
    st.subheader("Contagem por classe e direção")
    chart = alt.Chart(summary).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("veículos:Q", title="Eventos de cruzamento"),
        y=alt.Y("class_name:N", title="Classe"),
        color=alt.Color("direction:N", title="Direção"),
        tooltip=["class_name", "direction", "veículos"],
    ).properties(height=220)
    st.altair_chart(chart, width="stretch")
    st.dataframe(events, hide_index=True)
    download_csv(events, "contagem_cruzamentos.csv", "count_csv")
else:
    st.info("Nenhuma trajetória cruzou a linha nesta configuração. Mude a posição da linha, processe mais quadros ou reduza o limiar.")

teaching_prompt("Se o tracker troca o ID de um veículo antes da linha, a contagem diminui, aumenta ou permanece correta?")
with st.expander("Como interpretar esta medição"):
    st.markdown("A página conta eventos, não veículos existentes na imagem. Cada ID pode ser contado uma vez por direção. O resultado depende da linha, do tempo observado, da detecção e da continuidade do tracking. Validação operacional exige contagem manual de referência por classe e direção.")
