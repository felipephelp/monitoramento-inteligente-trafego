"""Laboratório local de visão computacional para tráfego urbano."""
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Tráfego inteligente | Laboratório", page_icon=":material/traffic:", layout="wide")

for name in ("detection_result", "tracking_result", "counting_result", "reid_result", "person_reid_result", "occlusion_result", "vlm_result", "fusion_vlm_result"):
    st.session_state.setdefault(name, None)

pages = {
    "Laboratório": [
        st.Page("app_pages/home.py", title="Visão geral", icon=":material/traffic:", default=True),
        st.Page("app_pages/detection.py", title="Detecção", icon=":material/frame_inspect:", url_path="detection"),
        st.Page("app_pages/tracking.py", title="Tracking", icon=":material/route:", url_path="tracking"),
        st.Page("app_pages/counting.py", title="Contagem", icon=":material/calculate:", url_path="counting"),
        st.Page("app_pages/reid.py", title="Re-ID de veículos", icon=":material/compare:", url_path="reid"),
        st.Page("app_pages/person_reid.py", title="Re-ID de pessoas", icon=":material/group:", url_path="person-reid"),
        st.Page("app_pages/occlusion.py", title="Oclusão", icon=":material/visibility_off:", url_path="occlusion"),
    ],
    "Conteúdo da apresentação": [
        st.Page("app_pages/vlm.py", title="VLM & multimodal", icon=":material/neurology:", url_path="vlm"),
        st.Page("app_pages/applications.py", title="Aplicações", icon=":material/grid_view:", url_path="applications"),
        st.Page("app_pages/guide.py", title="Guia & fontes", icon=":material/menu_book:", url_path="guide"),
    ],
}
page = st.navigation(pages, position="sidebar")
with st.sidebar:
    st.markdown("### Tráfego inteligente")
    st.caption("Laboratório de visão computacional\n\nFelipe Euphrasio · 60 min")
    st.badge("Inferência local", icon=":material/memory:", color="blue")
    st.caption("YOLO11n · ByteTrack / BoT-SORT\n\nContagem por linha · Re-ID · VLM e fusão multimodal")
    st.caption("Altere parâmetros, execute e compare. Os resultados pré-carregados também foram calculados pelos modelos.")
page.run()
