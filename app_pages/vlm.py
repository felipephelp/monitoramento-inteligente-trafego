import streamlit as st

from ui import (
    caption_image,
    header,
    multimodal_report,
    source_image,
    teaching_prompt,
    translate_to_portuguese,
)


header(
    "07",
    "VLM e redes multimodais",
    "Um VLM relaciona evidência visual e linguagem. A fusão multimodal acrescenta contexto de sensores e da operação.",
)

view = st.segmented_control(
    "Experimento",
    ["BLIP: imagem e linguagem", "Fusão: câmera e contexto"],
    default="BLIP: imagem e linguagem",
    key="vlm_view",
)

if view == "BLIP: imagem e linguagem":
    image, image_bytes = source_image("vlm", default="Tráfego urbano · câmera fixa")
    if image is None:
        st.stop()

    image_col, model_col = st.columns([1.25, 1], vertical_alignment="top")
    with image_col:
        st.image(image, caption="Entrada visual escolhida para a demonstração", width="stretch")
    with model_col:
        st.subheader("Modelo executado", icon=":material/model_training:")
        st.markdown(
            "**BLIP image captioning base** combina um codificador visual com um decodificador de linguagem. "
            "Ele gera uma legenda condicionada pela imagem e, opcionalmente, por um início de frase."
        )
        st.table(
            {
                "Entrada visual": "Pixels transformados em características e tokens visuais.",
                "Entrada textual": "Início de frase opcional que orienta a geração.",
                "Saída": "Sequência de palavras produzida pelo decodificador.",
            },
            border="horizontal",
        )

    with st.form("vlm_caption_controls", border=True):
        prompt_mode = st.segmented_control(
            "Orientação textual",
            ["Sem orientação", "Cena de tráfego", "Condição da via"],
            default="Sem orientação",
            key="vlm_prompt_mode",
        )
        submitted = st.form_submit_button(
            "Gerar descrição com BLIP",
            icon=":material/play_arrow:",
            type="primary",
        )

    prompts = {
        "Sem orientação": "",
        "Cena de tráfego": "a traffic monitoring image showing",
        "Condição da via": "the traffic condition is",
    }
    if submitted:
        result_slot = st.container()
        try:
            with result_slot.skeleton(height=170):
                text_en = caption_image(image_bytes, prompts[prompt_mode])
                text_pt = translate_to_portuguese(text_en)
            st.session_state["vlm_result"] = {
                "caption_en": text_en,
                "caption_pt": text_pt,
                "prompt_mode": prompt_mode,
            }
        except Exception as exc:
            st.error(
                "O modelo não pôde ser executado. Na primeira utilização, confirme o acesso à internet "
                f"para baixar os pesos. Detalhe: {exc}"
            )

    result = st.session_state.get("vlm_result")
    if result and "caption_pt" in result:
        with st.container(border=True):
            st.subheader("Resposta em português", icon=":material/subtitles:")
            st.markdown(f"### {result['caption_pt']}")
            st.caption(
                f"Orientação utilizada: {result['prompt_mode']}. A imagem é descrita pelo BLIP e a saída é "
                "traduzida para português brasileiro pelo Qwen2.5-1.5B-Instruct."
            )
            with st.expander("Ver saída original do BLIP"):
                st.code(result["caption_en"], language=None)

    st.mermaid_chart(
        """flowchart LR
            A[Imagem] --> B[Codificador visual]
            P[Início de frase] --> C[Fusão visão e linguagem]
            B --> C
            C --> D[Decodificador textual]
            D --> E[Legenda]
        """
    )
    teaching_prompt(
        "A legenda descreve todos os veículos e conflitos da cena? Que informação ainda exige detecção, tracking ou calibração?"
    )

else:
    st.subheader("Mesmo vídeo, interpretações diferentes", icon=":material/join_inner:")
    st.markdown(
        "Este exemplo executa uma **geração multimodal rastreável**. O BLIP extrai a evidência da imagem e o Qwen "
        "traduz e combina essa descrição com radar, controlador semafórico e condição ambiental."
    )

    image, image_bytes = source_image("fusion", default="Tráfego urbano · câmera fixa")
    if image is None:
        st.stop()

    image_col, control_col = st.columns([1.25, 1], vertical_alignment="top")
    with image_col:
        st.image(image, caption="Modalidade visual: câmera RGB", width="stretch")
    with control_col:
        speed = st.slider("Velocidade medida pelo radar", 0, 100, 42, 1, key="fusion_speed")
        signal = st.segmented_control(
            "Estado do semáforo",
            ["Verde", "Amarelo", "Vermelho"],
            default="Vermelho",
            key="fusion_signal",
        )
        weather = st.segmented_control(
            "Condição ambiental",
            ["Seco", "Chuva", "Baixa visibilidade"],
            default="Seco",
            key="fusion_weather",
        )

    with st.container(border=True):
        st.subheader("Modelo gerador", icon=":material/model_training:")
        st.markdown(
            "**BLIP image captioning base + Qwen2.5-1.5B-Instruct**. O BLIP converte os pixels em evidência "
            "textual; o Qwen traduz para português brasileiro e gera a síntese com os dados dos sensores."
        )
        st.table(
            {
                "Imagem": "Pixels da câmera transformados em tokens visuais.",
                "Contexto": f"{speed} km/h · semáforo {signal.lower()} · {weather.lower()}.",
                "Geração": "Texto condicionado pela imagem e pelo contexto informado.",
            },
            border="horizontal",
        )

    if st.button(
        "Gerar relatório com o VLM",
        icon=":material/play_arrow:",
        type="primary",
        key="fusion_generate",
    ):
        result_slot = st.container()
        try:
            with result_slot.skeleton(height=190):
                report = multimodal_report(image_bytes, speed, signal, weather)
            st.session_state["fusion_vlm_result"] = {
                **report,
                "speed": speed,
                "signal": signal,
                "weather": weather,
            }
        except Exception as exc:
            st.error(
                "O pipeline multimodal não pôde ser executado. Na primeira utilização, confirme o acesso à internet "
                f"para baixar os pesos. Detalhe: {exc}"
            )

    fusion_result = st.session_state.get("fusion_vlm_result")
    if fusion_result and "visual_evidence_en" in fusion_result:
        with st.container(border=True):
            st.subheader("Relatório gerado em português", icon=":material/assignment:")
            st.markdown(f"### {fusion_result['generated_pt']}")
            st.markdown(
                f"**Contexto usado na geração:** {fusion_result['speed']} km/h · "
                f"semáforo {fusion_result['signal'].lower()} · {fusion_result['weather'].lower()}."
            )
            with st.expander("Auditar entradas e saída original"):
                st.markdown("**Evidência visual original produzida pelo BLIP**")
                st.code(fusion_result["visual_evidence_en"], language=None)
                st.markdown("**Evidência visual traduzida para português**")
                st.code(fusion_result["visual_evidence_pt"], language=None)
                st.markdown("**Prompt multimodal enviado ao Qwen**")
                st.code(fusion_result["prompt_pt"], language=None)
            st.caption(
                "Esta é uma demonstração de geração, não uma comprovação automática de infração. Em produção, "
                "sincronização, calibração e evidência verificável continuam obrigatórias."
            )

    st.mermaid_chart(
        """flowchart LR
            A[Câmera RGB] --> V[BLIP: evidência visual]
            B[Radar] --> F
            C[Fase semafórica] --> F
            D[Clima] --> F
            V --> T[Qwen2.5: tradução]
            T --> F
            F --> G[Qwen2.5: síntese gerativa]
            G --> H[Relatório em português]
        """
    )
    teaching_prompt(
        "A imagem sozinha comprova avanço de sinal? Quais relógios, linhas de retenção e regras precisam estar calibrados?"
    )

st.subheader("Onde aplicar", icon=":material/location_city:")
applications = st.columns(2)
with applications[0].container(border=True, height="stretch"):
    st.markdown("**TrafficVLM e modelos semelhantes**")
    st.markdown(
        "Descrição temporal de veículos e pedestres, geração de legendas para incidentes e criação de dados pesquisáveis "
        "em câmeras embarcadas ou superiores."
    )
    st.link_button("Artigo TrafficVLM", "https://arxiv.org/abs/2404.09275")
with applications[1].container(border=True, height="stretch"):
    st.markdown("**NVIDIA Metropolis VSS**")
    st.markdown(
        "Busca semântica em vídeo, revisão de alertas, perguntas sobre gravações e resumos de fluxos ao vivo ou arquivados."
    )
    st.link_button("Documentação NVIDIA VSS", "https://developer.nvidia.com/metropolis")

with st.expander("Artigos e limites do experimento", icon=":material/article:"):
    st.markdown(
        "- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)\n"
        "- [TrafficVLM: traffic video captioning](https://arxiv.org/abs/2404.09275)\n"
        "- [NVIDIA VSS: vídeo, VLM, LLM e recuperação](https://developer.nvidia.com/blog/advance-video-analytics-ai-agents-using-the-nvidia-ai-blueprint-for-video-search-and-summarization/)\n\n"
        "O BLIP desta página demonstra captioning de uma imagem. Ele não foi ajustado para tráfego, não processa a "
        "sequência temporal e pode omitir ou inventar detalhes. Decisões operacionais devem usar evidências verificáveis."
    )
