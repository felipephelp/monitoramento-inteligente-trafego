import streamlit as st

from ui import (
    header,
    multimodal_report,
    scenario_vlm_report,
    source_image,
    teaching_prompt,
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

    prompt_templates = {
        "Cena de tráfego": (
            "Use somente [EVIDÊNCIA VISUAL DO BLIP]. Descreva os atores explicitamente mencionados e informe "
            "que movimento, velocidade, direção do fluxo e interações não podem ser determinados por uma imagem "
            "estática quando não estiverem presentes na evidência. Não trate veículos estacionados como congestionamento."
        ),
        "Condição da via": (
            "Use somente [EVIDÊNCIA VISUAL DO BLIP]. Mencione superfície, visibilidade, ocupação, congestionamento, "
            "obstruções ou riscos apenas quando estiverem explícitos. Caso contrário, declare que não é possível "
            "determinar a condição da via. Não trate veículos estacionados como congestionamento."
        ),
    }

    with st.form("vlm_caption_controls", border=True):
        scenario = st.segmented_control(
            "Cenário analisado",
            ["Cena de tráfego", "Condição da via"],
            default="Cena de tráfego",
            key="vlm_prompt_mode",
        )
        st.subheader("Prompt do cenário", icon=":material/terminal:")
        st.code(prompt_templates[scenario], language=None)
        st.caption(
            "Na execução, o trecho entre colchetes é substituído pela descrição visual produzida pelo BLIP."
        )
        submitted = st.form_submit_button(
            "Analisar cenário com BLIP + Qwen",
            icon=":material/play_arrow:",
            type="primary",
        )

    if submitted:
        result_slot = st.container()
        try:
            with result_slot.skeleton(height=170):
                report = scenario_vlm_report(image_bytes, scenario)
            st.session_state["vlm_result"] = report
        except Exception as exc:
            st.error(
                "O modelo não pôde ser executado. Na primeira utilização, confirme o acesso à internet "
                f"para baixar os pesos. Detalhe: {exc}"
            )

    result = st.session_state.get("vlm_result")
    if result and "prompt_pt" in result and "generated_pt" in result:
        st.header("Prompt e saída do modelo", icon=":material/compare_arrows:")
        prompt_col, output_col = st.columns(2, vertical_alignment="top")
        with prompt_col.container(border=True, height="stretch"):
            st.subheader("1. Prompt enviado", icon=":material/input:")
            st.code(result["prompt_pt"], language=None)
            st.caption(f"Cenário selecionado: {result['scenario']}.")
        with output_col.container(border=True, height="stretch"):
            st.subheader("2. Saída final em português", icon=":material/subtitles:")
            st.markdown(f"## {result['generated_pt']}")

        with st.container(border=True):
            st.subheader("Evidência intermediária", icon=":material/data_object:")
            raw_col, translated_col = st.columns(2, vertical_alignment="top")
            with raw_col:
                st.markdown("**BLIP — descrição original em inglês**")
                st.code(result["visual_evidence_en"], language=None)
            with translated_col:
                st.markdown("**Qwen — tradução para português**")
                st.code(result["visual_evidence_pt"], language=None)
            st.caption(
                "O Qwen aplica o prompt específico do cenário apenas depois que o BLIP descreve a imagem. "
                "A resposta não substitui detecção, medição ou validação operacional."
            )

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

    prompt_preview = (
        "Use exatamente os quatro dados em uma única frase: "
        "evidência visual = [descrição que será produzida pelo BLIP]; "
        f"radar = {speed} km/h; semáforo = {signal.lower()}; ambiente = {weather.lower()}. "
        "Formato: A evidência visual mostra ...; o radar informa ...; "
        "o semáforo está ...; o ambiente está ...."
    )
    with st.container(border=True):
        st.subheader("Prompt preparado para a fusão", icon=":material/terminal:")
        st.code(prompt_preview, language=None)
        st.caption(
            "O campo entre colchetes será substituído pela descrição real produzida pelo BLIP antes de o prompt "
            "ser enviado ao Qwen."
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
        st.header("Prompt e saída do modelo", icon=":material/compare_arrows:")
        prompt_col, output_col = st.columns(2, vertical_alignment="top")
        with prompt_col.container(border=True, height="stretch"):
            st.subheader("1. Prompt enviado ao Qwen", icon=":material/input:")
            st.code(fusion_result["prompt_pt"], language=None)
        with output_col.container(border=True, height="stretch"):
            st.subheader("2. Saída gerada pelo Qwen", icon=":material/assignment:")
            st.markdown(f"## {fusion_result['generated_pt']}")

        with st.container(border=True):
            st.subheader("Saídas intermediárias do pipeline", icon=":material/account_tree:")
            raw_col, translated_col = st.columns(2, vertical_alignment="top")
            with raw_col:
                st.markdown("**BLIP — descrição original em inglês**")
                st.code(fusion_result["visual_evidence_en"], language=None)
            with translated_col:
                st.markdown("**Qwen — descrição traduzida para português**")
                st.code(fusion_result["visual_evidence_pt"], language=None)
            st.markdown(
                f"**Contexto utilizado:** {fusion_result['speed']} km/h · semáforo "
                f"{fusion_result['signal'].lower()} · {fusion_result['weather'].lower()}."
            )
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
