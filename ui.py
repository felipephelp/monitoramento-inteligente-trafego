"""Recursos, inferências em cache e componentes nativos compartilhados."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import os
import threading
import uuid
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
EXAMPLES = ASSETS / "examples"
EXPORTS = ROOT / "exports"
REFERENCES = ASSETS / "referencias"
os.environ.setdefault("HF_HOME", str(ROOT / ".model-cache"))

def png_bytes(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_resource(max_entries=1)
def detector():
    from vision import load_detector
    return load_detector(), threading.RLock()

@st.cache_resource(max_entries=2)
def embedder(device="cpu"):
    from vision import load_embedder
    return load_embedder(device=device), threading.RLock()

@st.cache_resource(max_entries=1)
def vlm_captioner():
    """Carrega BLIP uma vez; o primeiro uso baixa os pesos do modelo."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model.eval()
    return processor, model, threading.RLock()

@st.cache_resource(max_entries=1)
def fusion_text_generator():
    """Carrega o gerador textual que sintetiza evidência visual e sensores."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    model.eval()
    return tokenizer, model, threading.RLock()

@st.cache_data(max_entries=30, show_spinner=False)
def infer(data, conf, imgsz, device="cpu"):
    from vision import detect_image
    model, lock = detector()
    with lock:
        return detect_image(model, Image.open(BytesIO(data)).convert("RGB"), conf=conf, imgsz=imgsz, device=device)

@st.cache_data(max_entries=20, show_spinner=False)
def rank(data, gallery_bytes, names, device="cpu"):
    from vision import rank_gallery
    model, lock = embedder(device)
    with lock:
        return rank_gallery(model, Image.open(BytesIO(data)).convert("RGB"), [Image.open(BytesIO(b)).convert("RGB") for b in gallery_bytes], names=names)

@st.cache_data(max_entries=20, show_spinner=False)
def caption_image(data, prompt=""):
    """Gera uma legenda real com BLIP; retorna o texto bruto produzido pelo modelo."""
    import torch

    processor, model, lock = vlm_captioner()
    image = Image.open(BytesIO(data)).convert("RGB")
    inputs = processor(images=image, text=prompt or None, return_tensors="pt")
    with lock, torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=45, num_beams=3)
    return processor.decode(output[0], skip_special_tokens=True).strip()

@st.cache_data(max_entries=30, show_spinner=False)
def translate_to_portuguese(text):
    """Traduz a resposta do BLIP para português brasileiro com o Qwen local."""
    import torch

    tokenizer, model, lock = fusion_text_generator()
    messages = [
        {
            "role": "system",
            "content": "Traduza para português brasileiro. Responda somente com a tradução, sem explicação.",
        },
        {"role": "user", "content": text},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt")
    with lock, torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=70, do_sample=False, repetition_penalty=1.1)
    generated_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

@st.cache_data(max_entries=30, show_spinner=False)
def scenario_vlm_report(data, scenario):
    """Gera uma resposta em português orientada ao cenário escolhido."""
    import torch

    visual_evidence_en = caption_image(data)
    visual_evidence_pt = translate_to_portuguese(visual_evidence_en)
    prompts = {
        "Cena de tráfego": (
            "Produza exatamente duas frases, sem título e sem lista. Use este formato obrigatório: "
            f"\"A evidência visual mostra {visual_evidence_pt}. Por ser uma imagem estática, não é possível "
            "determinar movimento, velocidade, direção do fluxo ou interações que não estejam explicitamente "
            "visíveis.\" Não acrescente identidade, intenção, infração, congestionamento ou qualquer outro fato."
        ),
        "Condição da via": (
            "Produza exatamente duas frases, sem título e sem lista. Use este formato obrigatório: "
            f"\"A evidência visual mostra {visual_evidence_pt}. A descrição disponível não permite determinar "
            "superfície, visibilidade, congestionamento, obstruções ou riscos da via.\" Não transforme veículos "
            "estacionados em congestionamento e não acrescente clima, velocidade, dano, acidente ou pavimento."
        ),
    }
    if scenario not in prompts:
        raise ValueError(f"Cenário VLM não reconhecido: {scenario}")
    user_prompt = prompts[scenario]
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente de monitoramento urbano. Use exclusivamente a evidência fornecida, "
                "diferencie observação de inferência e responda sem títulos ou listas."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    tokenizer, model, lock = fusion_text_generator()
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt")
    with lock, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=110,
            do_sample=False,
            repetition_penalty=1.1,
        )
    generated_tokens = output[0][inputs.input_ids.shape[1]:]
    generated_pt = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return {
        "scenario": scenario,
        "visual_evidence_en": visual_evidence_en,
        "visual_evidence_pt": visual_evidence_pt,
        "prompt_pt": user_prompt,
        "generated_pt": generated_pt,
    }

@st.cache_data(max_entries=30, show_spinner=False)
def multimodal_report(data, speed, signal, weather):
    """Extrai evidência visual com BLIP e gera a síntese multimodal com Qwen."""
    import torch

    visual_evidence = caption_image(data)
    visual_evidence_pt = translate_to_portuguese(visual_evidence)
    user_prompt = (
        "Use exatamente os quatro dados em uma única frase: "
        f"evidência visual = {visual_evidence_pt}; "
        f"radar = {int(speed)} km/h; "
        f"semáforo = {signal.lower()}; "
        f"ambiente = {weather.lower()}. "
        "Formato: A evidência visual mostra ...; o radar informa ...; "
        "o semáforo está ...; o ambiente está ...."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Você redige uma frase factual de monitoramento de tráfego em português brasileiro. "
                "Copie todos os valores exatamente, não interprete, não use conhecimento externo e não "
                "declare limites, causalidade, identidade ou infrações."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    tokenizer, model, lock = fusion_text_generator()
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt")
    with lock, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=70,
            do_sample=False,
            repetition_penalty=1.1,
        )
    generated_tokens = output[0][inputs.input_ids.shape[1]:]
    generated_pt = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return {
        "visual_evidence_en": visual_evidence,
        "visual_evidence_pt": visual_evidence_pt,
        "prompt_pt": user_prompt,
        "generated_pt": generated_pt,
    }

@st.cache_data(max_entries=15, show_spinner=False)
def occlude(data, box, conf, device="cpu"):
    from vision import occlusion_experiment
    model, lock = detector()
    with lock:
        return occlusion_experiment(model, Image.open(BytesIO(data)).convert("RGB"), box, conf=conf, device=device)

def source_image(key, default="Tráfego urbano · câmera fixa"):
    choices = ["Tráfego urbano · câmera fixa", "Vista aérea · objetos pequenos", "Cena urbana · ônibus e pedestres", "Enviar minha imagem"]
    choice = st.selectbox("Imagem de entrada", choices, index=2 if default.startswith("Cena") else 0, key=key+"_source")
    uploaded = None
    if choice == "Enviar minha imagem":
        uploaded = st.file_uploader("Imagem JPG, PNG ou WebP", type=["jpg", "jpeg", "png", "webp"], key=key+"_upload")
        if uploaded is None:
            st.info("Envie uma imagem ou selecione um dos exemplos já disponíveis.")
            return None, None
        data = uploaded.getvalue()
    else:
        file = "urban_traffic_frame.jpg" if choice.startswith("Tráfego") else "vehicles.jpg" if choice.startswith("Vista") else "bus.jpg"
        data = (EXAMPLES / file).read_bytes()
    try:
        img = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
        img.thumbnail((2400, 2400))
        return img, png_bytes(img)
    except Exception as exc:
        st.error(f"Não foi possível ler esta imagem: {exc}")
        return None, None

def download_csv(df, name, key):
    st.download_button("Baixar tabela CSV", df.to_csv(index=False).encode("utf-8-sig"), name, "text/csv", icon=":material/download:", key=key)

def save_upload(upload):
    suffix = Path(upload.name).suffix.lower()
    out = ROOT / "uploads" / (hashlib.sha256(upload.getvalue()).hexdigest()[:20] + suffix)
    out.parent.mkdir(exist_ok=True)
    if not out.exists():
        out.write_bytes(upload.getvalue())
    return out

def new_result_dir():
    path = ROOT / "results" / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    return path

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def header(number, title, subtitle):
    st.caption(f"LABORATÓRIO / {number}")
    st.title(title)
    st.markdown(subtitle)

def teaching_prompt(text):
    with st.container(border=True):
        st.markdown(f":material/question_answer: **Conversa com a audiência**\n\n{text}")

def figure(name, caption):
    path = REFERENCES / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")

def monitoring_scenario(kind, task):
    """Mostra a geometria e os desafios do cenário antes do processamento."""
    scenarios = {
        "fixed": {
            "title": "Câmera fixa · corredor viário",
            "image": EXAMPLES / "urban_traffic_frame.jpg",
            "caption": "Ponto de vista oblíquo e estável · exemplo público Ultralytics",
            "badge": "Geometria estável",
            "text": (
                "A câmera não se move, então a linha virtual permanece no mesmo lugar. "
                "Perspectiva, veículos sobrepostos e congestionamento ainda alteram a escala aparente."
            ),
            "focus": "Calibrar uma vez, validar a linha por faixa e observar oclusões próximas à câmera.",
        },
        "drone": {
            "title": "Drone · vista aérea",
            "image": EXAMPLES / "vehicles.jpg",
            "caption": "Vista quase nadir com objetos pequenos · exemplo público Ultralytics",
            "badge": "Campo amplo",
            "text": (
                "A visão aérea cobre várias vias, mas os veículos ocupam poucos pixels. "
                "Movimento, vibração e mudança de altitude deslocam linhas e trajetórias no quadro."
            ),
            "focus": "Estabilizar o vídeo, usar resolução adequada e compensar o movimento da câmera.",
        },
    }
    item = scenarios[kind]
    with st.container(border=True):
        image_col, text_col = st.columns([1.35, 1], vertical_alignment="center")
        with image_col:
            st.image(str(item["image"]), caption=item["caption"], width="stretch")
        with text_col:
            st.subheader(item["title"], icon=":material/videocam:")
            st.badge(item["badge"], color="blue")
            st.markdown(item["text"])
            st.markdown(f"**No {task}:** {item['focus']}")

def visual_ranking(table, gallery, title="Ranking visual"):
    """Renderiza o ranking de Re-ID com imagens e resultados legíveis."""
    st.subheader(title, icon=":material/photo_library:")
    st.markdown(
        "Os candidatos abaixo já estão **reordenados do mais semelhante para o menos semelhante**. "
        "O cosseno mede proximidade visual; não é probabilidade de identidade."
    )
    rows = list(table.reset_index(drop=True).iterrows())
    for start in range(0, len(rows), 2):
        columns = st.columns(2)
        for column, (position, row) in zip(columns, rows[start:start + 2]):
            original_index = int(row["index"])
            score = float(row["cosine_similarity"])
            with column:
                with st.container(border=True):
                    st.subheader(f"{position + 1}º · {row['name']}")
                    st.image(gallery[original_index], width="stretch")
                    st.metric("Similaridade do cosseno", f"{score:.3f}")
                    relation = row.get("relation", "")
                    if relation:
                        st.markdown(f"**Relação no experimento:** {relation}")

def runtime_choice(key):
    import torch
    options = ["CPU"] + (["GPU CUDA"] if torch.cuda.is_available() else [])
    value = st.selectbox("Processamento", options, key=key)
    return "0" if value.startswith("GPU") else "cpu"
