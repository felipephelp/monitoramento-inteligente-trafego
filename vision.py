"""Modelos reais e experimentos reproduzíveis do laboratório de tráfego.

Não depende do Streamlit. Imagens de entrada/saída são PIL RGB; tabelas são
pandas. Métricas de oclusão medem estabilidade contra uma detecção de referência,
nunca acurácia contra anotação humana. Re-ID usa um baseline genérico ImageNet.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import subprocess
import time
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "assets" / "models"
DETECTION_COLUMNS = ["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
TRACK_COLUMNS = ["frame", "time_s", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "cx", "cy"]
CROSSING_COLUMNS = ["frame", "time_s", "track_id", "class_name", "direction"]
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7]


def load_detector(weights: str = "yolo11n.pt"):
    """Carrega pesos locais; na primeira chamada baixa o modelo oficial."""
    settings = ROOT / "assets" / "settings"
    (settings / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
    from ultralytics import YOLO
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(weights)
    if not path.is_absolute() and path.parent == Path("."):
        path = MODEL_DIR / path
    return YOLO(str(path))


def _table(result) -> pd.DataFrame:
    rows = []
    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().tolist()
            cls = int(box.cls[0].item())
            rows.append([cls, result.names[cls], float(box.conf[0].item()), *xyxy])
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def detect_image(model, image: Image.Image, conf: float = .25, imgsz: int = 640,
                 device: str = "cpu", classes=None) -> dict:
    start = time.perf_counter()
    result = model.predict(source=image.convert("RGB"), conf=conf, imgsz=imgsz,
                           device=device, classes=classes, verbose=False)[0]
    elapsed = (time.perf_counter() - start) * 1000
    return {"image": Image.fromarray(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)),
            "detections": _table(result), "inference_ms": elapsed,
            "model_inference_ms": float(result.speed.get("inference", 0))}


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU axis-aligned, com caixas degeneradas definidas como zero."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    intersection = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
    union = max(0., ax2-ax1)*max(0., ay2-ay1) + max(0., bx2-bx1)*max(0., by2-by1) - intersection
    return intersection / union if union > 0 else 0.


def apply_occlusion(image: Image.Image, box: Sequence[float], fraction: float,
                    color=(50, 57, 65)) -> Image.Image:
    """Mascara fração horizontal da bbox; 0 não altera nenhum pixel."""
    if not 0 <= fraction <= 1:
        raise ValueError("A fração de oclusão deve estar entre 0 e 1.")
    output = image.convert("RGB").copy()
    if fraction == 0:
        return output
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Caixa de referência vazia.")
    arr = np.array(output)
    left, top = max(0, int(round(x1))), max(0, int(round(y1)))
    right = min(output.width, int(round(x1 + (x2-x1)*fraction)))
    bottom = min(output.height, int(round(y2)))
    arr[top:bottom, left:right] = color
    return Image.fromarray(arr)


def occlusion_experiment(model, image: Image.Image, box: Sequence[float],
                         levels=(0., .2, .4, .6, .8), conf=.25, imgsz=640,
                         device="cpu", match_iou=.3, target_class=None) -> dict:
    """Retenção da detecção, não mAP/recall: referência é a imagem sem máscara."""
    if target_class is None:
        baseline = detect_image(model, image, conf, imgsz, device)["detections"]
        if len(baseline):
            scores = [box_iou(box, r[["x1", "y1", "x2", "y2"]]) for _, r in baseline.iterrows()]
            best = int(np.argmax(scores))
            if scores[best] >= match_iou:
                target_class = int(baseline.iloc[best].class_id)
    images, masked_images, rows = [], [], []
    for level in levels:
        altered = apply_occlusion(image, box, float(level))
        result = detect_image(model, altered, conf, imgsz, device)
        candidates = result["detections"]
        if target_class is not None:
            candidates = candidates[candidates.class_id == target_class]
        candidates = candidates.copy()
        candidates["reference_iou"] = [box_iou(box, r[["x1", "y1", "x2", "y2"]]) for _, r in candidates.iterrows()]
        eligible = candidates[candidates.reference_iou >= match_iou]
        best = eligible.sort_values("reference_iou", ascending=False).iloc[0] if len(eligible) else None
        rows.append({"level": float(level), "matched": best is not None,
                     "confidence": float(best.confidence) if best is not None else 0.,
                     "iou": float(best.reference_iou) if best is not None else 0.,
                     "detections": len(result["detections"]), "inference_ms": result["inference_ms"]})
        images.append(result["image"])
        masked_images.append(altered)
    return {"images": images, "masked_images": masked_images,
            "metrics": pd.DataFrame(rows), "target_class": target_class,
            "reference_box": list(map(float, box)), "match_iou": match_iou}


def load_embedder(device="cpu") -> dict:
    """ResNet18 ImageNet, sem cabeça classificadora; NÃO treinado para Re-ID."""
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(MODEL_DIR / "torch_hub"))
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return {"model": model, "transform": weights.transforms(), "device": device,
            "name": "ResNet18 / ImageNet-1K / embedding 512D"}


def embed_images(embedder: dict, images: Sequence[Image.Image]) -> np.ndarray:
    import torch
    if not images:
        return np.zeros((0, 512), dtype=np.float32)
    batch = torch.stack([embedder["transform"](im.convert("RGB")) for im in images])
    with torch.inference_mode():
        features = embedder["model"](batch.to(embedder["device"]))
        features = torch.nn.functional.normalize(features, dim=1)
    return features.cpu().numpy()


def rank_gallery(embedder: dict, query: Image.Image, gallery: Sequence[Image.Image],
                 names=None) -> pd.DataFrame:
    names = list(names) if names is not None else [f"Candidato {i+1}" for i in range(len(gallery))]
    if len(names) != len(gallery):
        raise ValueError("Nomes e galeria devem ter o mesmo tamanho.")
    if not gallery:
        return pd.DataFrame(columns=["index", "name", "cosine_similarity"])
    vectors = embed_images(embedder, [query, *gallery])
    scores = np.clip(vectors[1:] @ vectors[0], -1, 1)
    return pd.DataFrame({"index": range(len(gallery)), "name": names,
                         "cosine_similarity": scores}).sort_values("cosine_similarity", ascending=False).reset_index(drop=True)


def crop_class_detections(image: Image.Image, detections: pd.DataFrame,
                          class_name: str, padding=.06, minimum_area=1500) -> list[Image.Image]:
    """Recorta detecções de uma classe com margem, preservando apenas caixas úteis."""
    source = image.convert("RGB")
    rows = detections[detections.class_name == class_name].copy()
    if rows.empty:
        return []
    rows["area"] = (rows.x2-rows.x1).clip(lower=0) * (rows.y2-rows.y1).clip(lower=0)
    rows = rows[rows.area >= minimum_area].sort_values("area", ascending=False)
    crops = []
    for _, row in rows.iterrows():
        width, height = row.x2-row.x1, row.y2-row.y1
        pad_x, pad_y = width*padding, height*padding
        box = (max(0, int(row.x1-pad_x)), max(0, int(row.y1-pad_y)),
               min(source.width, int(row.x2+pad_x)), min(source.height, int(row.y2+pad_y)))
        crops.append(source.crop(box))
    return crops


def person_reid_challenges(query: Image.Image, distractors: Sequence[Image.Image],
                           rotation=25., occlusion=.35, brightness=.65) -> dict:
    """Galeria didática: variações do mesmo recorte e outras pessoas como distratores."""
    base = query.convert("RGB")
    rotated = base.rotate(float(rotation), resample=Image.Resampling.BICUBIC,
                          expand=True, fillcolor=(70, 78, 88))
    masked = apply_occlusion(base, [0, 0, base.width, base.height], float(occlusion))
    gallery = [ImageEnhance.Brightness(base).enhance(float(brightness)), rotated,
               masked, ImageOps.mirror(base), *[x.convert("RGB") for x in distractors]]
    names = ["Mesma pessoa · pouca luz", f"Mesma pessoa · rotação {rotation:.0f}°",
             f"Mesma pessoa · oclusão {occlusion:.0%}", "Mesma pessoa · espelhada"]
    names.extend([f"Outra pessoa {i+1}" for i in range(len(distractors))])
    relations = ["Variação controlada do mesmo recorte"]*4 + ["Pessoa diferente na mesma cena"]*len(distractors)
    return {"query": base, "gallery": gallery, "names": names, "relations": relations}


def video_info(source) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo.")
    info = {"fps": float(cap.get(cv2.CAP_PROP_FPS)) or 25.,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    info["duration_s"] = info["frames"] / info["fps"]
    return info


def video_frame(source, frame=0) -> Image.Image:
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame)))
    ok, data = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Quadro {frame} indisponível.")
    return Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))


def _browser_video(source: Path, target: Path) -> Path:
    """H264 via ffmpeg empacotado, sem depender de instalação global."""
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
                        "-an", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(target)], check=True,
                       capture_output=True, timeout=180,
                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
        return target
    except (ImportError, subprocess.SubprocessError, OSError):
        return source


def track_video(weights, source, output_dir, conf=.25, tracker="bytetrack.yaml",
                max_frames=180, device="cpu", line_y=.6, imgsz=640,
                classes=TRAFFIC_CLASSES, occlusion_rect=None, mask_start=0,
                mask_end=None, progress: Callable[[int, int], None] | None = None) -> dict:
    """Rastreia quadros consecutivos; tracker independente por execução.

    line_y e occlusion_rect são normalizados na imagem. Conta cada ID uma vez
    por direção, com histerese de 1% da altura para reduzir ruído na linha.
    IDs e cruzamentos são estimativas do modelo, não identidades/contagens reais.
    """
    if tracker not in {"bytetrack.yaml", "botsort.yaml"}:
        raise ValueError("Tracker permitido: bytetrack.yaml ou botsort.yaml.")
    if not 0 < line_y < 1:
        raise ValueError("A linha deve estar dentro da imagem.")
    if occlusion_rect is not None and (len(occlusion_rect) != 4 or not all(0 <= v <= 1 for v in occlusion_rect)):
        raise ValueError("Retângulo deve conter quatro coordenadas de 0 a 1.")
    model = load_detector(str(weights))
    info = video_info(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "tracking_mpeg4.mp4"
    cap = cv2.VideoCapture(str(source))
    width, height = info["width"], info["height"]
    # H264 exige dimensões pares; redimensionamento só se fonte for ímpar.
    width, height = width - width % 2, height - height % 2
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Codificador de vídeo indisponível.")
    maximum = min(int(max_frames), info["frames"]) if info["frames"] > 0 else int(max_frames)
    records, crossings, trajectories = [], [], defaultdict(list)
    stable_side, counted = {}, set()
    count = 0
    preview = None
    started = time.perf_counter()
    try:
        while count < maximum:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            if occlusion_rect is not None and count >= mask_start and (mask_end is None or count < mask_end):
                x1, y1, x2, y2 = occlusion_rect
                frame[int(y1*height):int(y2*height), int(x1*width):int(x2*width)] = (65, 57, 50)
            result = model.track(source=frame, conf=conf, imgsz=imgsz, device=device,
                                 tracker=tracker, persist=True, classes=classes, verbose=False)[0]
            annotated = result.plot()
            if result.boxes is not None and result.boxes.id is not None:
                for coordinates, ident, cls, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                         result.boxes.id.int().cpu().tolist(), result.boxes.cls.int().cpu().tolist(),
                         result.boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = coordinates
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    name = result.names[cls]
                    records.append([count, count/info["fps"], ident, cls, name, confidence,
                                    x1, y1, x2, y2, cx, cy])
                    trajectories[ident].append((int(cx), int(cy)))
                    points = np.asarray(trajectories[ident][-45:], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [points], False, (230, 230, 0), 2)
                    distance = cy/height-line_y
                    side = 1 if distance > .01 else -1 if distance < -.01 else 0
                    previous = stable_side.get(ident, 0)
                    if side and previous and previous != side:
                        direction = "descendo" if side > previous else "subindo"
                        if (ident, direction) not in counted:
                            counted.add((ident, direction))
                            crossings.append([count, count/info["fps"], ident, name, direction])
                    if side:
                        stable_side[ident] = side
            line_pixel = round(line_y*height)
            cv2.line(annotated, (0, line_pixel), (width, line_pixel), (0, 180, 255), 2)
            cv2.rectangle(annotated, (8, 8), (min(width-8, 525), 58), (30, 36, 48), -1)
            cv2.putText(annotated, f"{tracker.split('.')[0]} | frame {count} | crossings {len(crossings)}",
                        (18, 39), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            writer.write(annotated)
            preview = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            if progress is not None and (count % 10 == 0 or count == maximum-1):
                progress(count+1, maximum)
            count += 1
    finally:
        cap.release()
        writer.release()
    if preview is None:
        raise ValueError("Nenhum quadro processado.")
    processing_seconds = time.perf_counter() - started
    tracks = pd.DataFrame(records, columns=TRACK_COLUMNS)
    events = pd.DataFrame(crossings, columns=CROSSING_COLUMNS)
    tracks.to_csv(destination / "tracks.csv", index=False)
    events.to_csv(destination / "crossings.csv", index=False)
    preview.save(destination / "tracking_preview.png")
    output_path = _browser_video(raw_path, destination / "tracking.mp4")
    stats = {"frames": count, "video_seconds": count/info["fps"], "unique_ids": len(trajectories),
             "crossings": len(events), "processing_seconds": processing_seconds,
             "processing_fps": count/max(processing_seconds, .001), "tracker": tracker,
             "device": device, "conf": conf, "model": Path(weights).name,
             "source_fps": info["fps"], "line_y": line_y,
             "imgsz": imgsz, "occlusion_rect": occlusion_rect,
             "mask_start": mask_start, "mask_end": mask_end,
             "browser_compatible": output_path.name == "tracking.mp4"}
    return {"video_path": str(output_path), "tracks": tracks, "crossings": events,
            "preview": preview, "stats": stats}
