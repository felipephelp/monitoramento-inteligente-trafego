"""Gera resultados REAIS para a aula. Execute com o Python do ambiente .venv.

Mantém os insumos oficiais em assets/examples e grava artefatos em exports.
Os resultados dependem do hardware e das versões instaladas; não são benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "assets" / "settings"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import torch

from vision import (load_detector, detect_image, occlusion_experiment, load_embedder,
                    rank_gallery, apply_occlusion, track_video, video_frame,
                    crop_class_detections, person_reid_challenges)

EXAMPLES = ROOT / "assets" / "examples"
EXPORTS = ROOT / "exports"
NAVY, BLUE, ORANGE, GRAY = "#15334C", "#0B79A7", "#E77D35", "#607181"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.labelcolor": NAVY, "text.color": NAVY, "font.size": 13})


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def detection_outputs(model, device):
    metadata = {}
    for name in ["vehicles", "bus", "urban_traffic_frame"]:
        source = Image.open(EXAMPLES / f"{name}.jpg").convert("RGB")
        resolution = 1920 if name == "vehicles" else 640
        result = detect_image(model, source, conf=.25, imgsz=resolution, device=device)
        result["image"].save(EXPORTS / f"detection_{name}.png")
        result["detections"].to_csv(EXPORTS / f"detection_{name}.csv", index=False)
        metadata[name] = {"n_detections": len(result["detections"]),
                          "imgsz": resolution,
                          "wall_ms": result["inference_ms"],
                          "model_inference_ms": result["model_inference_ms"],
                          "classes": {str(k): int(v) for k,v in result["detections"].class_name.value_counts().items()}}
        if name == "urban_traffic_frame":
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
            for ax, im, title in zip(axes, [source, result["image"]], ["Entrada: tráfego urbano real", "Saída: YOLO11n / classes e confiança"]):
                ax.imshow(im)
                ax.set_title(title, loc="left", fontsize=17, pad=14)
                ax.axis("off")
            fig.text(.02, .025, "Inferência local · Pesos COCO · Confiança mínima 0,25 · Fonte: Ultralytics dashboard_sample.mp4", fontsize=11, color=GRAY)
            fig.tight_layout(rect=[0, .04, 1, 1])
            fig.savefig(EXPORTS / "detection_before_after.png", dpi=160, facecolor="white")
            plt.close(fig)
    return metadata


def resolution_outputs(model, device):
    source = Image.open(EXAMPLES / "vehicles.jpg").convert("RGB")
    rows = []
    fig, axes = plt.subplots(1,2,figsize=(16,6.5),facecolor="white")
    for ax, size in zip(axes,[640,1920]):
        result = detect_image(model,source,imgsz=size,device=device)
        rows.append({"imgsz":size,"detections":len(result["detections"]),"model_inference_ms":result["model_inference_ms"]})
        ax.imshow(result["image"])
        ax.set_title(f"Entrada {size} px · {len(result['detections'])} detecções",loc="left",fontsize=18)
        ax.axis("off")
    fig.suptitle("Objetos pequenos e mudança de ponto de vista",x=.02,ha="left",fontsize=22,fontweight="bold")
    fig.text(.02,.03,"Mesma imagem aérea, YOLO11n e limiar 0,25 · Mais resolução preserva detalhes, mas aumenta o custo e não elimina o erro",fontsize=11,color=GRAY)
    fig.tight_layout(rect=[0,.07,1,.91])
    fig.savefig(EXPORTS/"resolution_comparison.png",dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(EXPORTS/"resolution_comparison.csv",index=False)
    return rows


def occlusion_outputs(model, device):
    image = Image.open(EXAMPLES / "urban_traffic_frame.jpg").convert("RGB")
    detection = detect_image(model, image, conf=.25, device=device)["detections"]
    cars = detection[detection.class_id == 2].copy()
    # Objeto grande e integralmente dentro da imagem; evita bordas truncadas.
    cars = cars[(cars.x1 > 20) & (cars.y1 > 20) & (cars.x2 < image.width-20) & (cars.y2 < image.height-20)]
    cars["area"] = (cars.x2-cars.x1)*(cars.y2-cars.y1)
    reference = cars.sort_values("area", ascending=False).iloc[0]
    box = reference[["x1", "y1", "x2", "y2"]].tolist()
    result = occlusion_experiment(model, image, box, levels=[0, .2, .4, .6, .8, 1.],
                                  conf=.25, device=device, target_class=2)
    result["metrics"].to_csv(EXPORTS / "occlusion_curve.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
    pad = 80
    crop = (max(0, box[0]-pad), max(0, box[1]-pad), min(image.width,box[2]+pad), min(image.height,box[3]+pad))
    for ax, annotated, row in zip(axes.flat, result["images"], result["metrics"].itertuples()):
        ax.imshow(annotated.crop(crop))
        status = f"conf. {row.confidence:.2f}" if row.matched else "sem correspondência"
        ax.set_title(f"Máscara {row.level:.0%} · {status}", loc="left", fontsize=15, color=BLUE if row.matched else ORANGE)
        ax.axis("off")
    fig.suptitle("Oclusão controlada: o mesmo carro, seis entradas", x=.02, ha="left", fontsize=22, fontweight="bold")
    fig.text(.02, .018, "Percentual da bounding box mascarada · Referência = detecção sem máscara · Correspondência: mesma classe e IoU ≥ 0,30", fontsize=10.8, color=GRAY)
    fig.subplots_adjust(left=.02,right=.985,top=.89,bottom=.08,wspace=.035,hspace=.22)
    fig.savefig(EXPORTS / "occlusion_montage.png", dpi=170, facecolor="white")
    plt.close(fig)
    metrics = result["metrics"]
    fig, ax = plt.subplots(figsize=(11, 5.7), facecolor="white")
    ax.plot(metrics.level*100, metrics.confidence, "o-", color=BLUE, linewidth=3, markersize=9)
    for r in metrics.itertuples():
        ax.annotate(f"{r.confidence:.2f}" if r.matched else "ausente", (r.level*100,r.confidence),
                    xytext=(0, 12), textcoords="offset points", ha="center", color=BLUE if r.matched else ORANGE)
    ax.axhline(.25, linestyle="--", color=GRAY, linewidth=1, label="Limiar de confiança = 0,25")
    ax.set(xlabel="Área da caixa de referência mascarada (%)", ylabel="Confiança da detecção correspondente",
           ylim=(-.04,1.08), xticks=metrics.level*100, title="Oclusão × estabilidade de uma detecção")
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    fig.text(.08,.018,"Experimento pontual executado no laboratório. Confiança não equivale a probabilidade calibrada nem a acurácia.",fontsize=10,color=GRAY)
    fig.tight_layout(rect=[0,.045,1,1])
    fig.savefig(EXPORTS / "occlusion_curve.png", dpi=180)
    plt.close(fig)
    return {"reference_box":box, "rows":json.loads(metrics.to_json(orient="records"))}


def reid_outputs(model, device):
    image = Image.open(EXAMPLES / "urban_traffic_frame.jpg").convert("RGB")
    table = detect_image(model, image, device=device)["detections"]
    cars = table[(table.class_id == 2) & (table.x1 > 10) & (table.x2 < image.width-10)].copy()
    cars["area"] = (cars.x2-cars.x1)*(cars.y2-cars.y1)
    cars = cars.sort_values("area",ascending=False).head(6)
    crops = [image.crop(tuple(map(int,row[["x1","y1","x2","y2"]]))) for _,row in cars.iterrows()]
    query = crops[0]
    reid_dir = EXAMPLES / "reid"
    reid_dir.mkdir(exist_ok=True)
    query.save(reid_dir / "query.png")
    query.save(EXPORTS / "reid_query.png")
    gallery = [crops[1], ImageEnhance.Brightness(query).enhance(.6), crops[2],
               apply_occlusion(query,[0,0,query.width,query.height], .35), crops[3], ImageOps.mirror(query)]
    names = ["B · outro veículo", "A · menos luz", "C · outro veículo", "A · 35% oclusão", "D · outro veículo", "A · espelhado"]
    relations = ["Outro recorte da mesma imagem", "Mesmo recorte de consulta: brilho reduzido artificialmente",
                 "Outro recorte da mesma imagem", "Mesmo recorte de consulta: máscara artificial de 35% da caixa",
                 "Outro recorte da mesma imagem", "Mesmo recorte de consulta: espelhamento artificial"]
    payload = {"query":"reid/query.png", "query_name":"A · consulta", "gallery":[],
               "experiment":"Perturbações controladas do MESMO recorte e distratores de uma única imagem; não é ensaio multicâmera.",
               "model":"ResNet18 ImageNet-1K, embedding 512D normalizado; sem treinamento para Re-ID.",
               "source":"Ultralytics dashboard_sample.mp4, frame 0"}
    for index,(crop,name,relation) in enumerate(zip(gallery,names,relations),1):
        relative = f"reid/candidate_{index:02d}.png"
        crop.save(EXAMPLES / relative)
        crop.save(EXPORTS / f"reid_candidate_{index:02d}.png")
        payload["gallery"].append({"path":relative,"name":name,"relation":relation})
    save_json(EXAMPLES / "reid_manifest.json", payload)
    embedder = load_embedder(device)
    ranking = rank_gallery(embedder,query,gallery,names)
    ranking.to_csv(EXPORTS / "reid_ranking.csv",index=False)
    save_json(EXPORTS / "reid_manifest.json",payload)
    fig = plt.figure(figsize=(16,7.5),facecolor="white")
    grid = fig.add_gridspec(2,4,width_ratios=[1.2,1,1,1])
    ax = fig.add_subplot(grid[:,0])
    ax.imshow(query)
    ax.set_title("CONSULTA A",loc="left",color=NAVY,fontsize=18)
    ax.axis("off")
    for rank,row in enumerate(ranking.itertuples(),1):
        ax = fig.add_subplot(grid[(rank-1)//3,1+(rank-1)%3])
        ax.imshow(gallery[row.index])
        ax.set_title(f"{rank}º · {row.name}\ncosseno = {row.cosine_similarity:.3f}",loc="left",fontsize=13)
        ax.axis("off")
    fig.suptitle("Re-ID didático: similaridade de aparência com uma rede real",x=.02,ha="left",fontsize=21,fontweight="bold")
    fig.text(.02,.014,"ResNet18 ImageNet, 512 dimensões · Mesma imagem e perturbações artificiais · Cosseno não é probabilidade de identidade",fontsize=11,color=GRAY)
    fig.tight_layout(rect=[0,.05,1,.93])
    fig.savefig(EXPORTS / "reid_gallery.png",dpi=170)
    plt.close(fig)
    return json.loads(ranking.to_json(orient="records"))


def person_reid_outputs(model, device):
    image = Image.open(EXAMPLES / "bus.jpg").convert("RGB")
    detected = detect_image(model, image, conf=.20, device=device)["detections"]
    people = crop_class_detections(image, detected, "person")
    if len(people) < 3:
        raise RuntimeError("O exemplo bus.jpg não produziu recortes suficientes de pessoas.")
    demo = person_reid_challenges(people[0], people[1:4], rotation=25, occlusion=.35, brightness=.65)
    folder = EXAMPLES / "person_reid"
    folder.mkdir(exist_ok=True)
    demo["query"].save(folder / "query.png")
    demo["query"].save(EXPORTS / "person_reid_query.png")
    gallery_paths = []
    for index, image_item in enumerate(demo["gallery"], 1):
        name = f"candidate_{index:02d}.png"
        image_item.save(folder / name)
        image_item.save(EXPORTS / f"person_reid_{name}")
        gallery_paths.append(f"person_reid/{name}")
    embedder = load_embedder(device)
    ranking = rank_gallery(embedder, demo["query"], demo["gallery"], demo["names"])
    ranking["relation"] = [demo["relations"][int(i)] for i in ranking["index"]]
    ranking.to_csv(EXPORTS / "person_reid_ranking.csv", index=False)
    payload = {"query": "person_reid/query.png", "gallery": gallery_paths,
               "names": demo["names"], "relations": demo["relations"],
               "source": "Ultralytics bus.jpg; variações artificiais do mesmo recorte e outras pessoas da cena.",
               "model": "ResNet18 ImageNet-1K, baseline genérico de aparência."}
    save_json(EXAMPLES / "person_reid_manifest.json", payload)
    save_json(EXPORTS / "person_reid_manifest.json", payload)
    return json.loads(ranking.to_json(orient="records"))


def tracking_outputs(device, max_frames):
    result = track_video("yolo11n.pt",EXAMPLES/"traffic.mp4",EXPORTS/"tracking",
                         max_frames=max_frames,device=device,line_y=.55)
    result["preview"].save(EXPORTS / "tracking_preview.png")
    result["tracks"].to_csv(EXPORTS / "tracking_tracks.csv",index=False)
    result["crossings"].to_csv(EXPORTS / "tracking_crossings.csv",index=False)
    fig, ax = plt.subplots(figsize=(13.6,8),facecolor="white")
    ax.imshow(video_frame(EXAMPLES/"traffic.mp4",0),alpha=.43)
    for identity,group in result["tracks"].groupby("track_id"):
        if len(group) > 8:
            ax.plot(group.cx,group.cy,linewidth=2.5)
            ax.text(group.cx.iloc[-1],group.cy.iloc[-1],str(identity),fontsize=9,
                    bbox=dict(facecolor="white",alpha=.75,edgecolor="none",pad=1))
    ax.axhline(720*.55,color=ORANGE,linestyle="--",linewidth=2)
    ax.set_title("Rastreamento real: IDs e trajetórias estimadas ao longo do clipe",loc="left",fontsize=20,pad=12)
    ax.axis("off")
    fig.text(.03,.015,f"YOLO11n + ByteTrack · {result['stats']['frames']} quadros consecutivos · IDs do rastreador, sem ground truth de identidade",fontsize=11,color=GRAY)
    fig.tight_layout(rect=[0,.03,1,1])
    fig.savefig(EXPORTS/"tracking_trajectories.png",dpi=160)
    plt.close(fig)
    masked = track_video("yolo11n.pt",EXAMPLES/"traffic.mp4",EXPORTS/"tracking_occluded",
                         max_frames=max_frames,device=device,line_y=.55,
                         occlusion_rect=(.38,.1,.56,.95),mask_start=35,mask_end=110)
    masked["preview"].save(EXPORTS/"tracking_occluded_preview.png")
    comparison = pd.DataFrame([{**result["stats"],"scenario":"original"},{**masked["stats"],"scenario":"máscara artificial"}])
    comparison.to_csv(EXPORTS/"tracking_comparison.csv",index=False)
    fig, axes = plt.subplots(1,2,figsize=(16,6.5),facecolor="white")
    for ax, data, title in zip(axes,[result,masked],["Vídeo original", "Obstrução artificial durante o rastreamento"]):
        ax.imshow(video_frame(data["video_path"],60))
        ax.set_title(title,loc="left",fontsize=16)
        ax.axis("off")
    fig.suptitle("Oclusão também afeta a continuidade das trajetórias",x=.02,ha="left",fontsize=22,fontweight="bold")
    fig.text(.02,.034,f"6 segundos · Original: {result['stats']['unique_ids']} IDs estimados · Com máscara: {masked['stats']['unique_ids']} IDs · IDs extras sugerem fragmentação; não são contagem real",fontsize=11,color=GRAY)
    fig.tight_layout(rect=[0,.07,1,.91])
    fig.savefig(EXPORTS/"tracking_comparison.png",dpi=170)
    plt.close(fig)
    return {"original":result["stats"],"occluded":masked["stats"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--frames",type=int,default=180)
    parser.add_argument("--skip-tracking",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    EXPORTS.mkdir(exist_ok=True)
    start = time.perf_counter()
    report = {"model":"yolo11n.pt", "device":args.device,"conf":.25,"imgsz":640,
              "torch_version":torch.__version__,"generated_at":time.strftime("%Y-%m-%d %H:%M:%S"),
              "interpretation":"Inferências reais ilustrativas. Sem ground truth, sem estimativa de acurácia ou promessa operacional."}
    print("Carregando detector...",flush=True)
    model = load_detector()
    print("Detecção...",flush=True)
    report["detection"] = detection_outputs(model,args.device)
    report["resolution"] = resolution_outputs(model,args.device)
    print("Oclusão...",flush=True)
    report["occlusion"] = occlusion_outputs(model,args.device)
    print("Re-ID baseline ResNet18...",flush=True)
    report["reid"] = reid_outputs(model,args.device)
    print("Re-ID de pessoas: oclusão, rotação e semelhança...",flush=True)
    report["person_reid"] = person_reid_outputs(model,args.device)
    if not args.skip_tracking:
        print("Tracking original e oclusão...",flush=True)
        report["tracking"] = tracking_outputs(args.device,args.frames)
    report["elapsed_s"] = time.perf_counter()-start
    save_json(EXPORTS/"experiment_report.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
