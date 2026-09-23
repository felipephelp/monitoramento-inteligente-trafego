"""Conteúdo didático, referências e roteiro do laboratório."""

SOURCES = [
    ("YOLO11: documentação oficial", "https://docs.ultralytics.com/models/yolo11/", "Detector pré-treinado no COCO; não adaptado especificamente à câmera desta aula."),
    ("ByteTrack — ECCV 2022", "https://arxiv.org/abs/2110.06864", "Associação em duas etapas; detecções de baixa confiança podem recuperar trajetórias."),
    ("BoT-SORT — 2022", "https://arxiv.org/abs/2206.14651", "Associação de movimento, compensação de câmera e aparência opcional."),
    ("DETR — ECCV 2020", "https://arxiv.org/abs/2005.12872", "Detecção como predição de conjuntos com Transformer."),
    ("Vision Transformer — ICLR 2021", "https://arxiv.org/abs/2010.11929", "Patches e atenção; classificação por si só não produz rastreamento ou eventos."),
    ("ResNet18 / torchvision", "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html", "Baseline genérico ImageNet usado no ranking de aparência deste laboratório."),
    ("CityFlow — CVPR 2019", "https://arxiv.org/abs/1903.09254", "Benchmark de veículos em múltiplas câmeras; oferece contexto de avaliação além de uma imagem."),
    ("HOTA — IJCV", "https://arxiv.org/abs/2009.07736", "Métrica que combina detecção e associação em tracking com anotações de referência."),
    ("NVIDIA Metropolis", "https://developer.nvidia.com/metropolis", "Plataforma para compor aplicações de vídeo analítico de borda a nuvem."),
    ("AXIS Object Analytics", "https://www.axis.com/products/axis-object-analytics", "Produto de análise em câmeras para detectar, classificar, rastrear e contar."),
]

LESSON_PLAN = [
    ("00–08", "O problema urbano", "Uma câmera observa; uma operação precisa de contagem, trajetórias e decisões.", "Visão geral"),
    ("08–18", "Detecção e modelos", "Altere confiança e resolução. Observe caixas que desaparecem e objetos pequenos.", "Detecção"),
    ("18–29", "Rastreamento", "Execute ByteTrack. Leia IDs e cruzamentos de linha; compare com BoT-SORT.", "Tracking"),
    ("29–34", "Contagem", "Compare cruzamentos por direção e discuta erros causados por IDs fragmentados.", "Contagem"),
    ("34–42", "Reidentificação", "Compare veículos e pessoas sob iluminação, rotação, semelhança e oclusão.", "Re-ID"),
    ("42–49", "Oclusão e falhas", "Cubra progressivamente um objeto e execute novamente o detector.", "Oclusão"),
    ("49–56", "Da pesquisa à operação", "Relacione aplicações, homografia, arquitetura e produtos.", "Pesquisa & operação"),
    ("56–60", "Avaliação e discussão", "Defina um piloto, ground truth e critérios operacionais de aceitação.", "Guia & fontes"),
]

GLOSSARY = {
    "Detecção": "Localiza instâncias em um quadro e atribui classes e scores. Não estabelece identidade persistente.",
    "Tracking": "Associa observações ao longo dos quadros de uma sequência. Um ID do tracker é uma hipótese temporária.",
    "Contagem": "Transforma trajetórias em eventos quando um ID cruza uma linha ou muda entre zonas definidas.",
    "Re-ID": "Compara descritores de aparência de objetos para ordenar candidatos, combinando-os depois com restrições da rede viária.",
    "Confiança": "Score do detector. Não deve ser lido automaticamente como probabilidade calibrada de acerto.",
    "IoU": "Área da interseção de duas caixas dividida pela área da união. Serve para associação espacial, não para identidade.",
    "mAP": "Resumo de precisão–recall por classe e limiares IoU; só existe com anotações de referência e protocolo definido.",
    "IDF1 / HOTA": "Métricas de identidade e associação em sequências anotadas. Não podem ser inferidas só olhando um vídeo.",
    "Homografia": "Transformação projetiva entre planos. Permite calibrar pontos do pavimento sob hipóteses geométricas.",
}
