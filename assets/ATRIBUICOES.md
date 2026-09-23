# Fontes dos ativos e modelos

Download para demonstração local educacional em 17/09/2026.

- `examples/bus.jpg`: exemplo oficial Ultralytics, https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg
- `examples/vehicles.jpg`: exemplo aéreo oficial Ultralytics, https://github.com/ultralytics/assets/releases/download/v0.0.0/vehicles.jpg
- `examples/traffic.mp4` e `examples/urban_traffic.mp4`: vídeo de tráfego veicular Ultralytics (nome original `dashboard_sample.mp4`), https://github.com/ultralytics/assets/releases/download/v0.0.0/dashboard_sample.mp4
- `examples/pedestrian_circulation.mp4`: vídeo de circulação de pedestres Ultralytics (nome original `solutions_ci_demo.mp4`), https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4
- Catálogo de origem: https://github.com/ultralytics/assets/releases/tag/v0.0.0
- Detector YOLO11n pré-treinado em COCO, https://docs.ultralytics.com/models/yolo11/
- Tracking ByteTrack e BoT-SORT via Ultralytics, https://docs.ultralytics.com/modes/track/
- ResNet18 com pesos ImageNet-1K: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

Os arquivos `examples/generated/` são saídas reais do código `prepare_examples.py` executado localmente. Não são benchmarks de acurácia. As máscaras de oclusão são perturbações artificiais controladas; a fração refere-se à área da caixa de referência, não ao objeto segmentado. A tabela de retenção usa a detecção original como referência, não uma anotação humana.

A galeria Re-ID usa recortes de veículos em instantes diferentes da MESMA câmera; IDs iniciais vêm do tracker e são pseudo-rótulos que precisam de conferência visual. ResNet18 é um baseline de similaridade de aparência, sem treinamento específico em reidentificação de veículos. Similaridade cosseno não é probabilidade de identidade. Nenhum resultado aqui demonstra desempenho entre câmeras diferentes.

Verifique as licenças dos respectivos fornecedores para redistribuição ou uso comercial. A biblioteca/modelos Ultralytics adotam opções AGPL-3.0 ou licença Enterprise; https://www.ultralytics.com/license. Os recortes mantêm a atribuição do vídeo de origem.
