# Monitoramento inteligente do tráfego urbano

Laboratório local para acompanhar uma apresentação de 60 minutos de Felipe Euphrasio. O aplicativo contém exemplos reais de trânsito, figuras dos materiais IPT/defesa e experimentos com modelos reais.

## Instalar e abrir

No Windows, com Python 3.12 instalado:

```powershell
git clone https://github.com/felipephelp/monitoramento-inteligente-trafego.git
cd monitoramento-inteligente-trafego
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_examples.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Depois, abra <http://localhost:8502>. Nas próximas execuções, use **INICIAR.bat** (duplo clique) ou:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

O servidor fica vinculado a `127.0.0.1`, acessível apenas nesta máquina. Feche o terminal ou pressione `Ctrl+C` para encerrá-lo. Se ele já estiver executando em segundo plano, basta acessar o endereço.

## Experimentos

| Página | O que executa | O que observar |
|---|---|---|
| Detecção | YOLO11n pré-treinado no COCO, com confiança/resolução variáveis | Caixas, classe, score e objetos perdidos |
| Tracking | YOLO11n + ByteTrack ou BoT-SORT padrão | IDs temporários e trajetórias em câmera fixa ou vídeo de drone |
| Contagem | YOLO11n + tracker + linha virtual | Eventos por classe e direção; geometria fixa versus plataforma móvel |
| Re-ID de veículos | ResNet18 ImageNet sem classificador, embeddings 512D e cosseno | Ranking visual com imagem, posição e similaridade de cada candidato |
| Re-ID de pessoas | Detecção YOLO + ResNet18 em recortes de pessoas | Ranking visual sob oclusão, rotação, iluminação, semelhança e distratores |
| Oclusão | YOLO executado a 0, 20, 40, 60 e 80% de máscara | Retenção da detecção e score da correspondência |
| Pesquisa & operação | Figuras extraídas dos materiais originais + fluxos | CNN/Transformer, homografia, arquitetura e produtos |
| Aplicações | Evidências reais do laboratório | Detecção, tracking, contagem, Re-ID, comportamento e anomalias |

As páginas aceitam imagens/vídeos próprios. Tracking e contagem apresentam cenários de câmera fixa e drone; para executar o cenário aéreo, envie um vídeo de drone estabilizado. No Re-ID, recorte o objeto antes de enviar e use uma consulta mais duas ou mais imagens de galeria.

**Demonstração ≠ benchmark.** Não se apresentam mAP, recall, IDF1 ou HOTA sem anotações de referência. Os cruzamentos e IDs são estimativas. A curva de oclusão usa a detecção original como referência, não ground truth; pode não ser monotônica. Similaridade de aparência não é probabilidade de identidade. O Re-ID usa um baseline ImageNet genérico e o exemplo padrão é um experimento de perturbação de recortes, não uma avaliação multicâmera. BoT-SORT usa a configuração padrão, sem módulo Re-ID ativado.

## Roteiro sugerido

1. Visão geral: da câmera ao indicador (8 min).
2. Detecção: variar confiança e resolução (10 min).
3. Tracking e contagem: executar o vídeo e interpretar cruzamentos (12 min).
4. Re-ID: comparar veículos e pessoas sob diferentes perturbações (10 min).
5. Oclusão: prever, executar e discutir a curva (7 min).
6. Aplicações, pesquisa e operação: comportamento, anomalias, arquitetura e piloto (8 min).
7. Discussão e conclusões (5 min).

## Estrutura e reprodução

- `app.py`: navegação.
- `app_pages/`: dez páginas didáticas.
- `vision.py`: backend puro, sem dependência do Streamlit.
- `ui.py`: modelos em cache e auxiliares da interface.
- `prepare_examples.py`: baixa pesos/exemplos e prepara artefatos reais.
- `assets/examples/`: exemplos prontos para a aula.
- `assets/models/`: pesos locais baixados no preparo (não versionados).
- `assets/referencias/`: figuras dos materiais fornecidos.
- `exports/`: resultados preparados e figuras para os slides (gerados localmente).
- `results/`: saídas de tracking por execução (geradas localmente).
- `uploads/`: vídeos enviados pelo usuário, identificados por hash.
- `tests/`: testes de backend e de interface.

O ambiente `.venv` está instalado no disco D com Python 3.12. As versões principais estão em `requirements.txt`. Para recriar:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_examples.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os pesos e exemplos precisam de internet apenas no preparo inicial. Depois de baixados, as inferências são locais. O backend funciona em CPU; GPU aparece na interface somente quando o PyTorch instalado reconhece CUDA. A instalação CPU entregue prioriza portabilidade e não faz uso da RTX sem uma build CUDA compatível.

## Fontes, dados e uso

Veja `assets/ATRIBUICOES.md` e a página **Guia & fontes** para URLs e origem das imagens/vídeos. YOLO/Ultralytics e pesos têm condições próprias de licenciamento; avalie essas condições antes de incorporar o laboratório a um produto. As figuras da tese e da apresentação IPT vieram dos documentos fornecidos pelo usuário e preservam sua atribuição.

Uploads e resultados permanecem nesta pasta. O app não publica os arquivos em serviços externos. Para uma implantação real, defina retenção, autorização de uso das imagens e política de acesso. Não utilize estes resultados demonstrativos como base autônoma de fiscalização ou identificação.
