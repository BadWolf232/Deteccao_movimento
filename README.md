# Detecção de movimento em videomonitoramento

Trabalho de Processamento de Imagens com OpenCV.

- **Autores:** Felipe Bourqui, Luiz Fogaça e Vitor Martins de Souza (Unisinos, Ciência da Computação)
- **Combinação sorteada:** 12
- **Tema:** 3 – Detecção de movimento em videomonitoramento
- **Subtema:** PETS 2006
- **Base:** CDnet 2014 (Change Detection Benchmark), categoria *Baseline*, sequência `PETS2006` (1.200 frames, 720×576)
- **Parte da base utilizada:** a sequência completa (frames 1–1200). O intervalo avaliado, definido pelo `temporalROI.txt`, é **300–1200**.

O programa compara dois métodos de detecção de movimento:

1. **Diferença entre frames consecutivos** (solução inicial).
2. **Modelo temporal de fundo com atualização progressiva** (média móvel exponencial, implementada com NumPy, com versão de atualização seletiva).

Os dois usam o mesmo pós-processamento (ROI, abertura, fechamento, filtro de área e caixas delimitadoras) e são avaliados com as máscaras de referência da base.

## Como funciona

Cada frame passa por quatro etapas, na ordem temporal. Todas as decisões e parâmetros estão no código, com um comentário ao lado.

**1. Pré-processamento.** O frame é convertido para tons de cinza e recebe um borrão gaussiano 5×5, para que o ruído da câmera não seja confundido com movimento.

**2. Máscara bruta de movimento** (é aqui que os dois métodos diferem):

- **Diferença de frames.** Compara o frame atual com o anterior: `|frame_t − frame_(t−1)|`. Pixels que mudaram mais de 20 níveis (de 0 a 255) viram movimento. Só enxerga o que se moveu entre dois frames seguidos, então objetos parados desaparecem e, em objetos de cor uniforme, aparece principalmente a borda.
- **Modelo de fundo com atualização progressiva.** Mantém uma imagem do fundo da cena, que começa como o primeiro frame. Pixels que diferem do fundo mais que um limiar viram movimento. Depois de cada frame, o fundo é atualizado por média móvel exponencial: `B = B + taxa · (frame − B)`. Assim ele acompanha mudanças lentas da cena. Na **versão seletiva**, os pixels em movimento usam taxa dez vezes menor (`α/10`), para que uma pessoa parada não seja absorvida pelo fundo. Os pixels "em movimento" são os da máscara final do frame, já depois do pós-processamento. O modelo é implementado diretamente com NumPy, sem subtratores de fundo prontos do OpenCV.

**3. Pós-processamento (igual para os dois).**

1. Aplica a ROI (região de interesse da base).
2. Abertura morfológica 3×3: remove pontos isolados de ruído.
3. Fechamento morfológico 9×9: preenche buracos e une fragmentos de um mesmo objeto.
4. Extrai as regiões conectadas e descarta as menores que 80 px.
5. O que sobra é a **máscara final**. Dela saem as caixas delimitadoras, o número de regiões e a área em movimento de cada frame.

**4. Avaliação com a referência da base.** A máscara final é comparada, pixel a pixel, com a máscara de referência (`groundtruth`) dos frames avaliados (300–1200):

- TP: detectado como movimento e a referência diz movimento.
- FP: detectado como movimento e a referência diz fundo.
- FN: não detectado, mas a referência diz movimento.
- Os rótulos 170 (desconhecido), 50 (sombra) e 85 (fora da ROI) são **ignorados**: nesses pixels, a detecção não conta como acerto nem como erro.

Daí saem precisão, recall, F1 e IoU, somando todos os frames do conjunto. O tempo de processamento é medido por frame (sem contar leitura de arquivo, avaliação e gravação de vídeo).

**Separação dos dados.** Os frames são divididos no tempo:

| Frames | Uso |
|---|---|
| 1–299 | aquecimento (o modelo de fundo aprende a cena; não são avaliados) |
| 300–599 | **ajuste**: escolha dos parâmetros do modelo de fundo |
| 600–1200 | **avaliação final**: não participa da escolha de parâmetros |

O programa testa 8 configurações do modelo de fundo (α ∈ {0,01; 0,05}, limiar ∈ {20; 30}, atualização seletiva ou não), escolhe a de maior F1 em 300–599 e só então a avalia em 600–1200. Os parâmetros da diferença de frames são fixos e não foram otimizados.

## Resultados obtidos

Avaliação final (frames 600–1200), tabela completa em `saidas/metricas_finais.csv`:

| Método | Precisão | Recall | F1 | IoU | Tempo por frame |
|---|---|---|---|---|---|
| Diferença de frames | 0,723 | 0,224 | 0,342 | 0,206 | 1,66 ms |
| Modelo de fundo (α=0,01, limiar 20, seletivo) | 0,772 | 0,946 | 0,850 | 0,740 | 6,52 ms |

O modelo de fundo detecta bem mais movimento (recall), principalmente pessoas paradas, ao custo de cerca de 4 vezes mais tempo por frame. Os tempos foram medidos em um Ryzen 5 3600 com 16 GB de RAM, em uma única execução, e variam um pouco entre execuções. A análise de onde cada método falha (por trechos de 100 frames, sombra e reflexo no piso, objeto residual) está no relatório e é reproduzida por `python analise_falhas.py` (ver Execução).

## Instalação

Requer Python 3.12 (testado com 3.12.10) no Windows.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependências: `numpy` e `opencv-python` (versões em `requirements.txt`).

## Dados

A sequência `PETS2006` (categoria *Baseline*) já está no repositório, em `assets/PETS2006/`. Para obtê-la de novo, baixe em
<https://changedetection.vitalab.usherbrooke.ca/dataset2014/> e extraia de modo que a estrutura fique:

```
assets/PETS2006/
    input/          in000001.jpg ... in001200.jpg
    groundtruth/    gt000001.png ... gt001200.png
    ROI.bmp
    temporalROI.txt
```

## Execução

Na raiz do projeto:

```powershell
python ptz.py        # pipeline completo (ajuste, avaliação final, vídeos, tabelas, gráfico)
python ptz.py ver    # só a diferença de frames em janelas, sem ajuste nem gravação
python analise_falhas.py   # análise de falhas do relatório (rodar depois de ptz.py)
```

No pipeline completo, o ajuste de parâmetros roda sem janelas. Na execução final, o vídeo anotado e a máscara de cada método são exibidos em janelas enquanto o programa processa (primeiro a diferença de frames, depois o modelo de fundo). A tecla `e`, com uma janela em foco, fecha as janelas e o programa segue processando sem exibir. Sem fechar, o pipeline completo leva cerca de 2,5 minutos por causa da exibição; sem janelas, cerca de 1,5 minuto.

O que ele faz, em ordem:

1. Ajusta os parâmetros do modelo de fundo (8 configurações) usando **somente** os frames 300–599.
2. Escolhe a configuração de maior F1 nesse conjunto.
3. Roda os dois métodos nos frames 1–1200 e avalia 300–1200. Os resultados do conjunto de avaliação final (600–1200) são reportados separadamente.
4. Grava vídeos anotados, tabelas, série temporal, gráfico e imagens de análise de falhas.

## Estrutura do código

| Arquivo | Conteúdo |
|---|---|
| `ptz.py` | Leitura da sequência, pré-processamento, os dois métodos, pós-processamento, avaliação e `main` |
| `resultados.py` | Geração de saídas: CSVs, gráfico, imagens de análise e desenho das caixas |
| `analise_falhas.py` | Análise de falhas do relatório: métricas por trechos de 100 frames, falsos positivos do fundo e imagens de erro (lê `saidas/`, roda o fundo de novo e não altera as saídas de `ptz.py`) |
| `requirements.txt` | Dependências com versões |
| `assets/PETS2006/` | Dados da base (ver seção Dados) |
| `saidas/` | Resultados gerados pelo programa (incluídos no repositório) |

Os parâmetros (limiares, tamanhos de kernel, área mínima) estão no próprio código, com um comentário ao lado.

## Saídas (pasta `saidas/`)

| Arquivo | Descrição |
|---|---|
| `video_diff.mp4`, `video_fundo.mp4` | Vídeo anotado com caixas delimitadoras de cada método |
| `serie_temporal_diff.csv`, `serie_temporal_fundo.csv` | Por frame: nº de regiões e área em movimento (px) |
| `serie_temporal.png` | Gráfico das duas séries |
| `ajuste.csv` | Diferença de frames (1 configuração) e modelo de fundo (8 configurações) no conjunto de ajuste (300–599) |
| `metricas_finais.csv` | Precisão, recall, F1, IoU e tempo, por método e conjunto |
| `metricas_por_frame.csv` | TP, FP, FN, métricas e tempo de cada frame avaliado |
| `rotulos_referencia.csv` | Distribuição dos rótulos da referência no intervalo avaliado |
| `analise/` | Melhores e piores frames de cada método, com mapa de erros (`*_melhor_*`, `*_pior_*`), e os frames citados no relatório (`falha_*`, `fantasma_*`, gerados por `analise_falhas.py`) |
| `metricas_por_janela.csv` | Precisão, recall, F1 e IoU por trechos de 100 frames do teste final (gerado por `analise_falhas.py`) |
| `fp_fundo_por_janela.csv` | Falsos positivos do modelo de fundo por trecho: por frame, proximidade do movimento da referência e pixels em local fixo (gerado por `analise_falhas.py`) |

## Tratamento dos rótulos da referência

| Valor | Significado | Tratamento |
|---|---|---|
| 0 | fundo | negativo (dentro da ROI) |
| 255 | movimento | positivo |
| 170 | desconhecido | **ignorado** na avaliação |
| 50 | sombra | **ignorado** (não ocorre nesta sequência; na base oficial conta como fundo) |
| 85 | fora da ROI | **ignorado** (não ocorre nesta sequência) |

## Referências da base

- Y. Wang, P.-M. Jodoin, F. Porikli, J. Konrad, Y. Benezeth, P. Ishwar. *CDnet 2014: An Expanded Change Detection Benchmark Dataset*. IEEE CVPR Workshops, 2014.
- N. Goyette, P.-M. Jodoin, F. Porikli, J. Konrad, P. Ishwar. *changedetection.net: A New Change Detection Benchmark Dataset*. IEEE CVPR Workshops, 2012.
- Página oficial: <https://changedetection.vitalab.usherbrooke.ca/dataset2014/>

## Material consultado

- M. Piccardi. *Background subtraction techniques: a review*. IEEE International Conference on Systems, Man and Cybernetics, 2004, v. 4, p. 3099-3104. (média móvel e atualização seletiva do fundo)
- Documentação do OpenCV (<https://docs.opencv.org/>) e do NumPy.
