"""Análise de falhas citada no relatório (seção de resultados).

Rodar depois de `python ptz.py`. Lê saidas/ajuste.csv e saidas/metricas_por_frame.csv,
roda de novo os dois métodos (as máscaras são determinísticas) e grava, em saidas/:
  - metricas_por_janela.csv: precisão, recall, F1 e IoU por trechos de 100 frames do teste final
  - fp_fundo_por_janela.csv: falsos positivos do modelo de fundo por trecho
  - analise/falha_*.png e analise/fantasma_*.png: frames citados no relatório
Não altera nenhuma outra saída do ptz.py.
"""
import csv
from pathlib import Path
import cv2
import numpy as np

from ptz import load_sequence, process_sequence, metrics
from resultados import write_csv, round_row, draw_boxes, error_panel

DATA_DIR = Path("assets/PETS2006")
OUT_DIR = Path("saidas")
SHAPE = (576, 720)
TEST_START = 600          # início da avaliação final (mesma divisão do ptz.py)
WINDOW = 100              # tamanho do trecho, em frames
FRAMES_PAINEL = [("diff", 786), ("fundo", 750), ("fundo", 950), ("fundo", 1195)]


def best_background_params():
    # mesma escolha do ptz.py: configuração do fundo de maior F1 no ajuste (ajuste.csv)
    rows = [r for r in csv.DictReader(open(OUT_DIR / "ajuste.csv", encoding="utf-8")) if r["metodo"] == "fundo"]
    best = max(rows, key=lambda r: float(r["f1"]))
    return {"alpha": float(best["alpha"]), "thresh": int(best["limiar"]), "selective": best["seletivo"] == "True"}


def window_metrics(eval_end):
    # soma TP/FP/FN por trecho de 100 frames (600-699, 700-799, ...) lendo metricas_por_frame.csv
    frames = list(csv.DictReader(open(OUT_DIR / "metricas_por_frame.csv", encoding="utf-8")))
    rows = []
    for method in ("diff", "fundo"):
        for start in range(TEST_START, eval_end + 1, WINDOW):
            end = min(start + WINDOW - 1, eval_end)
            sel = [r for r in frames if r["metodo"] == method and start <= int(r["frame"]) <= end]
            tp, fp, fn = (sum(int(r[k]) for r in sel) for k in ("tp", "fp", "fn"))
            precision, recall, f1, iou = metrics(tp, fp, fn)
            rows.append(round_row({"metodo": method, "janela": f"{start}-{end}", "frames": len(sel),
                                   "tp": tp, "fp": fp, "fn": fn, "precisao": precision,
                                   "recall": recall, "f1": f1, "iou": iou}))
    write_csv(OUT_DIR / "metricas_por_janela.csv", rows)


def unpack(mask_bits):
    return np.unpackbits(mask_bits)[: SHAPE[0] * SHAPE[1]].reshape(SHAPE) > 0


def background_false_positives(masks, gt_dir, roi_mask, frame_paths, eval_end):
    # por trecho: falsos positivos por frame, quantos ficam a <=10/25 px do movimento da referência
    # (255 ou 170) e quantos pixels são falso positivo em >=50% dos frames (local fixo = objeto residual)
    k10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    rows = []
    for start in range(TEST_START, eval_end + 1, WINDOW):
        end = min(start + WINDOW - 1, eval_end)
        total = near10 = near25 = 0
        acc = np.zeros(SHAPE, np.float32)
        count = 0
        for num in range(start, end + 1):
            gt = cv2.imread(str(gt_dir / f"gt{num:06d}.png"), cv2.IMREAD_GRAYSCALE)
            fp = unpack(masks[num]) & (gt == 0) & (roi_mask > 0)
            near_src = ((gt == 255) | (gt == 170)).astype(np.uint8)
            total += int(fp.sum())
            near10 += int((fp & (cv2.dilate(near_src, k10) > 0)).sum())
            near25 += int((fp & (cv2.dilate(near_src, k25) > 0)).sum())
            acc += fp
            count += 1
        fixed = (acc / count) >= 0.5
        rows.append(round_row({
            "janela": f"{start}-{end}", "frames": count, "fp_por_frame": total / count,
            "pct_ate_10px_do_movimento": 100 * near10 / total if total else 0.0,
            "pct_ate_25px_do_movimento": 100 * near25 / total if total else 0.0,
            "pixels_local_fixo": int(fixed.sum())}))
        if start == 1100:  # trecho com objeto residual: salva o mapa sobre um frame do trecho
            mid = (start + end) // 2
            vis = cv2.imread(str(frame_paths[mid - 1]))
            vis[fixed] = (0, 0, 255)
            cv2.putText(vis, f"frame {mid}: vermelho = falso positivo fixo ({start}-{end})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(str(OUT_DIR / "analise" / f"fantasma_{start}_{end}.png"), vis)
    write_csv(OUT_DIR / "fp_fundo_por_janela.csv", rows)


def save_panel(method, num, records, masks, frame_paths, gt_dir, roi_mask):
    # frame com caixas + mapa de erros (verde = acerto, vermelho = FP, azul = FN, cinza = ignorado)
    boxes = next(r["boxes"] for r in records if r["frame"] == num)
    gt = cv2.imread(str(gt_dir / f"gt{num:06d}.png"), cv2.IMREAD_GRAYSCALE)
    left = draw_boxes(cv2.imread(str(frame_paths[num - 1])), boxes, f"{method} frame {num}")
    right = error_panel(unpack(masks[num]).astype(np.uint8) * 255, gt, roi_mask)
    image = cv2.resize(np.hstack([left, right]), None, fx=0.7, fy=0.7)
    cv2.imwrite(str(OUT_DIR / "analise" / f"falha_{method}_frame{num:04d}.png"), image)


def main():
    (OUT_DIR / "analise").mkdir(parents=True, exist_ok=True)
    frame_paths, eval_start, eval_end, roi_mask = load_sequence(DATA_DIR)
    gt_dir = DATA_DIR / "groundtruth"

    window_metrics(eval_end)
    print("saidas/metricas_por_janela.csv gravado")

    params = {"diff": {}, "fundo": best_background_params()}
    print("Configuração do fundo usada (a mesma do ptz.py):", params["fundo"])
    results = {}
    for method, p in params.items():
        print(f"Processando {method}...")
        results[method] = process_sequence(frame_paths, method, p, gt_dir, roi_mask,
                                           (eval_start, eval_end), keep_masks=True)

    background_false_positives(results["fundo"][1], gt_dir, roi_mask, frame_paths, eval_end)
    print("saidas/fp_fundo_por_janela.csv gravado")

    for method, num in FRAMES_PAINEL:
        records, masks = results[method]
        save_panel(method, num, records, masks, frame_paths, gt_dir, roi_mask)
    print("Imagens em saidas/analise/ (falha_*.png e fantasma_*.png)")


if __name__ == "__main__":
    main()
