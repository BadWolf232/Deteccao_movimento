from pathlib import Path
import sys
import time
import cv2
import numpy as np

from resultados import (
    draw_boxes, write_csv, round_row, save_failure_cases, plot_time_series,
    save_time_series_csv, save_frame_metrics, save_label_distribution,
)


#  LEITURA E PRÉ-PROCESSAMENTO 

def load_sequence(data_dir):
    # lista os frames em ordem temporal e carrega intervalo de avaliação e ROI
    frame_paths = sorted((data_dir / "input").glob("in*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"Nenhum frame em {data_dir / 'input'}")

    start, end = map(int, (data_dir / "temporalROI.txt").read_text().split())

    roi = cv2.imread(str(data_dir / "ROI.bmp"), cv2.IMREAD_GRAYSCALE)
    _, roi_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY)

    return frame_paths, start, end, roi_mask


def preprocess(frame):
    # cinza + borrão para reduzir ruído antes da comparação
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


#  MÉTODO 1: DIFERENÇA DE FRAMES 

def frame_diff_mask(previous_gray, current_gray):
    # máscara bruta: pixels que mudaram mais de 20 níveis entre frames consecutivos
    diff = cv2.absdiff(previous_gray, current_gray)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    return mask


#  MÉTODO 2: MODELO DE FUNDO 

def background_mask(background, gray, thresh):
    # máscara bruta: pixels que diferem do modelo de fundo mais que 'thresh' níveis
    diff = np.abs(gray.astype(np.float32) - background)
    return (diff > thresh).astype(np.uint8) * 255


def update_background(background, gray, mask, alpha, selective):
    # média móvel exponencial: B = B + taxa * (frame - B)
    # na versão seletiva, pixels em movimento são atualizados 10x mais devagar
    rate = np.full(gray.shape, alpha, np.float32)
    if selective:
        rate[mask > 0] = alpha / 10
    background += rate * (gray.astype(np.float32) - background)


#  PÓS-PROCESSAMENTO (COMUM) 

def postprocess(mask, roi_mask):
    # aplica ROI, remove ruído (abertura), fecha buracos (fechamento) e filtra regiões pequenas
    mask = cv2.bitwise_and(mask, roi_mask)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # clean = máscara final (só regiões com área >= 80 px); boxes = caixas delimitadoras
    clean = np.zeros_like(mask)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue
        boxes.append(cv2.boundingRect(contour))
        cv2.drawContours(clean, [contour], -1, 255, cv2.FILLED)

    return clean, boxes


#  AVALIAÇÃO 

def evaluate(mask, gt, roi_mask):
    # 255 = movimento, 0 = fundo; 170 (desconhecido), 85 (sombra) e 50 (fora da ROI) são ignorados
    pred = mask > 0
    moving = gt == 255
    background = (gt == 0) & (roi_mask > 0)
    tp = int(np.count_nonzero(pred & moving))
    fp = int(np.count_nonzero(pred & background))
    fn = int(np.count_nonzero(~pred & moving))
    return tp, fp, fn


def metrics(tp, fp, fn):
    # precisão, recall, F1 e IoU a partir das contagens de pixels
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return precision, recall, f1, iou


def aggregate(records, first, last):
    # soma TP/FP/FN dos frames avaliados em [first, last] e calcula as métricas agregadas
    sel = [r for r in records if r["tp"] is not None and first <= r["frame"] <= last]
    tp = sum(r["tp"] for r in sel)
    fp = sum(r["fp"] for r in sel)
    fn = sum(r["fn"] for r in sel)
    precision, recall, f1, iou = metrics(tp, fp, fn)
    return {
        "frames": len(sel), "precisao": precision, "recall": recall, "f1": f1, "iou": iou,
        "ms_por_frame": sum(r["ms"] for r in sel) / len(sel),
    }


#  PROCESSAMENTO DA SEQUÊNCIA 

def process_sequence(frame_paths, method, params, gt_dir, roi_mask, eval_range,
                     writer=None, keep_masks=False, show=False):
    # percorre os frames em ordem; method = "diff" ou "fundo"
    # params (só fundo): alpha, thresh, selective. eval_range: frames comparados com a referência
    records = []
    masks = {}
    previous = None
    background = None

    for path in frame_paths:
        frame_num = int(path.stem[2:])
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Falha ao ler o frame: {path}")

        t0 = time.perf_counter()
        gray = preprocess(frame)

        # o primeiro frame só inicializa o método (não há frame anterior / fundo ainda)
        if method == "diff" and previous is None:
            previous = gray
            if writer is not None:
                writer.write(frame)
            continue
        if method == "fundo" and background is None:
            background = gray.astype(np.float32)
            if writer is not None:
                writer.write(frame)
            continue

        if method == "diff":
            raw = frame_diff_mask(previous, gray)
            previous = gray
            mask, boxes = postprocess(raw, roi_mask)
        else:
            raw = background_mask(background, gray, params["thresh"])
            mask, boxes = postprocess(raw, roi_mask)
            update_background(background, gray, mask, params["alpha"], params["selective"])
        ms = (time.perf_counter() - t0) * 1000

        record = {
            "frame": frame_num, "regioes": len(boxes), "area": int(np.count_nonzero(mask)),
            "ms": ms, "tp": None, "fp": None, "fn": None,
            "precisao": None, "recall": None, "f1": None, "iou": None, "boxes": boxes,
        }
        if eval_range[0] <= frame_num <= eval_range[1]:
            gt = cv2.imread(str(gt_dir / f"gt{frame_num:06d}.png"), cv2.IMREAD_GRAYSCALE)
            record["tp"], record["fp"], record["fn"] = evaluate(mask, gt, roi_mask)
            record["precisao"], record["recall"], record["f1"], record["iou"] = metrics(
                record["tp"], record["fp"], record["fn"])
            if keep_masks:
                masks[frame_num] = np.packbits(mask > 0)
        records.append(record)

        if writer is not None or show:
            tag = "AVALIACAO" if eval_range[0] <= frame_num <= eval_range[1] else "aquecimento"
            vis = draw_boxes(frame, boxes, f"frame {frame_num} [{tag}] regioes: {len(boxes)}")
            if writer is not None:
                writer.write(vis)
            if show:
                cv2.imshow(f"Frame ({method})", vis)
                cv2.imshow(f"Mascara ({method})", mask)
                if cv2.waitKey(30) & 0xFF == ord("e"):  # tecla 'e' fecha as janelas e segue sem exibir
                    show = False
                    cv2.destroyAllWindows()

    if show:
        cv2.destroyAllWindows()
    return records, masks


#  PIPELINE COMPLETO 

def main():
    data_dir = Path("assets/PETS2006")
    out_dir = Path("saidas")
    (out_dir / "analise").mkdir(parents=True, exist_ok=True)
    frame_paths, eval_start, eval_end, roi_mask = load_sequence(data_dir)
    gt_dir = data_dir / "groundtruth"

    # divisão dos dados: 1-299 aquecimento | 300-599 ajuste | 600-1200 avaliação final
    tune_range = (eval_start, 599)
    test_range = (600, eval_end)

    # ---------- 1) ajuste de parâmetros (só frames 300-599) ----------
    print("Ajuste de parâmetros nos frames 300-599...")
    tune_rows = []
    records, _ = process_sequence(frame_paths[:tune_range[1]], "diff", {}, gt_dir, roi_mask, tune_range)
    tune_rows.append(round_row({"metodo": "diff", "alpha": "", "limiar": 20, "seletivo": "",
                                **aggregate(records, *tune_range)}))
    print("  diff (parametros fixos): F1 =", tune_rows[-1]["f1"])

    best = None
    for alpha in (0.01, 0.05):
        for thresh in (20, 30):
            for selective in (False, True):
                params = {"alpha": alpha, "thresh": thresh, "selective": selective}
                records, _ = process_sequence(frame_paths[:tune_range[1]], "fundo", params,
                                              gt_dir, roi_mask, tune_range)
                result = aggregate(records, *tune_range)
                tune_rows.append(round_row({"metodo": "fundo", "alpha": alpha, "limiar": thresh,
                                            "seletivo": selective, **result}))
                print(f"  fundo alpha={alpha} limiar={thresh} seletivo={selective}: F1={result['f1']:.4f}")
                if best is None or result["f1"] > best[0]:
                    best = (result["f1"], params)
    write_csv(out_dir / "ajuste.csv", tune_rows)
    best_params = best[1]
    print("Melhor configuração do fundo (pelo F1 no ajuste):", best_params)

    # ---------- 2) execução final: frames 1-1200, vídeo anotado, métricas ----------
    final = {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for method, params in (("diff", {}), ("fundo", best_params)):
        print(f"Execução final: {method}...")
        writer = cv2.VideoWriter(str(out_dir / f"video_{method}.mp4"), fourcc, 25, (720, 576))
        records, masks = process_sequence(frame_paths, method, params, gt_dir, roi_mask,
                                          (eval_start, eval_end), writer=writer, keep_masks=True,
                                          show=True)  # exibe o vídeo anotado enquanto processa
        writer.release()
        final[method] = (records, masks)
        save_failure_cases(method, records, masks, frame_paths, gt_dir, roi_mask,
                           out_dir / "analise", *test_range)

    # ---------- 3) tabelas e gráficos (código em resultados.py) ----------
    summary_rows = []
    for method, (records, _) in final.items():
        for label, rng in (("ajuste 300-599", tune_range), ("teste final 600-1200", test_range),
                           ("oficial 300-1200", (eval_start, eval_end))):
            summary_rows.append(round_row({"metodo": method, "conjunto": label, **aggregate(records, *rng)}))
    write_csv(out_dir / "metricas_finais.csv", summary_rows)
    save_frame_metrics(out_dir, final)
    save_time_series_csv(out_dir, final, eval_start, eval_end)
    plot_time_series({m: recs for m, (recs, _) in final.items()}, out_dir / "serie_temporal.png",
                     eval_start, test_range[0])
    save_label_distribution(out_dir, gt_dir, eval_start, eval_end)

    print("\nResultados finais:")
    for row in summary_rows:
        print(f"  {row['metodo']:5s} {row['conjunto']:22s} P={row['precisao']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f} IoU={row['iou']:.3f} {row['ms_por_frame']:.1f} ms/frame")
    print("Saídas em:", out_dir.resolve())


def visualizar():
    # modo visual: só a diferença de frames, sem ajuste nem gravação de saídas
    frame_paths, eval_start, eval_end, roi_mask = load_sequence(Path("assets/PETS2006"))
    process_sequence(frame_paths, "diff", {}, Path("assets/PETS2006/groundtruth"), roi_mask,
                     (eval_start, eval_end), show=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ver":
        visualizar()
    else:
        main()
