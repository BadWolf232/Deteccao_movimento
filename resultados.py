import csv
import cv2
import numpy as np


# ============================ AUXILIARES ============================

def write_csv(path, rows):
    # grava uma lista de dicionários em CSV
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def round_row(row):
    # arredonda os floats para ficar legível no CSV
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


def draw_boxes(frame, boxes, text):
    # desenha as caixas delimitadoras e um texto no topo do frame
    vis = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis


# ============================ TABELAS ============================

def save_frame_metrics(out_dir, final):
    # métricas de cada frame avaliado, para os dois métodos
    rows = []
    for method, (records, _) in final.items():
        for r in records:
            if r["tp"] is not None:
                rows.append(round_row({
                    "metodo": method, "frame": r["frame"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                    "precisao": r["precisao"], "recall": r["recall"], "f1": r["f1"], "iou": r["iou"],
                    "ms": r["ms"]}))
    write_csv(out_dir / "metricas_por_frame.csv", rows)


def save_time_series_csv(out_dir, final, eval_start, eval_end):
    # série temporal: nº de regiões e área em movimento por frame
    for method, (records, _) in final.items():
        write_csv(out_dir / f"serie_temporal_{method}.csv", [
            {"frame": r["frame"], "regioes": r["regioes"], "area_px": r["area"],
             "avaliado": eval_start <= r["frame"] <= eval_end} for r in records])


def save_label_distribution(out_dir, gt_dir, eval_start, eval_end):
    # distribuição dos rótulos da referência no intervalo avaliado (usada no relatório)
    counts = {}
    for num in range(eval_start, eval_end + 1):
        gt = cv2.imread(str(gt_dir / f"gt{num:06d}.png"), cv2.IMREAD_GRAYSCALE)
        for value, amount in zip(*np.unique(gt, return_counts=True)):
            counts[int(value)] = counts.get(int(value), 0) + int(amount)
    total = sum(counts.values())
    names = {0: "fundo", 255: "movimento", 170: "desconhecido", 85: "sombra", 50: "fora da ROI"}
    write_csv(out_dir / "rotulos_referencia.csv", [
        {"valor": v, "significado": names.get(v, "outro"), "pixels": c, "percentual": round(100 * c / total, 3)}
        for v, c in sorted(counts.items())])


# ============================ IMAGENS ============================

def error_panel(mask, gt, roi_mask):
    # mapa de erros: verde = acerto, vermelho = falso positivo, azul = falso negativo, cinza = ignorado
    pred = mask > 0
    panel = np.zeros((*mask.shape, 3), np.uint8)
    panel[(gt == 170) | (gt == 85) | (gt == 50)] = (128, 128, 128)
    panel[pred & (gt == 255)] = (0, 200, 0)
    panel[pred & (gt == 0) & (roi_mask > 0)] = (0, 0, 255)
    panel[~pred & (gt == 255)] = (255, 0, 0)
    return panel


def save_failure_cases(name, records, masks, frame_paths, gt_dir, roi_mask, out_dir, first, last):
    # salva os 2 melhores e os 3 piores frames (por F1) do conjunto de avaliação final
    candidates = [r for r in records
                  if r["tp"] is not None and first <= r["frame"] <= last and r["tp"] + r["fn"] >= 500]
    candidates.sort(key=lambda r: r["f1"])
    chosen = [("pior", r) for r in candidates[:3]] + [("melhor", r) for r in candidates[-2:]]

    shape = (576, 720)
    for kind, r in chosen:
        num = r["frame"]
        mask = np.unpackbits(masks[num])[: shape[0] * shape[1]].reshape(shape) * 255
        gt = cv2.imread(str(gt_dir / f"gt{num:06d}.png"), cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(str(frame_paths[num - 1]))
        left = draw_boxes(frame, r["boxes"], f"{name} frame {num} F1={r['f1']:.2f}")
        right = error_panel(mask, gt, roi_mask)
        cv2.putText(right, "verde=acerto vermelho=FP azul=FN cinza=ignorado", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        image = cv2.resize(np.hstack([left, right]), None, fx=0.7, fy=0.7)
        cv2.imwrite(str(out_dir / f"{name}_{kind}_frame{num:04d}.png"), image)


def plot_time_series(series, path, eval_start, test_start):
    # gráfico simples (OpenCV): nº de regiões e área em movimento ao longo do tempo
    width, height, margin = 1200, 700, 60
    canvas = np.full((height, width, 3), 255, np.uint8)
    colors = {"diff": (0, 0, 220), "fundo": (200, 100, 0)}
    names = {"diff": "diferenca de frames", "fundo": "modelo de fundo"}
    n_frames = 1200
    panels = [("regioes", "Numero de regioes moveis", 0), ("area", "Area em movimento (px)", 1)]

    for key, title, idx in panels:
        top = margin + idx * (height - margin) // 2
        bottom = top + (height - margin) // 2 - margin
        peak = max(max(r[key] for r in recs) for recs in series.values()) or 1

        def x_of(f):
            return int(margin + (f / n_frames) * (width - 2 * margin))

        # faixas: ajuste (cinza claro) e teste final (cinza médio)
        cv2.rectangle(canvas, (x_of(eval_start), top), (x_of(test_start), bottom), (240, 240, 240), -1)
        cv2.rectangle(canvas, (x_of(test_start), top), (x_of(n_frames), bottom), (215, 215, 215), -1)
        cv2.rectangle(canvas, (margin, top), (width - margin, bottom), (0, 0, 0), 1)
        cv2.putText(canvas, f"{title} (max {peak})", (margin, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        for method, recs in series.items():
            pts = np.array([[x_of(r["frame"]), int(bottom - (r[key] / peak) * (bottom - top))]
                            for r in recs], np.int32)
            cv2.polylines(canvas, [pts], False, colors[method], 1)

    for i, method in enumerate(series):
        cv2.putText(canvas, names[method], (margin + i * 260, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[method], 2)
    cv2.putText(canvas, "frames: 1 a 1200 | cinza claro = ajuste (300-599) | cinza escuro = teste (600-1200)",
                (margin + 560, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)
