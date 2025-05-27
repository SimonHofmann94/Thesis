import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import hydra
from omegaconf import DictConfig
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

from project.postprocessing import (
    binarize_threshold, binarize_gmm, apply_morphology
)


def ensure_gt_masks(root: str, cfg: DictConfig):
    """
    Prüft, ob GT-Masken als PNGs existieren. Wenn nicht, erstellt sie aus LabelMe-JSONs (nur für BSData).
    Unterstützt sowohl LabelMe 'shapes' als auch ältere 'polygons'.
    Zeichnet Polygone aus Original- oder generischer JSON-Struktur,
    nutzt JSON-Felder 'imageWidth'/'imageHeight' oder fällt zurück auf cfg.data.img_size.
    """
    dataset_cfg = cfg.data.bsdata
    gt_json_dir = os.path.join(root, dataset_cfg.label_dir)
    gt_png_dir = os.path.join(root, cfg.eval.gt_mask_dir)
    os.makedirs(gt_png_dir, exist_ok=True)

    for jpath in glob.glob(os.path.join(gt_json_dir, "*.json")):
        base = os.path.splitext(os.path.basename(jpath))[0]
        out_path = os.path.join(gt_png_dir, f"{base}.png")
        if os.path.exists(out_path):
            continue
        with open(jpath, 'r') as f:
            data = json.load(f)
        # Bestimme Bildgröße
        img_w = data.get('imageWidth', cfg.data.img_size)
        img_h = data.get('imageHeight', cfg.data.img_size)
                # Erzeuge Maske mit weißen Pixeln (0 oder 255)
        mask_orig = Image.new('L', (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask_orig)
        # Zeichne alle Polygone in Weiß (255)
        if 'shapes' in data:
            for shape in data['shapes']:
                pts = shape.get('points', [])
                poly = [(int(x), int(y)) for x, y in pts]
                if poly:
                    draw.polygon(poly, outline=255, fill=255)
        elif 'polygons' in data:
            for poly_pts in data['polygons']:
                poly = [(int(x), int(y)) for x, y in poly_pts]
                if poly:
                    draw.polygon(poly, outline=255, fill=255)
        else:
            continue
        # Auf Zielgröße bringen (Nearest für harte Kanten)
        size = cfg.data.img_size

        # Neu:
        # mask_resized = mask_orig.resize((size, size), resample=Image.NEAREST)
        # # Alle Pixel > 0 auf 255 setzen, damit die Maske weiß ist:
        # mask_resized = mask_resized.point(lambda x: 255 if x > 0 else 0)
        # mask_resized.save(out_path)

        # Oder
        mask_resized = mask_orig.resize((size, size), resample=Image.NEAREST)
        mask_resized = Image.eval(mask_resized, lambda x: x * 255)
        mask_resized.save(out_path)


        # Debug: nachzeichnen
        arr_res = np.array(mask_resized)
        print("  After save unique values:", np.unique(arr_res), "non-zero count:", np.count_nonzero(arr_res))
        print("  Saved GT mask to", out_path)

    # Ende ensure_gt_masks

def overlay_masks(root: str, base: str, cfg: DictConfig, gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Erzeugt eine farbige Overlay-Visualisierung (GT in Rot, Prediction in Blau) und speichert sie.
    """
    img_dir = os.path.join(root, cfg.data.bsdata.data_dir)
    out_dir = os.path.join(root, cfg.output.mask_comparison_dir,
                           cfg.model.backbone, cfg.xai.method)
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.join(img_dir, f"{base}.png")
    img = Image.open(img_path).convert("RGB")
    overlay = img.convert("RGBA")

    # Rote Fläche für GT
    gt_overlay = Image.new('RGBA', img.size, (255, 0, 0, 100))
    gt_mask_img = Image.fromarray((gt_mask * 255).astype('uint8'), mode='L')
    overlay.paste(gt_overlay, (0, 0), gt_mask_img)

    # Blaue Fläche für Prediction
    pred_overlay = Image.new('RGBA', img.size, (0, 0, 255, 100))
    pred_mask_img = Image.fromarray((pred_mask * 255).astype('uint8'), mode='L')
    overlay.paste(pred_overlay, (0, 0), pred_mask_img)

    combo = Image.alpha_composite(img.convert('RGBA'), overlay)
    combo.save(os.path.join(out_dir, f"comparison_{base}.png"))


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    root = hydra.utils.get_original_cwd()

    # 1) GT-Masken erzeugen, falls noch nicht vorhanden
    ensure_gt_masks(root, cfg)

    # 2) Pfade zu GT-Masken sammeln
    gt_dir = os.path.join(root, cfg.eval.gt_mask_dir)
    gt_paths = {os.path.splitext(f)[0]: os.path.join(gt_dir, f)
                for f in os.listdir(gt_dir) if f.endswith('.png')}

    # 3) Postprocessing-Varianten aus Config
    pp_methods = []
    for val in cfg.eval.postproc.threshold_fixed_values:
        pp_methods.append({
            'suffix': f"threshold_fixed_{val}",
            'method': 'threshold',
            'param': {'method': 'fixed', 'value': val}
        })
    for p in cfg.eval.postproc.threshold_percentiles:
        pp_methods.append({
            'suffix': f"threshold_percentile_{p}",
            'method': 'threshold',
            'param': {'method': 'percentile', 'percentile': p}
        })
    pp_methods.append({
        'suffix': f"gmm_{cfg.eval.postproc.gmm.n_components}_{cfg.eval.postproc.gmm.covariance_type}",
        'method': 'gmm',
        'param': {
            'n_components': cfg.eval.postproc.gmm.n_components,
            'covariance_type': cfg.eval.postproc.gmm.covariance_type
        }
    })

    records = []

    # 4) Schleife über Varianten und Bilder
    for cfg_pp in pp_methods:
        suffix = cfg_pp['suffix']
        mask_dir = os.path.join(
            root,
            cfg.output.mask_dir,
            cfg.model.backbone,
            cfg.xai.method,
            suffix
        )
        if not os.path.isdir(mask_dir):
            continue

        for base, gt_rel in gt_paths.items():
            # Nur positive klassifizierte Bilder evaluieren?
            gt_mask = np.array(Image.open(gt_rel).convert('L')) > 127
            pred_path = os.path.join(mask_dir, f"{base}.png")
            if not os.path.exists(pred_path):
                continue
            pred_mask = np.array(Image.open(pred_path).convert('L')) > 127

            # Overlay-Bild speichern
            overlay_masks(root, base, cfg, gt_mask, pred_mask)

            # Flatten für Metriken
            gt_flat = gt_mask.flatten().astype(int)
            pr_flat = pred_mask.flatten().astype(int)

            # Metriken berechnen
            iou = jaccard_score(gt_flat, pr_flat)
            dice = f1_score(gt_flat, pr_flat)
            prec = precision_score(gt_flat, pr_flat, zero_division=0)
            rec = recall_score(gt_flat, pr_flat, zero_division=0)

            records.append({
                'variant': suffix,
                'image': base,
                'iou': iou,
                'dice': dice,
                'precision': prec,
                'recall': rec
            })

    # 5) Abbruch bei Leerdaten
    if not records:
        print("⚠️  Keine Bilder ausgewertet. Prüfe GT-Masken und Predictions.")
        return

    # 6) DataFrame & Summary
    df = pd.DataFrame.from_records(records)
    summary = df.groupby('variant').agg({
        'iou': ['mean', 'std'],
        'dice': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    print(summary)

    # 7) Speichern
    out_csv = os.path.join(
        root,
        cfg.output.mask_dir,
        f"seg_eval_{cfg.model.backbone}_{cfg.xai.method}.csv"
    )
    df.to_csv(out_csv, index=False)
    summary.to_csv(out_csv.replace('.csv', '_summary.csv'))
    print(f"Results saved to {out_csv}")


if __name__ == '__main__':
    main()
