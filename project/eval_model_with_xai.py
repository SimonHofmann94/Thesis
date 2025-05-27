# evaluate_model_with_xai.py
import os
import json
import argparse
import glob
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Für Colormaps
from omegaconf import OmegaConf # Falls für get_transforms benötigt

# --- Projekt-spezifische Imports (Pfade anpassen!) ---
try:
    from project.models import ClassifierModel 
    from project.train import get_transforms 
    from project.utils import create_full_mask_from_png_object 
except ImportError:
    print("WARNUNG: Konnte Projektmodule nicht importieren. Stelle sicher, dass Pfade korrekt sind.")
    print("Definiere Fallbacks oder stelle die Module bereit.")
    # Fallback-Definitionen hier einfügen, wenn nötig (wie in inspect_image_predictions.py)
    # ... (ClassifierModel, get_transforms, create_full_mask_from_png_object) ...

class DummyExplainer:
    def __init__(self, model): 
        self.model = model
        print("INFO: DummyExplainer initialisiert.") # Für Debugging
    def attribute(self, inputs, target): 
        print(f"INFO: DummyExplainer.attribute aufgerufen für Target {target}. Gebe Zufalls-Attributionen zurück.") # Für Debugging
        return torch.rand_like(inputs) 

# --- XAI Bibliothek (z.B. Captum) ---
try:
    from captum.attr import LRP, LayerConductance, DeepLift 
    XAI_AVAILABLE = True
    print("INFO: Captum erfolgreich importiert.")
except ImportError:
    print("WARNUNG: Captum nicht gefunden. XAI-Funktionalität wird auf DummyExplainer zurückfallen.")
    XAI_AVAILABLE = False
    # Überschreibe die Captum-Klassen mit dem Dummy, wenn Captum nicht da ist
    LRP = DummyExplainer 
    LayerConductance = DummyExplainer
    DeepLift = DummyExplainer



# --- Globale Konfigurationen (könnten auch aus Hydra-Config kommen) ---
# Diese sollten mit dem Training des zu evaluierenden Modells übereinstimmen
STRIP_HEIGHT = 224 
STRIP_WIDTH_FOR_TRANSFORM = -1 # -1 für variable Breite / nur Höhe resizen in get_transforms
NUM_CLASSES = 5
MODEL_BACKBONE = 'densenet121' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["No Defect", "Defect 1", "Defect 2", "Defect 3", "Defect 4"]
DEFECT_CLASS_TO_IDX = {"defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4}

# Standard Augmentierungs-Config für Inferenz (wird an get_transforms übergeben)
DEFAULT_AUG_CFG_DICT = { "enable": False, "verbose_transforms": False } 


def load_model(model_path: str) -> nn.Module:
    # ... (Implementierung wie in inspect_image_predictions.py)
    model = ClassifierModel(backbone_name=MODEL_BACKBONE, pretrained=False, num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Modell erfolgreich von {model_path} geladen.")
    except Exception as e:
        print(f"FEHLER beim Laden des Modells von {model_path}: {e}")
        raise
    model.to(DEVICE)
    model.eval()
    return model

def get_xai_explainer(method_name: str, model: nn.Module) -> object:
    """Gibt ein XAI-Erklärungsobjekt zurück."""
    if not XAI_AVAILABLE and method_name != "dummy":
        print(f"WARNUNG: XAI-Methode '{method_name}' erfordert Captum, das nicht verfügbar ist. Verwende DummyExplainer.")
        return DummyExplainer(model)

    method_name_lower = method_name.lower()
    if method_name_lower == "lrp":
        # LRP muss oft wissen, welche Layer relevant sind, oder man verwendet LRP(model)
        # Für komplexe Modelle kann die Konfiguration von LRP spezifisch sein.
        # Ggf. muss man hier Layer auswählen, wenn Captums LRP nicht direkt mit allen Layern umgehen kann.
        # Z.B. model.features für den Feature-Extraktor.
        return LRP(model)
    elif method_name_lower == "deeplift":
        return DeepLift(model)
    # Hier weitere XAI-Methoden hinzufügen
    # elif method_name_lower == "integratedgradients":
    #    return IntegratedGradients(model)
    elif method_name_lower == "dummy":
        return DummyExplainer(model)
    else:
        raise ValueError(f"Unbekannte XAI-Methode: {method_name}. Verfügbar: lrp, deeplift, dummy.")


def aggregate_attributions(attributions: torch.Tensor, mode="sum_positive") -> np.ndarray:
    """Aggregiert mehrkanalige Attributionskarten zu einem einzelnen Kanal."""
    # Input attributions: (N, C, H, W) oder (C, H, W)
    # Output: (H, W)
    if attributions.ndim == 4: # (N, C, H, W) -> nehme erstes Sample
        attributions = attributions[0]
    
    attributions_np = attributions.cpu().detach().numpy() # (C, H, W)

    if mode == "sum_positive":
        attr_agg = np.sum(np.maximum(0, attributions_np), axis=0) # Summiere positive Beiträge über Kanäle
    elif mode == "sum_abs":
        attr_agg = np.sum(np.abs(attributions_np), axis=0)
    elif mode == "max_abs_channel": # Max abs Wert über Kanäle pro Pixel
        attr_agg = np.max(np.abs(attributions_np), axis=0)
    elif mode == "sum_all":
        attr_agg = np.sum(attributions_np, axis=0)
    else: # Fallback: erster Kanal (wenn Graustufen oder nur 1 Kanal relevant)
        attr_agg = attributions_np[0]
    return attr_agg

def process_single_image(
    img_path: str, 
    ann_path: Optional[str], 
    model: nn.Module, 
    xai_explainer: object, # Das Objekt vom get_xai_explainer
    transform: T.Compose, 
    output_dir: str,
    pred_threshold: float = 0.5, # Threshold für Klassenvorhersage, um XAI zu triggern
    xai_target_class_idx: Optional[int] = None # Wenn None, XAI für alle Klassen > Threshold
):
    print(f"\nVerarbeite Bild: {os.path.basename(img_path)}")
    try:
        image_pil_original = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  Fehler beim Laden des Bildes: {e}")
        return

    original_w, original_h = image_pil_original.size

    # Lade GT-Maske (optional)
    gt_mask_original_np = None
    if ann_path and os.path.exists(ann_path):
        gt_mask_original_np = load_full_gt_mask(ann_path, original_h, original_w) # Funktion von oben
    
    # --- Streifen-Extraktion und Inferenz (ähnlich wie in inspect_image_predictions.py) ---
    # Für XAI ist es oft besser, die Inferenz pro Streifen zu machen und dann XAI
    # auf die Streifen anzuwenden, für die eine interessante Klasse vorhergesagt wurde.
    
    all_strip_tensors = []
    all_strip_coords = []
    
    # Hier die Streifenextraktionslogik (Random Vertical Crop ist für Eval nicht ideal, besser feste Crops oder Sliding Window)
    # Für dieses Skript nehmen wir erstmal einen einfachen Sliding Window Ansatz für die Streifen.
    # Die `strip_height` kommt aus der globalen Konfiguration.
    # Stride für die Streifen kann hier auch konfiguriert werden.
    strip_stride_eval = STRIP_HEIGHT // 2 # Beispiel: 50% Überlappung für Evaluation

    for y_start in range(0, original_h - STRIP_HEIGHT + 1, strip_stride_eval):
        strip_pil = image_pil_original.crop((0, y_start, original_w, y_start + STRIP_HEIGHT))
        all_strip_tensors.append(transform(strip_pil))
        all_strip_coords.append((0, y_start)) # x ist immer 0, da volle Breite

    if not all_strip_tensors:
        print("  Keine Streifen extrahiert.")
        return

    batch_strip_tensors = torch.stack(all_strip_tensors).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(batch_strip_tensors)
        probabilities = torch.sigmoid(outputs) # (num_strips, num_classes)

    # --- XAI und Stitching der Attributionskarten ---
    # Erstelle eine leere Gesamt-Attributionskarte für das Bild
    # Wir speichern hier die aggregierte LRP-Karte für die "am wahrscheinlichsten" Defektklasse pro Pixel
    # oder eine Kombination. Für den Anfang: eine Karte pro Bild, die die maximale Relevanz über alle
    # als defekt klassifizierten Streifen und Klassen zeigt.
    
    stitched_xai_heatmap = np.zeros((original_h, original_w), dtype=float)
    xai_overlap_counter = np.zeros((original_h, original_w), dtype=int)

    for i, strip_tensor in enumerate(batch_strip_tensors):
        strip_probs = probabilities[i] # Wahrscheinlichkeiten für diesen Streifen
        x_coord, y_coord = all_strip_coords[i] # Obere linke Ecke des Streifens
        
        # Finde Klassen, die den Threshold überschreiten (nur Defektklassen 1-4)
        predicted_defect_classes_indices = [
            cls_idx for cls_idx, prob in enumerate(strip_probs) 
            if cls_idx > 0 and prob >= pred_threshold # cls_idx > 0 ignoriert "No Defect"
        ]

        if not predicted_defect_classes_indices:
            continue # Kein Defekt stark genug in diesem Streifen vorhergesagt

        # Wenn xai_target_class_idx gegeben ist, nur dafür XAI machen, wenn es vorhergesagt wurde
        targets_for_xai = []
        if xai_target_class_idx is not None:
            if xai_target_class_idx in predicted_defect_classes_indices:
                targets_for_xai.append(xai_target_class_idx)
        else: # Sonst für alle vorhergesagten Defektklassen
            targets_for_xai = predicted_defect_classes_indices
        
        # Wir brauchen den Input mit erforderten Gradienten für Captum
        strip_tensor_for_xai = strip_tensor.unsqueeze(0).clone().detach().requires_grad_(True)

        # Sammle Attributionskarten für diesen Streifen
        strip_combined_attribution_map = np.zeros((STRIP_HEIGHT, original_w), dtype=float)

        for target_idx in targets_for_xai:
            print(f"  ... Berechne XAI für Streifen bei y={y_coord}, Zielklasse: {CLASS_NAMES[target_idx]} (Prob: {strip_probs[target_idx]:.2f})")
            try:
                # Captum's LRP erwartet target als Index
                attribution = xai_explainer.attribute(strip_tensor_for_xai, target=target_idx)
                # Attribution hat Shape (1, Channels, H, W)
                agg_attribution_strip = aggregate_attributions(attribution, mode="sum_positive") # (H,W)
                strip_combined_attribution_map = np.maximum(strip_combined_attribution_map, agg_attribution_strip) # Nimm max. Relevanz
            except Exception as e_xai:
                print(f"    Fehler bei XAI-Berechnung für Klasse {target_idx}: {e_xai}")
                continue
        
        # Füge die (ggf. kombinierte) Attributionskarte des Streifens zur Gesamtkarte hinzu
        stitched_xai_heatmap[y_coord : y_coord + STRIP_HEIGHT, :] += strip_combined_attribution_map
        xai_overlap_counter[y_coord : y_coord + STRIP_HEIGHT, :] += 1

    xai_overlap_counter[xai_overlap_counter == 0] = 1 # Division durch Null vermeiden
    stitched_xai_heatmap /= xai_overlap_counter
    
    # Normalisiere die Heatmap für bessere Visualisierung (optional, aber oft gut)
    if np.max(stitched_xai_heatmap) > 0:
        stitched_xai_heatmap = (stitched_xai_heatmap - np.min(stitched_xai_heatmap)) / \
                               (np.max(stitched_xai_heatmap) - np.min(stitched_xai_heatmap) + 1e-8)


    # --- Visualisierung ---
    aspect_ratio = original_h / original_w if original_w > 0 else 0.5 
    fig_width_inches = 20  # Erhöhe die Gesamtbreite der Figur deutlich
    fig_height_inches = fig_width_inches / 3 * aspect_ratio # Höhe proportional zur Breite eines Subplots
    fig_height_inches = max(6, fig_height_inches) # Stelle eine Mindesthöhe sicher

    # Man könnte auch DPI erhöhen für schärfere Bilder, aber das erhöht auch die Dateigröße.
    # fig_dpi = 150 

    fig, axes = plt.subplots(1, 3, figsize=(fig_width_inches, fig_height_inches)) # , dpi=fig_dpi)
    
    fig.suptitle(f"Evaluation: {os.path.basename(img_path)} (XAI: {type(xai_explainer).__name__})", fontsize=16) # Größere Schrift für Titel

    # 1. Originalbild
    axes[0].imshow(image_pil_original)
    axes[0].set_title("Originalbild", fontsize=12) # Größere Schrift
    axes[0].axis('off')

    # 2. Originalbild mit GT-Maske
    axes[1].imshow(image_pil_original) 
    if gt_mask_original_np is not None:
        gt_overlay = np.zeros((*gt_mask_original_np.shape, 4), dtype=np.uint8)
        temp_mask_colors_rgba = [ 
            (255, 0, 0, 128),   # Defect 1 (Rot), etwas stärkeres Alpha
            (0, 255, 0, 128),   # Defect 2 (Grün)
            (0, 0, 255, 128),   # Defect 3 (Blau)
            (255, 255, 0, 128)  # Defect 4 (Gelb)
        ] 
        for i_cls_gt in range(1, 5): 
            if (i_cls_gt - 1) < len(temp_mask_colors_rgba):
                gt_overlay[gt_mask_original_np == i_cls_gt] = temp_mask_colors_rgba[i_cls_gt-1]
        axes[1].imshow(Image.fromarray(gt_overlay, 'RGBA'))
    axes[1].set_title("Original + GT-Maske", fontsize=12) # Größere Schrift
    axes[1].axis('off')

    # 3. XAI Heatmap
    axes[2].imshow(image_pil_original) 
    if np.any(stitched_xai_heatmap): 
        axes[2].imshow(stitched_xai_heatmap, cmap='hot', alpha=0.6) 
    else:
        axes[2].text(0.5, 0.5, 'Keine XAI-Relevanz\ngefunden/berechnet', 
                     horizontalalignment='center', verticalalignment='center', 
                     transform=axes[2].transAxes, color='gray', fontsize=10)

    axes[2].set_title(f"XAI Heatmap ({type(xai_explainer).__name__})", fontsize=12) # Größere Schrift
    axes[2].axis('off')

    # Anpassen des Layouts, um Überlappungen zu vermeiden und Platz für Titel zu schaffen
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # rect=[left, bottom, right, top]
    
    # Dateinamen für Output
    base_out_name = os.path.splitext(os.path.basename(img_path))[0]
    xai_method_name_for_file = type(xai_explainer).__name__.lower()
    if isinstance(xai_explainer, globals().get('DummyExplainer', object)):
        xai_method_name_for_file = "dummy_xai"
        
    output_filename = os.path.join(output_dir, f"{base_out_name}_eval_viz_{xai_method_name_for_file}.png")
    plt.savefig(output_filename) # Optional: dpi=fig_dpi beim Speichern hinzufügen
    print(f"  Vergleichsbild gespeichert: {output_filename}")
    plt.close(fig)


# (load_full_gt_mask Funktion von inspect_image_predictions.py hier einfügen)
def load_full_gt_mask(ann_path: str, img_height: int, img_width: int) -> Optional[np.ndarray]:
    if not os.path.exists(ann_path): return None
    gt_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    try:
        with open(ann_path, 'r') as f: annotation = json.load(f)
        if "objects" in annotation and annotation["objects"]:
            for obj in annotation["objects"]:
                class_title = obj.get("classTitle")
                if class_title in DEFECT_CLASS_TO_IDX:
                    class_idx_for_mask = DEFECT_CLASS_TO_IDX[class_title]
                    if "bitmap" in obj:
                        b64_data = obj["bitmap"].get("data")
                        origin = obj["bitmap"].get("origin")
                        if b64_data and origin:
                            single_obj_mask_np = create_full_mask_from_png_object(b64_data, origin, img_height, img_width)
                            if single_obj_mask_np is not None: gt_mask[single_obj_mask_np == 1] = class_idx_for_mask
        return gt_mask
    except Exception as e: print(f"Fehler beim Laden der Annotation {ann_path}: {e}"); return None


def main():
    parser = argparse.ArgumentParser(description="Evaluiert ein Modell mit XAI-Heatmaps auf einem Bildordner.")
    parser.add_argument("--img_dir", required=True, help="Pfad zum Ordner mit den Originalbildern.")
    parser.add_argument("--ann_dir", required=True, help="Pfad zum Ordner mit den Annotationen.")
    parser.add_argument("--model_path", required=True, help="Pfad zum trainierten Modell (.pt Datei).")
    parser.add_argument("--output_dir", required=True, help="Pfad zum Speicherort der Ergebnisbilder.")
    parser.add_argument("--xai_method", type=str, default="lrp", choices=["lrp", "deeplift", "dummy"], # Füge hier mehr hinzu
                        help="Zu verwendende XAI-Methode.")
    parser.add_argument("--pred_threshold", type=float, default=0.5, 
                        help="Schwellenwert für die Vorhersage einer Defektklasse, um XAI zu triggern.")
    parser.add_argument("--xai_target_class", type=int, default=None, choices=range(1,NUM_CLASSES),
                        help="Optional: Spezifischer Defektklassen-Index (1-4), für den XAI durchgeführt wird. "
                             "Wenn nicht gesetzt, für alle vorhergesagten Klassen > Threshold.")
    parser.add_argument("--limit_images", type=int, default=None, help="Optional: Limitiere die Anzahl der zu verarbeitenden Bilder.")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Lade Modell
    model = load_model(args.model_path)
    
    # Hole XAI Explainer
    xai_explainer = get_xai_explainer(args.xai_method, model)

    # Hole Transformationen (für Inferenz)
    # Erstelle ein minimales cfg-Objekt für get_transforms, wenn es DictConfig erwartet
    # cfg_for_transforms = OmegaConf.create({"augment": DEFAULT_AUG_CFG_DICT}) # Wenn get_transforms DictConfig braucht
    # Oder direkt das dict übergeben, wenn get_transforms angepasst wurde:
    inference_transforms = get_transforms(DEFAULT_AUG_CFG_DICT, STRIP_HEIGHT, STRIP_WIDTH_FOR_TRANSFORM, is_train=False)
    
    image_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(args.img_dir, "*.jpeg"))) + \
                  sorted(glob.glob(os.path.join(args.img_dir, "*.png")))

    print(f"[DEBUG] Gefundene Bildpfade (erste 5): {image_paths[:5]}") # DEBUG
    print(f"[DEBUG] Anzahl gefundener Bildpfade: {len(image_paths)}") # DEBUG

    if args.limit_images is not None:
        image_paths = image_paths[:args.limit_images]
        print(f"Limitiere Verarbeitung auf die ersten {args.limit_images} Bilder.")

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Versuche, passende Annotationsdatei zu finden
        ann_path = os.path.join(args.ann_dir, base_name + ".json")
        if not os.path.exists(ann_path): # Fallback für z.B. originalname.jpg.json
             ann_path = os.path.join(args.ann_dir, os.path.basename(img_path) + ".json")
             if not os.path.exists(ann_path):
                  print(f"Keine Annotation für {os.path.basename(img_path)} gefunden. Überspringe GT-Anzeige.")
                  ann_path = None
        
        process_single_image(
            img_path, ann_path, model, xai_explainer, inference_transforms, 
            args.output_dir, args.pred_threshold, args.xai_target_class
        )

    print("\nEvaluation abgeschlossen.")

if __name__ == "__main__":
    # Stelle sicher, dass die Fallback-Definitionen für ClassifierModel etc. oben sind,
    # falls die Imports fehlschlagen.
    main()