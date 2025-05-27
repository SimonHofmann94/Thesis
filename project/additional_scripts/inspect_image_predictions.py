# inspect_image_predictions.py
import os
import json
import argparse
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T # Alias T für torchvision.transforms
from PIL import Image, ImageDraw, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf # Für die cfg-Struktur in get_transforms

# --- Annahme: Die folgenden Definitionen/Importe sind verfügbar ---
# Wenn models.py und utils.py im PYTHONPATH sind oder im selben Verzeichnis:
try:
    from project.models import ClassifierModel # Passe den Importpfad an deine Struktur an
    # Wenn get_transforms in train.py ist und train.py im Projekt-Root liegt
    # und dieses Skript z.B. auch im Projekt-Root oder einem Unterordner liegt:
    from project.train import get_transforms  # Passe den Importpfad an
    from project.utils import create_full_mask_from_png_object # Passe an
except ImportError as e:
    print(f"Fehler beim Importieren von Projektmodulen: {e}")
    print("Stelle sicher, dass die Pfade korrekt sind und __init__.py Dateien existieren.")
    print("Für diesen Entwurf werden ClassifierModel und get_transforms direkt unten definiert.")
    # Fallback: Definitionen hier einfügen, wenn Importe nicht direkt klappen
    # (Hier ClassifierModel und get_transforms von dir einfügen)

    # --- ClassifierModel Definition (von dir bereitgestellt) ---
    import torchvision.models as tv_models # tv_models statt models
    from torchvision.models import (
        VGG11_Weights, ResNet18_Weights, EfficientNet_B0_Weights, DenseNet121_Weights
    )
    class ClassifierModel(nn.Module):
        def __init__(self, backbone_name: str = 'densenet121', pretrained: bool = True, num_classes: int = 5):
            super().__init__()
            if backbone_name == 'vgg11': # ... (Rest deiner Modelldefinition) ...
                weights = VGG11_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = tv_models.vgg11(weights=weights)
                self.features = backbone.features
                self.avgpool = backbone.avgpool
                in_features = backbone.classifier[-1].in_features
                classifier_layers = list(backbone.classifier.children())[:-1]
                classifier_layers.append(nn.Linear(in_features, num_classes))
                self.classifier = nn.Sequential(*classifier_layers)
            elif backbone_name in ['resnet18', 'resnet50']:
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = getattr(tv_models, backbone_name)(weights=weights)
                in_features = backbone.fc.in_features
                backbone.fc = nn.Identity()
                self.features = backbone
                self.avgpool = nn.Identity()
                self.classifier = nn.Linear(in_features, num_classes)
            elif backbone_name.startswith('efficientnet'):
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = getattr(tv_models, backbone_name)(weights=weights)
                in_features = backbone.classifier[-1].in_features
                backbone.classifier = nn.Identity()
                self.features = backbone
                self.avgpool = nn.Identity()
                self.classifier = nn.Linear(in_features, num_classes)
            elif backbone_name == 'densenet121':
                weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = tv_models.densenet121(weights=weights)
                in_features = backbone.classifier.in_features
                backbone.classifier = nn.Identity() # Entferne den originalen Klassifikator
                self.features = backbone.features
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Stellt sicher, dass der Output (1,1) ist
                self.classifier = nn.Linear(in_features, num_classes)
            else: raise ValueError(f"Unbekannter Backbone: {backbone_name}")
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1) # Flatten
            x = self.classifier(x)
            return x

    # --- get_transforms Definition (von dir bereitgestellt) ---
    # (Achte darauf, dass transforms hier als T importiert wurde, also T.Compose etc.)
    def get_transforms_local(cfg_augment: dict, img_h_config: int, img_w_config: int, is_train: bool = True):
        aug = cfg_augment
        pil_transforms_list = []
        if is_train:
            if aug.get("use_resizedcrop", False): pil_transforms_list.append(T.RandomResizedCrop((img_h_config, img_w_config), scale=(0.8, 1.0), ratio=(0.9, 1.1)))
            else: pil_transforms_list.append(T.Resize((img_h_config, img_w_config)))
            if aug.get("use_hflip", False): pil_transforms_list.append(T.RandomHorizontalFlip())
            if aug.get("use_vflip", False): pil_transforms_list.append(T.RandomVerticalFlip())
            if aug.get("use_rotation", False): pil_transforms_list.append(T.RandomRotation(aug.get("rotation_degree", 15)))
            if aug.get("use_colorjitter", False): pil_transforms_list.append(T.ColorJitter(brightness=aug.get("cj_brightness", 0.2), contrast=aug.get("cj_contrast", 0.2), saturation=aug.get("cj_saturation", 0.2), hue=aug.get("cj_hue", 0.1)))
            if aug.get("use_perspective", False): pil_transforms_list.append(T.RandomPerspective(distortion_scale=aug.get("perspective_distortion_scale",0.5), p=aug.get("perspective_p", 0.5)))
        else: pil_transforms_list.append(T.Resize((img_h_config, img_w_config)))
        pil_transforms_list.append(T.ToTensor())
        tensor_transforms_list = []
        if is_train and aug.get("use_blur", False): tensor_transforms_list.append(T.GaussianBlur(kernel_size=tuple(aug.get("blur_kernel", [3,3])), sigma=tuple(aug.get("blur_sigma", [0.1, 2.0]))))
        if is_train and aug.get("use_erasing", False): tensor_transforms_list.append(T.RandomErasing(p=aug.get("erasing_p",0.5), scale=tuple(aug.get("erasing_scale", [0.02, 0.33])), ratio=tuple(aug.get("erasing_ratio", [0.3, 3.3]))))
        tensor_transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return T.Compose(pil_transforms_list + tensor_transforms_list)
    
    # Definition für create_full_mask_from_png_object (aus utils.py, vereinfacht für den Fall, dass utils nicht importierbar ist)
    import base64, zlib, io # Für Fallback
    def create_full_mask_from_png_object_local(b64_str, origin_xy, target_h, target_w): # Dummy, wenn utils.py nicht da ist
        print("WARNUNG: utils.create_full_mask_from_png_object nicht importiert. GT-Masken können nicht geladen werden.")
        return np.zeros((target_h, target_w), dtype=np.uint8)
    
    # Überschreibe die importierten Funktionen mit den lokalen, falls Import fehlschlug
    if 'ClassifierModel' not in globals(): ClassifierModel = ClassifierModel
    if 'get_transforms' not in globals(): get_transforms = get_transforms_local
    if 'create_full_mask_from_png_object' not in globals(): create_full_mask_from_png_object = create_full_mask_from_png_object_local


# --- Globale Konfigurationen ---
MODEL_PATH = "../checkpoints/densenet121/severstal/best_model_severstal_patches.pt"
PATCH_H, PATCH_W = 256, 256
STRIDE_H, STRIDE_W = 128, 128
NUM_CLASSES = 5  # Inklusive "kein Defekt"
MODEL_BACKBONE = 'densenet121' # Wichtig für das Laden des Modells
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard-Augmentierungskonfiguration für get_transforms (entspricht typischerweise cfg.augment aus train.py)
# Diese wird für is_train=False verwendet, also sind die meisten 'use_...' Flags irrelevant.
DEFAULT_AUG_CFG = {
    "enable": False, # Für Inferenz typischerweise False oder nur Resize/ToTensor/Normalize
    "use_resizedcrop": False, "use_hflip": False, "use_vflip": False,
    "use_rotation": False, "use_colorjitter": False, "use_perspective": False,
    "use_blur": False, "use_erasing": False,
    # Parameter, die von get_transforms erwartet werden könnten, auch wenn nicht genutzt:
    "rotation_degree": 15, "cj_brightness": 0.2, "cj_contrast": 0.2,
    "cj_saturation": 0.2, "cj_hue": 0.1, "perspective_distortion_scale":0.5,
    "perspective_p":0.5, "blur_kernel": [3,3], "blur_sigma": [0.1,2.0],
    "erasing_p":0.5, "erasing_scale": [0.02,0.33], "erasing_ratio":[0.3,3.3]
}
# Erstelle ein einfaches DictConfig-ähnliches Objekt für get_transforms
# Wenn du Hydra oder OmegaConf nicht global installieren willst/kannst
class SimpleCfg:
    def __init__(self, data):
        self.augment = data

CFG_FOR_TRANSFORMS = SimpleCfg(DEFAULT_AUG_CFG)


# Defektklassen-Namen (optional, für bessere Lesbarkeit der Ausgabe)
CLASS_NAMES = ["No Defect", "Defect 1", "Defect 2", "Defect 3", "Defect 4"]
DEFECT_CLASS_TO_IDX = {"defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4} # Für GT-Masken


def load_trained_model(model_path: str, backbone_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Lädt das trainierte Klassifikationsmodell."""
    model = ClassifierModel(backbone_name=backbone_name, pretrained=False, num_classes=num_classes) # pretrained=False, da wir Gewichte laden
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modell erfolgreich von {model_path} geladen.")
    except FileNotFoundError:
        print(f"FEHLER: Modelldatei nicht gefunden unter {model_path}")
        exit()
    except RuntimeError as e:
        print(f"FEHLER: RuntimeError beim Laden des Modells (möglicherweise falsche Architektur/Klassenzahl?): {e}")
        exit()
    model.to(device)
    model.eval()
    return model

def extract_patches_and_predict(
    image_pil: Image.Image,
    model: nn.Module,
    transform: T.Compose,
    patch_size_hw: Tuple[int, int],
    stride_hw: Tuple[int, int],
    device: torch.device,
    prediction_threshold: float = 0.5 # Standard-Threshold für binäre Klassifikation
) -> List[dict]:
    """
    Extrahiert Patches aus einem Bild, führt Inferenz durch und gibt Infos zurück.
    """
    patch_h, patch_w = patch_size_hw
    stride_h, stride_w = stride_hw
    original_w, original_h = image_pil.size # PIL Image size ist (width, height)

    patches_data = []
    batch_patches_pil = []
    batch_coords = []
    
    # Um die transformierten Patches zu sammeln für einen Batch-Durchlauf
    transformed_patch_tensors = []


    print(f"Extrahiere Patches aus Bild der Größe {original_w}x{original_h}...")
    for y in range(0, original_h - patch_h + 1, stride_h):
        for x in range(0, original_w - patch_w + 1, stride_w):
            patch_pil = image_pil.crop((x, y, x + patch_w, y + patch_h))
            
            # Transformation anwenden (sollte ToTensor und Normalize enthalten)
            # Die get_transforms Funktion wird hier mit is_train=False aufgerufen
            transformed_patch = transform(patch_pil)
            
            patches_data.append({
                "coords": (x, y),
                "patch_pil": patch_pil, # Speichere das un-transformierte PIL-Image für die Visualisierung
                "transformed_tensor": transformed_patch # Für die Inferenz
            })

    if not patches_data:
        print("Keine Patches konnten extrahiert werden.")
        return []

    # Inferenz in Batches (optional, hier erstmal einzeln pro Patch für Einfachheit, kann aber langsam sein)
    # Für bessere Performance: Batched Inference
    print(f"Führe Inferenz für {len(patches_data)} Patches durch...")
    
    # Erstelle einen Batch von Tensoren
    batch_transformed_tensors = torch.stack([p_data["transformed_tensor"] for p_data in patches_data])
    
    results_for_patches = []
    
    # Hier könnte man batch_size für die Inferenz einführen, falls Speicherprobleme auftreten
    # Für jetzt: alle auf einmal, wenn es passt
    with torch.no_grad():
        outputs = model(batch_transformed_tensors.to(device))
        probabilities = torch.sigmoid(outputs).cpu().numpy() # Wahrscheinlichkeiten [0,1]
        # Binäre Vorhersagen basierend auf dem Threshold
        # predicted_classes_binary = (probabilities > prediction_threshold).astype(int)


    for i, p_data in enumerate(patches_data):
        results_for_patches.append({
            "coords": p_data["coords"],
            "patch_pil": p_data["patch_pil"],
            "probabilities": probabilities[i],
            # "predicted_classes": predicted_classes_binary[i] # Binäre Vorhersage
        })
        
    return results_for_patches


def visualize_patch_predictions(
    patch_results: List[dict],
    gt_mask_np_original: Optional[np.ndarray],
    patch_size_hw: Tuple[int, int],
    image_filename: str,
    prediction_threshold: float = 0.5
):
    num_patches = len(patch_results)
    if num_patches == 0:
        print("Keine Patch-Ergebnisse zum Visualisieren.")
        return

    patch_h, patch_w = patch_size_hw
    
    # Jede "logische" Patch-Darstellung braucht jetzt 2 Subplots (Pred | GT)
    # Wir layouten es so, dass Paare untereinander stehen.
    # Max. 3 Paare (also 6 Subplots) pro Zeile für Lesbarkeit.
    plots_per_patch_pair = 2
    pairs_per_row = 3 
    cols_per_fig = pairs_per_row * plots_per_patch_pair # Also 6 Spalten
    
    num_rows_needed = int(np.ceil(num_patches / pairs_per_row))
    num_rows_needed = max(1, num_rows_needed)


    fig, axes = plt.subplots(num_rows_needed, cols_per_fig, figsize=(cols_per_fig * 2.5, num_rows_needed * 3.5))
    if num_patches == 0: # Sollte durch obere Prüfung abgefangen sein
        plt.close(fig)
        return
    if num_rows_needed == 1 and cols_per_fig == 1: # Nur ein Patch, nur ein Paar
         axes = np.array([[axes, axes]]) # Trick um es 2D zu machen für flatten
    elif num_rows_needed == 1 : # Eine Zeile, mehrere Paare
        axes = axes.reshape(1, -1)
    elif cols_per_fig == 1: # Eine Spalte (unwahrscheinlich bei 2 Plots pro Patch)
        axes = axes.reshape(-1,1)

    axes = axes.flatten() # Um immer 1D-Indexierung zu haben

    plot_idx = 0 # Zählt die genutzten Subplots

    for i, res in enumerate(patch_results):
        if plot_idx + 1 >= len(axes): # Nicht genug Platz für das nächste Paar
            print(f"Warnung: Nicht genug Subplots für alle Patches. Stoppe bei Patch {i+1}/{num_patches}.")
            break

        # --- Subplot 1: Patch mit Vorhersage-Text ---
        ax_pred = axes[plot_idx]
        ax_pred.imshow(res["patch_pil"])
        
        title_parts_pred = []
        title_parts_pred.append(f"Patch ({res['coords'][0]},{res['coords'][1]})")
        pred_labels_str_parts = []
        for cls_idx, prob in enumerate(res["probabilities"]):
            is_pred_positive = prob >= prediction_threshold
            class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Cls {cls_idx}"
            pred_labels_str_parts.append(f"{class_name[:3]}: {prob:.2f}{'*' if is_pred_positive else ''}")
        title_parts_pred.append("Pred: " + " | ".join(pred_labels_str_parts))
        ax_pred.set_title("\n".join(title_parts_pred), fontsize=7)
        ax_pred.axis('off')
        plot_idx += 1

        # --- Subplot 2: Patch mit GT-Maske ---
        ax_gt = axes[plot_idx]
        ax_gt.imshow(res["patch_pil"]) # Zeige den originalen Patch als Hintergrund
        
        title_parts_gt = []
        title_parts_gt.append(f"Patch ({res['coords'][0]},{res['coords'][1]}) - GT") # Eindeutiger Titel
        gt_labels_str_parts = []

        if gt_mask_np_original is not None:
            x, y = res["coords"]
            gt_patch_mask_np = gt_mask_np_original[y:y+patch_h, x:x+patch_w]
            
            # GT Label Vektor bestimmen (wie im Testskript für Patch-Extraktion)
            # Hier vereinfacht: Wir nehmen an, min_positive_pixel_abs = 1 für die GT-Anzeige
            gt_patch_label_vector = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            unique_gt_classes = np.unique(gt_patch_mask_np)
            has_gt_defect = False
            for gt_cls_val in unique_gt_classes:
                if gt_cls_val >=1 and gt_cls_val <=4:
                    if np.sum(gt_patch_mask_np == gt_cls_val) > 0: # Mindestens 1 Pixel
                        gt_patch_label_vector[gt_cls_val] = 1.0
                        has_gt_defect = True
            if not has_gt_defect and np.all(gt_patch_mask_np == 0):
                 gt_patch_label_vector[0] = 1.0
            
            for cls_idx, val in enumerate(gt_patch_label_vector):
                if val == 1.0:
                    class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Cls {cls_idx}"
                    gt_labels_str_parts.append(class_name)
            title_parts_gt.append("GT Classes: " + (", ".join(gt_labels_str_parts) if gt_labels_str_parts else "None"))

            # GT Maske über den Patch legen
            # Farben für Defektklassen-Overlays (ähnlich wie in test_patch_extraction)
            gt_overlay_pil = Image.new("RGBA", res["patch_pil"].size, (0,0,0,0))
            gt_draw = ImageDraw.Draw(gt_overlay_pil)
            
            # MASK_COLORS_DEFECTS sind für Defekte 1-4, Indizes 0-3
            # combined_gt_mask_np enthält Werte 1-4 für Defekte
            temp_mask_colors_rgba = [ 
                (255, 0, 0, 100),   # Defect 1 (Rot)
                (0, 255, 0, 100),   # Defect 2 (Grün)
                (0, 0, 255, 100),   # Defect 3 (Blau)
                (255, 255, 0, 100)  # Defect 4 (Gelb)
            ]
            for defect_val_in_gt in range(1, NUM_CLASSES): # Defektklassen 1 bis 4
                if gt_patch_label_vector[defect_val_in_gt] == 1.0: # Wenn diese Klasse im GT-Label des Patches ist
                    # Finde Pixel in gt_patch_mask_np, die diesen Wert haben
                    # und zeichne sie mit der entsprechenden Farbe
                    mask_pixels_for_defect = (gt_patch_mask_np == defect_val_in_gt)
                    if np.any(mask_pixels_for_defect) and (defect_val_in_gt -1) < len(temp_mask_colors_rgba):
                        color = temp_mask_colors_rgba[defect_val_in_gt-1]
                        ys_gt, xs_gt = np.where(mask_pixels_for_defect)
                        for r_gt, c_gt in zip(ys_gt, xs_gt):
                            gt_draw.point((c_gt, r_gt), fill=color)
            
            ax_gt.imshow(gt_overlay_pil, alpha=0.5) # Überlagere die GT-Maske
        else:
            title_parts_gt.append("GT: N/A (no ann_dir)")

        ax_gt.set_title("\n".join(title_parts_gt), fontsize=7)
        ax_gt.axis('off')
        plot_idx += 1

    # Ungenutzte Subplots ausblenden
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output_filename = f"patch_pred_gt_{os.path.splitext(os.path.basename(image_filename))[0]}.png"
    plt.savefig(output_filename)
    print(f"Patch-Vorhersagen mit GT gespeichert als: {output_filename}")
    plt.close(fig)


def visualize_stitched_predictions(
    image_pil: Image.Image,
    patch_results: List[dict],
    patch_size_hw: Tuple[int, int],
    image_filename: str,
    prediction_threshold: float = 0.5,
    alpha: float = 0.4 # Transparenz des Overlays
):
    """Visualisiert das Originalbild mit überlagerten Klassifikationsergebnissen."""
    patch_h, patch_w = patch_size_hw
    stitched_heatmap = np.zeros((image_pil.height, image_pil.width, NUM_CLASSES), dtype=float)
    # Zähler für Überlappungen, um Mittelwert zu bilden
    overlap_counter = np.zeros((image_pil.height, image_pil.width, NUM_CLASSES), dtype=int)


    for res in patch_results:
        x, y = res["coords"]
        probs = res["probabilities"] # Shape (NUM_CLASSES,)
        
        stitched_heatmap[y:y+patch_h, x:x+patch_w, :] += probs
        overlap_counter[y:y+patch_h, x:x+patch_w, :] += 1

    # Mittelwert bilden wo Überlappungen sind
    # Vermeide Division durch Null, setze auf 1 wo Zähler 0 ist (sollte nicht passieren für genutzte Bereiche)
    overlap_counter[overlap_counter == 0] = 1
    stitched_heatmap /= overlap_counter
    
    # Für die Visualisierung: erstelle eine Maske für jede Klasse, die den Threshold überschreitet
    # Wir könnten die stärkste Klasse pro Pixel anzeigen oder alle überlappenden Klassen
    # Hier: Zeige für jede Defektklasse (1-4) ein Overlay, wenn ihre gemittelte Wahrscheinlichkeit hoch ist
    
    # Kopie des Originalbilds für das Overlay
    overlay_image_pil = image_pil.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay_image_pil, "RGBA")

    # Farben für Defektklassen-Overlays (ähnlich wie in test_patch_extraction)
    # Ohne "No Defect", da wir Defekte hervorheben wollen
    defect_colors_rgba = [ 
        (255, 0, 0, int(255*alpha)),   # Defect 1 (Rot)
        (0, 255, 0, int(255*alpha)),   # Defect 2 (Grün)
        (0, 0, 255, int(255*alpha)),   # Defect 3 (Blau)
        (255, 255, 0, int(255*alpha))  # Defect 4 (Gelb)
    ]

    for class_idx in range(1, NUM_CLASSES): # Iteriere nur über Defektklassen (Index 1 bis 4)
        # Erstelle eine binäre Maske für die aktuelle Klasse, wo die Wahrscheinlichkeit > Threshold ist
        class_pred_mask = (stitched_heatmap[:, :, class_idx] > prediction_threshold)
        
        # Wenn es Pixel für diese Klasse gibt, zeichne sie ein
        if np.any(class_pred_mask) and (class_idx -1) < len(defect_colors_rgba) :
            # Erstelle ein temporäres farbiges Overlay für diese Klasse
            color = defect_colors_rgba[class_idx-1] # -1 weil defect_colors_rgba bei 0 beginnt für Defekt 1
            # Konvertiere die binäre Maske in ein PIL-Image, das als Maske für fill dient
            # Wichtig: Die Maske für `draw.bitmap` oder `Image.paste` sollte '1', 'L' oder 'RGBA' sein.
            # Hier verwenden wir eine einfache Pixel-für-Pixel-Zeichnung für Transparenz.
            
            # Erstelle ein leeres Overlay für diese Klasse
            single_class_overlay_pil = Image.new("RGBA", image_pil.size, (0,0,0,0))
            single_class_draw = ImageDraw.Draw(single_class_overlay_pil)

            # Finde die Indizes, wo die Klasse vorhergesagt wird
            ys, xs = np.where(class_pred_mask)
            for r, c in zip(ys, xs):
                single_class_draw.point((c,r), fill=color) # c ist x, r ist y
            
            # Komponiere dieses Overlay auf das Gesamt-Overlay-Bild
            overlay_image_pil.alpha_composite(single_class_overlay_pil)


    fig, ax = plt.subplots(1, 1, figsize=(12, (12 / image_pil.width * image_pil.height) ))
    ax.imshow(overlay_image_pil)
    ax.set_title(f"Gestitchte Vorhersagen (Thresh: {prediction_threshold:.2f})")
    ax.axis('off')
    plt.tight_layout()
    output_filename = f"stitched_predictions_{os.path.splitext(os.path.basename(image_filename))[0]}.png"
    plt.savefig(output_filename)
    print(f"Gestitchte Vorhersagen gespeichert als: {output_filename}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspiziert Patch-Klassifikationen für ein einzelnes Bild.")
    parser.add_argument("image_path", type=str, help="Pfad zum zu analysierenden Originalbild.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Pfad zum trainierten Modell.")
    parser.add_argument("--ann_dir", type=str, default=None, help="Optional: Pfad zum Annotationsverzeichnis für GT-Labels.")
    parser.add_argument("--threshold", type=float, default=0.65, help="Vorhersage-Threshold für die Anzeige positiver Klassen (basierend auf deinem besten Validierungs-Threshold).") # Dein bester Threshold

    args = parser.parse_args()

    # Lade Originalbild
    try:
        original_image_pil = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"FEHLER: Originalbild nicht gefunden unter {args.image_path}")
        return
    except UnidentifiedImageError:
        print(f"FEHLER: Datei unter {args.image_path} ist kein gültiges Bildformat.")
        return
    
    # Lade Modell
    model = load_trained_model(args.model_path, MODEL_BACKBONE, NUM_CLASSES, DEVICE)

    # Hole Transformationen (für Inferenz, also is_train=False)
    # Wir verwenden die cfg-Struktur, die get_transforms erwartet
    inference_transforms = get_transforms(CFG_FOR_TRANSFORMS, PATCH_H, PATCH_W, is_train=False)
    
    # Extrahiere Patches und mache Vorhersagen
    patch_results = extract_patches_and_predict(
        original_image_pil, model, inference_transforms,
        (PATCH_H, PATCH_W), (STRIDE_H, STRIDE_W), DEVICE,
        prediction_threshold=args.threshold # Wird intern nicht für binäre Klass. genutzt, nur für Info
    )

    if not patch_results:
        return

    # Lade optionale Ground-Truth Maske
    gt_mask_original_np = None
    if args.ann_dir:
        img_filename = os.path.basename(args.image_path)
        base_name_no_ext = os.path.splitext(img_filename)[0]
        ann_path_variants = [
            os.path.join(args.ann_dir, f"{img_filename}.json"),
            os.path.join(args.ann_dir, f"{base_name_no_ext}.json")
        ]
        ann_path = None
        for p_variant in ann_path_variants:
            if os.path.exists(p_variant):
                ann_path = p_variant
                break
        if ann_path:
            print(f"Lade GT-Annotation von: {ann_path}")
            gt_mask_original_np = np.zeros((original_image_pil.height, original_image_pil.width), dtype=np.uint8)
            try:
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
                if "objects" in annotation and annotation["objects"]:
                    for obj in annotation["objects"]:
                        class_title = obj.get("classTitle")
                        if class_title in DEFECT_CLASS_TO_IDX:
                            class_idx_for_mask = DEFECT_CLASS_TO_IDX[class_title]
                            if "bitmap" in obj:
                                b64_data = obj["bitmap"].get("data")
                                origin = obj["bitmap"].get("origin")
                                if b64_data and origin:
                                    single_obj_mask_np = create_full_mask_from_png_object(
                                        b64_data, origin, original_image_pil.height, original_image_pil.width
                                    )
                                    if single_obj_mask_np is not None:
                                        gt_mask_original_np[single_obj_mask_np == 1] = class_idx_for_mask
            except Exception as e:
                print(f"Warnung: Fehler beim Laden der GT-Annotation: {e}")
                gt_mask_original_np = None # Fallback
        else:
            print(f"Keine Annotationsdatei für {img_filename} in {args.ann_dir} gefunden.")


    # Visualisiere Patch-Vorhersagen
    visualize_patch_predictions(patch_results, gt_mask_original_np, (PATCH_H, PATCH_W), args.image_path, args.threshold)

    # Visualisiere gestitchte Vorhersagen
    visualize_stitched_predictions(original_image_pil, patch_results, (PATCH_H, PATCH_W), args.image_path, args.threshold)

    print("\nInspektion abgeschlossen.")

if __name__ == "__main__":
    main()