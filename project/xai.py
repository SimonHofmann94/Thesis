# project/xai.py

import os
import torch
import hydra
import json
from omegaconf import DictConfig, ListConfig
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader # random_split, Subset nicht mehr direkt hier gebraucht

from models import ClassifierModel
from datasets import SeverstalDataset # Nur für Konstanten wie NUM_TOTAL_CLASSES, DEFECT_CLASS_TO_IDX
from train import get_transforms     # Importiere deine get_transforms Funktion
from utils import create_full_mask_from_png_object
from captum.attr import LRP
from captum.attr import visualization as viz 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

CLASS_NAMES = ["No Defect", "Defect 1", "Defect 2", "Defect 3", "Defect 4"] 
NUM_TOTAL_CLASSES = 5
DEFECT_CLASS_TO_IDX = {"defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4}

def denormalize_image(tensor_image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalisiert einen Tensor-Bild für die Visualisierung unter Verwendung von Out-of-Place-Operationen."""
    denormalized_channels = []
    for channel_idx in range(tensor_image.shape[0]):
        t_ch = tensor_image[channel_idx] 
        m = mean[channel_idx]
        s = std[channel_idx]
        denorm_ch = t_ch.mul(s).add(m) 
        denormalized_channels.append(denorm_ch)
    return torch.stack(denormalized_channels, dim=0)

def aggregate_lrp_channels(attribution_np_channels_first: np.ndarray, method: str = "sum_positive") -> np.ndarray:
    """
    Aggregiert die Kanal-Dimension einer LRP-Attributionskarte.
    Input: attribution_np (C, H, W)
    Output: aggregated_map (H, W)
    """
    if attribution_np_channels_first.ndim != 3:
        # Kann passieren, wenn LRP für ein einkanaliges Modell oder einen einkanaligen Input läuft
        # Oder wenn die Attribution schon aggregiert wurde.
        if attribution_np_channels_first.ndim == 2:
            # print(f"Warnung: aggregate_lrp_channels erhielt bereits 2D Attribution (Shape: {attribution_np_channels_first.shape}). Gebe sie direkt zurück.")
            return attribution_np_channels_first
        raise ValueError(f"Erwartet 3D Attribution (C,H,W) oder 2D (H,W), bekam {attribution_np_channels_first.shape}")

    if method == "sum":
        return np.sum(attribution_np_channels_first, axis=0)
    elif method == "mean":
        return np.mean(attribution_np_channels_first, axis=0)
    elif method == "max": # Max über Kanäle
        return np.max(attribution_np_channels_first, axis=0)
    elif method == "sum_positive":
        return np.sum(np.maximum(0, attribution_np_channels_first), axis=0)
    elif method == "abs_max_val": # Nimmt den Wert des Kanals, der den größten absoluten Beitrag hat
        # Finde den Index des Kanals mit dem größten absoluten Wert für jedes Pixel
        max_abs_channel_indices = np.argmax(np.abs(attribution_np_channels_first), axis=0)
        # Erstelle Gitter für die Indizierung
        h_dim, w_dim = attribution_np_channels_first.shape[1], attribution_np_channels_first.shape[2]
        row_indices, col_indices = np.ogrid[:h_dim, :w_dim]
        # Wähle die Werte aus den entsprechenden Kanälen
        return attribution_np_channels_first[max_abs_channel_indices, row_indices, col_indices]
    else:
        print(f"Warnung: Unbekannte LRP-Kanal-Aggregationsmethode '{method}'. Verwende 'sum_positive'.")
        return np.sum(np.maximum(0, attribution_np_channels_first), axis=0)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("Starte XAI (LRP) für Bilder aus Verzeichnis...")
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Verwende Device: {device}, Projekt-Root: {project_root}")

    # --- 1. Bestes Modell laden ---
    dataset_name_for_log = cfg.data.get("name", "unknown_dataset")
    model_name_suffix = f"best_model_{dataset_name_for_log}_patches.pt"
    model_path = os.path.join(project_root, cfg.train.ckpt_dir, cfg.model.backbone,
                              dataset_name_for_log, model_name_suffix)

    if not os.path.exists(model_path):
        print(f"FEHLER: Modelldatei nicht gefunden: {model_path}")
        return
    print(f"Lade Modell von: {model_path}")
    model = ClassifierModel( # Stelle sicher, dass ClassifierModel importiert/definiert ist
        backbone_name=cfg.model.backbone, pretrained=False,
        num_classes=SeverstalDataset.NUM_TOTAL_CLASSES # Stelle sicher, dass SeverstalDataset importiert/definiert ist
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Modell geladen.")

    # --- Parameter für Patch-Extraktion und XAI ---
    patch_cfg = cfg.data.severstal.patch_processing
    patch_size_hw = tuple(patch_cfg.patch_size_hw)
    patch_h, patch_w = patch_size_hw[0], patch_size_hw[1]
    stride_hw_cfg = patch_cfg.get("stride_hw_xai_analysis", patch_size_hw)
    stride_h, stride_w = tuple(stride_hw_cfg) if isinstance(stride_hw_cfg, (ListConfig, list, tuple)) else (stride_hw_cfg, stride_hw_cfg)

    # Transformationen für die Patches (Validierungs-Transforms)
    patch_transform = get_transforms(cfg, img_h_config=patch_h, img_w_config=patch_w, is_train=False) # Stelle sicher, dass get_transforms importiert/definiert ist

    # LRP Instanz
    lrp = LRP(model)

    # Output Verzeichnisse
    xai_run_name = cfg.xai.get("run_name", "default_xai_run")
    method_name = "LRP"
    base_attribution_dir = os.path.join(project_root, cfg.output.attributions_dir, cfg.model.backbone, f"{method_name}_{xai_run_name}")
    os.makedirs(base_attribution_dir, exist_ok=True)

    save_visualizations = cfg.xai.get("save_visualizations", True)
    visualization_dir = os.path.join(base_attribution_dir, "visualizations")
    if save_visualizations:
        os.makedirs(visualization_dir, exist_ok=True)

    # --- NEUE Logik für Ordner-Input ---
    xai_input_img_dir_rel = cfg.xai.get("input_img_dir")
    if not xai_input_img_dir_rel:
        print("FEHLER: 'xai.input_img_dir' nicht in der Konfiguration spezifiziert. Stoppe.")
        return
    xai_input_img_dir_abs = os.path.join(project_root, xai_input_img_dir_rel)
    if not os.path.isdir(xai_input_img_dir_abs):
        print(f"FEHLER: XAI Input-Bildverzeichnis nicht gefunden: {xai_input_img_dir_abs}")
        return
    print(f"Verarbeite Bilder aus Verzeichnis: {xai_input_img_dir_abs}")

    image_files_to_process = sorted([
        f for f in os.listdir(xai_input_img_dir_abs)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files_to_process:
        print(f"Keine Bilddateien im Verzeichnis {xai_input_img_dir_abs} gefunden. Stoppe.")
        return

    # Optional: Lade Pfad zum Annotationsverzeichnis für GT-Masken
    xai_input_ann_dir_abs = None
    xai_input_ann_dir_rel = cfg.xai.get("input_ann_dir")
    if xai_input_ann_dir_rel:
        temp_ann_dir = os.path.join(project_root, xai_input_ann_dir_rel)
        if os.path.isdir(temp_ann_dir):
            xai_input_ann_dir_abs = temp_ann_dir
            print(f"Verwende Annotationsverzeichnis für GT-Masken: {xai_input_ann_dir_abs}")
        else:
            print(f"Warnung: Annotationsverzeichnis '{temp_ann_dir}' nicht gefunden. GT-Masken werden nicht geladen.")

    prediction_threshold = cfg.eval.get("prediction_threshold", 0.7)
    lrp_agg_method = cfg.xai.get("lrp_channel_aggregation_method", "sum_positive")

    print(f"Analysiere {len(image_files_to_process)} Originalbilder. Patch-Größe: {patch_size_hw}, Stride: {(stride_h,stride_w)}")
    print(f"LRP Aggregationsmethode: {lrp_agg_method}")

    for img_filename_with_ext in image_files_to_process:
        original_img_path = os.path.join(xai_input_img_dir_abs, img_filename_with_ext)
        # File existence is already checked by os.listdir, but an explicit check here is fine
        if not os.path.exists(original_img_path): # Sollte eigentlich nicht passieren
            print(f"Warnung: Originalbild {original_img_path} nicht gefunden (unerwartet). Überspringe.")
            continue

        print(f"\n--- Analysiere Originalbild: {img_filename_with_ext} ---")
        try:
            image_pil_original = Image.open(original_img_path).convert("RGB")
        except Exception as e_img_load:
            print(f"FEHLER beim Laden des Bildes {original_img_path}: {e_img_load}. Überspringe.")
            continue
            
        original_h_img, original_w_img = image_pil_original.height, image_pil_original.width

        # Lade GT-Maske für das aktuelle Originalbild, falls ann_dir gegeben ist
        gt_mask_original_np = None
        if xai_input_ann_dir_abs:
            base_name_no_ext = os.path.splitext(img_filename_with_ext)[0]
            ann_path_variants = [
                os.path.join(xai_input_ann_dir_abs, f"{img_filename_with_ext}.json"),
                os.path.join(xai_input_ann_dir_abs, f"{base_name_no_ext}.json")
            ]
            ann_path = None
            for p_variant in ann_path_variants:
                if os.path.exists(p_variant):
                    ann_path = p_variant
                    break
            if ann_path:
                try:
                    gt_mask_original_np = np.zeros((original_h_img, original_w_img), dtype=np.uint8)
                    with open(ann_path, 'r') as f:
                        annotation = json.load(f)
                    if "objects" in annotation and annotation["objects"]:
                        for obj in annotation["objects"]:
                            class_title = obj.get("classTitle")
                            if class_title in SeverstalDataset.DEFECT_CLASS_TO_IDX:
                                class_idx_for_mask = SeverstalDataset.DEFECT_CLASS_TO_IDX[class_title]
                                if "bitmap" in obj:
                                    b64_data = obj["bitmap"].get("data")
                                    origin_coords = obj["bitmap"].get("origin") # Renamed 'origin' to 'origin_coords'
                                    if b64_data and origin_coords:
                                        single_obj_mask_np = create_full_mask_from_png_object(
                                            b64_data, origin_coords, original_h_img, original_w_img
                                        )
                                        if single_obj_mask_np is not None:
                                            gt_mask_original_np[single_obj_mask_np == 1] = class_idx_for_mask
                except Exception as e_ann:
                    print(f"    Warnung: Fehler beim Laden/Verarbeiten der Annotation {ann_path} für {img_filename_with_ext}: {e_ann}")
                    gt_mask_original_np = None
            # else:
                # print(f"    Info: Keine Annotationsdatei für {img_filename_with_ext} gefunden.")


        # Extrahiere Patches aus diesem Bild
        for y_coord in range(0, original_h_img - patch_h + 1, stride_h):
            for x_coord in range(0, original_w_img - patch_w + 1, stride_w):
                patch_pil = image_pil_original.crop((x_coord, y_coord, x_coord + patch_w, y_coord + patch_h))
                patch_id = f"{os.path.splitext(img_filename_with_ext)[0]}_x{x_coord}_y{y_coord}"

                input_patch_tensor = patch_transform(patch_pil).unsqueeze(0).to(device)
                input_patch_tensor.requires_grad_(True) # Wichtig für LRP mit Captum

                logits = model(input_patch_tensor)
                probs_single_patch = torch.sigmoid(logits).squeeze(0)

                for class_idx_model_output in range(1, SeverstalDataset.NUM_TOTAL_CLASSES): # Defektklassen 1-4
                    if probs_single_patch[class_idx_model_output] > prediction_threshold:
                        target_class_hr_name = "UnknownDefect"
                        for name, idx_map in SeverstalDataset.DEFECT_CLASS_TO_IDX.items():
                            if idx_map == class_idx_model_output: target_class_hr_name = name; break

                        print(f"  Verarbeite Patch {patch_id} für Klasse '{target_class_hr_name}' (Prob: {probs_single_patch[class_idx_model_output]:.3f})")

                        attribution = lrp.attribute(input_patch_tensor, target=class_idx_model_output)
                        attribution_np_ch_first = attribution.squeeze(0).cpu().detach().numpy()

                        class_raw_npy_dir = os.path.join(base_attribution_dir, target_class_hr_name, "npy_raw")
                        os.makedirs(class_raw_npy_dir, exist_ok=True)
                        np.save(os.path.join(class_raw_npy_dir, f"{patch_id}_attr_raw.npy"), attribution_np_ch_first)

                        aggregated_attr_map_2d = aggregate_lrp_channels(attribution_np_ch_first, method=lrp_agg_method) # Stelle sicher, dass aggregate_lrp_channels definiert ist

                        class_agg_npy_dir = os.path.join(base_attribution_dir, target_class_hr_name, f"npy_agg_{lrp_agg_method}")
                        os.makedirs(class_agg_npy_dir, exist_ok=True)
                        np.save(os.path.join(class_agg_npy_dir, f"{patch_id}_attr_agg.npy"), aggregated_attr_map_2d)

                        if save_visualizations:
                            patch_for_denorm = input_patch_tensor.squeeze(0).cpu().detach()
                            denormalized_patch_tensor = denormalize_image(patch_for_denorm) # Stelle sicher, dass denormalize_image definiert ist
                            patch_for_viz_np = np.transpose(denormalized_patch_tensor.numpy(), (1,2,0))
                            patch_for_viz_np = np.clip(patch_for_viz_np, 0, 1)

                            attr_for_captum_viz = aggregated_attr_map_2d
                            if aggregated_attr_map_2d.ndim == 2:
                                attr_for_captum_viz = aggregated_attr_map_2d[:, :, np.newaxis]
                            
                            # --- NEUE KOMBINIERTE VISUALISIERUNG ---
                            fig_combo, axes_combo = plt.subplots(1, 3, figsize=(cfg.xai.get("viz_fig_width_combo", 18), cfg.xai.get("viz_fig_height_combo", 6)))
                            title_prefix = f"Patch: {patch_id}\nPred: {target_class_hr_name} (Prob: {probs_single_patch[class_idx_model_output]:.2f})"

                            axes_combo[0].imshow(patch_for_viz_np)
                            axes_combo[0].set_title("Original Patch", fontsize=9)
                            axes_combo[0].axis('off')

                            axes_combo[1].imshow(patch_for_viz_np)
                            gt_title_suffix = "GT nicht geladen"
                            if gt_mask_original_np is not None:
                                gt_patch_mask_np = gt_mask_original_np[y_coord:y_coord+patch_h, x_coord:x_coord+patch_w]
                                gt_overlay_pil = Image.new("RGBA", patch_pil.size, (0,0,0,0))
                                gt_draw = ImageDraw.Draw(gt_overlay_pil)
                                temp_mask_colors_rgba = [(255,0,0,100), (0,255,0,100), (0,0,255,100), (255,255,0,100)] # Rot, Grün, Blau, Gelb
                                gt_classes_present_str_list = []

                                for defect_val_in_gt in range(1, SeverstalDataset.NUM_TOTAL_CLASSES):
                                    if np.sum(gt_patch_mask_np == defect_val_in_gt) > 0:
                                        gt_class_name_temp = "UnknownGT"
                                        for name, idx_map_gt in SeverstalDataset.DEFECT_CLASS_TO_IDX.items():
                                            if idx_map_gt == defect_val_in_gt: gt_class_name_temp = name; break
                                        gt_classes_present_str_list.append(gt_class_name_temp)
                                        
                                        mask_pixels_for_defect = (gt_patch_mask_np == defect_val_in_gt)
                                        if np.any(mask_pixels_for_defect) and (defect_val_in_gt - 1) < len(temp_mask_colors_rgba):
                                            color = temp_mask_colors_rgba[defect_val_in_gt - 1]
                                            ys_gt, xs_gt = np.where(mask_pixels_for_defect)
                                            for r_gt, c_gt in zip(ys_gt, xs_gt):
                                                gt_draw.point((c_gt, r_gt), fill=color)
                                axes_combo[1].imshow(gt_overlay_pil, alpha=0.6)
                                gt_title_suffix = ", ".join(gt_classes_present_str_list) if gt_classes_present_str_list else "No Defect (GT)"
                            
                            axes_combo[1].set_title(f"Patch mit GT ({gt_title_suffix})", fontsize=9)
                            axes_combo[1].axis('off')
                              # LRP Heatmap (Subplot 3)
                            try:
                                _ , captum_axes = viz.visualize_image_attr(
                                    attr_for_captum_viz, patch_for_viz_np,
                                    method='blended_heat_map', sign='positive',
                                    show_colorbar=False, use_pyplot=False,
                                    fig_size=(6,6) # Dummy-Größe, da wir es in unser Subplot plotten
                                )
                                if captum_axes is not None:
                                    # Prüfe ob es ein einzelnes Axes-Objekt ist oder eine Sammlung
                                    try:
                                        if hasattr(captum_axes, 'images') and captum_axes.images:
                                            # Einzelnes Axes-Objekt
                                            im_data_lrp = captum_axes.images[0].get_array()
                                            axes_combo[2].imshow(im_data_lrp)
                                            if hasattr(captum_axes, 'figure') and captum_axes.figure is not None:
                                                plt.close(captum_axes.figure)
                                        elif isinstance(captum_axes, (list, tuple)) and len(captum_axes) > 0:
                                            # Liste von Axes
                                            im_data_lrp = captum_axes[0].images[0].get_array()
                                            axes_combo[2].imshow(im_data_lrp)
                                            if captum_axes[0].figure is not None:
                                                plt.close(captum_axes[0].figure)
                                        else:
                                            raise ValueError("Unerwartetes Format der captum_axes")
                                    except Exception as e:
                                        print(f"    Warnung: Fehler beim Zugriff auf captum_axes: {e}")
                                        axes_combo[2].imshow(patch_for_viz_np) # Fallback
                                        axes_combo[2].text(0.5, 0.5, 'LRP Format Error', ha='center', va='center', transform=axes_combo[2].transAxes, color='red')
                                else: # Fallback, falls captum_axes None ist
                                    axes_combo[2].imshow(patch_for_viz_np) # Zeige nur Original, wenn LRP-Viz fehlschlägt
                                    axes_combo[2].text(0.5, 0.5, 'LRP Viz Error', horizontalalignment='center', verticalalignment='center', transform=axes_combo[2].transAxes, color='red')

                            except Exception as e_lrp_viz_subplot:
                                print(f"    FEHLER bei LRP-Subplot-Visualisierung für {patch_id}: {e_lrp_viz_subplot}")
                                axes_combo[2].imshow(patch_for_viz_np) # Fallback
                                axes_combo[2].text(0.5, 0.5, 'LRP Viz Error', ha='center', va='center', transform=axes_combo[2].transAxes, color='red')


                            axes_combo[2].set_title(f"LRP Heatmap ({lrp_agg_method})", fontsize=9)
                            axes_combo[2].axis('off')

                            fig_combo.suptitle(title_prefix, fontsize=11)
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Platz für suptitle anpassen

                            class_visualization_dir = os.path.join(visualization_dir, target_class_hr_name)
                            os.makedirs(class_visualization_dir, exist_ok=True)
                            vis_save_path_combo = os.path.join(class_visualization_dir, f"{patch_id}_viz_combo_{lrp_agg_method}.png")
                            fig_combo.savefig(vis_save_path_combo)
                            plt.close(fig_combo)
                            print(f"    Kombinierte Visualisierung gespeichert: {vis_save_path_combo}")
                            
    print("\nXAI-Analyse für Verzeichnis abgeschlossen.")

if __name__ == "__main__":
    # Stelle sicher, dass alle Hilfsfunktionen und Klassen (ClassifierModel, get_transforms, etc.)
    # entweder hier definiert sind oder korrekt importiert werden können.
    # Die Fallback-Definitionen oben im Skript sind eine Möglichkeit, dies zu handhaben.
    main()