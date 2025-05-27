# project/postprocess_attributions.py

import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, ListConfig # ListConfig hinzugefügt
from PIL import Image # Hinzugefügt für das Speichern als PNG

# --- Hilfsfunktionen für Postprocessing-Schritte (bleiben gleich wie zuvor) ---

def normalize_map(attr_map: np.ndarray) -> np.ndarray:
    min_val = np.min(attr_map)
    max_val = np.max(attr_map)
    if max_val - min_val > 1e-8:
        return (attr_map - min_val) / (max_val - min_val)
    return np.zeros_like(attr_map)

def threshold_fixed(attr_map: np.ndarray, threshold_value: float) -> np.ndarray:
    # Sicherstellen, dass der Input für cv2.threshold float32 ist, wenn er nicht schon uint8 ist
    map_for_thresh = attr_map
    if attr_map.dtype != np.uint8:
        map_for_thresh = attr_map.astype(np.float32)

    # Threshold auf 1 setzen, damit das Ergebnis direkt 0 oder 1 ist
    _, binary_mask = cv2.threshold(map_for_thresh, threshold_value, 1, cv2.THRESH_BINARY)
    return binary_mask.astype(np.uint8)

def threshold_percentile(attr_map: np.ndarray, percentile: float) -> np.ndarray:
    if attr_map.size == 0 : return np.zeros_like(attr_map, dtype=np.uint8) # Handle empty map
    threshold_value = np.percentile(attr_map, percentile)
    return threshold_fixed(attr_map, threshold_value)

def threshold_otsu(attr_map: np.ndarray) -> np.ndarray:
    if attr_map.size == 0 or np.all(attr_map == attr_map.flat[0]): # Handle empty or flat map
        # print("Warnung: Otsu nicht anwendbar auf leere oder flache Karte. Gebe leere Maske zurück.")
        return np.zeros_like(attr_map, dtype=np.uint8)
    norm_map_uint8 = (normalize_map(attr_map) * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(norm_map_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask.astype(np.uint8)

def threshold_gmm(attr_map: np.ndarray, n_components: int = 2, positive_component_threshold: float = 0.7) -> np.ndarray:
    if attr_map.ndim > 1:
        pixels = attr_map.reshape(-1, 1)
    else:
        pixels = attr_map.reshape(-1,1)

    if len(pixels) < n_components or len(np.unique(pixels)) < n_components : # Auch prüfen, ob genug unique values da sind
        # print("Warnung: Zu wenig Datenpunkte oder unique values für GMM-Fit. Gebe leere Maske zurück.")
        return np.zeros_like(attr_map, dtype=np.uint8)
        
    gmm = GaussianMixture(n_components=n_components, random_state=0, covariance_type='spherical') # covariance_type kann helfen
    try:
        gmm.fit(pixels)
    except ValueError as e:
        # print(f"Warnung: GMM fit fehlgeschlagen: {e}. Gebe leere Maske zurück.")
        return np.zeros_like(attr_map, dtype=np.uint8)

    means = gmm.means_.flatten()
    # Wähle die Komponente mit dem höheren Mittelwert (oder eine andere Logik, falls nötig)
    if len(means) < n_components: # Sollte nicht passieren, wenn fit erfolgreich war
        return np.zeros_like(attr_map, dtype=np.uint8)
    defect_component_idx = np.argmax(means)
    
    probs = gmm.predict_proba(pixels)
    if probs.shape[1] <= defect_component_idx: # Sicherstellen, dass der Index gültig ist
        return np.zeros_like(attr_map, dtype=np.uint8)

    defect_probs = probs[:, defect_component_idx]
    binary_mask_flat = (defect_probs > positive_component_threshold).astype(np.uint8)
    return binary_mask_flat.reshape(attr_map.shape)

def apply_morphology(binary_mask: np.ndarray, operations: ListConfig) -> np.ndarray: #... (angepasst für ListConfig)
    mask = binary_mask.copy()
    if not isinstance(operations, (list, ListConfig)): return mask
    for op_config_node in operations:
        op_config = dict(op_config_node) # Konvertiere zu Dict
        op_type = op_config.get("type")
        kernel_shape_str = op_config.get("kernel_shape", "rect").lower()
        ks_raw = op_config.get("kernel_size", [3,3])
        kernel_size = tuple(ks_raw) if isinstance(ks_raw, (list, ListConfig)) else (int(ks_raw), int(ks_raw))
        iterations = op_config.get("iterations", 1)

        if kernel_shape_str == "rect": kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        elif kernel_shape_str == "ellipse": kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        elif kernel_shape_str == "cross": kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        else: kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        if op_type == "open": mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif op_type == "close": mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif op_type == "erode": mask = cv2.erode(mask, kernel, iterations=iterations)
        elif op_type == "dilate": mask = cv2.dilate(mask, kernel, iterations=iterations) # Korrigiert zu cv2.dilate
        elif op_type == "area_open":
            min_area = op_config.get("min_area", 50)
            if np.sum(mask) == 0: continue # Nichts zu tun bei leerer Maske
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            new_mask = np.zeros_like(mask)
            for label_idx in range(1, num_labels):
                if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
                    new_mask[labels_im == label_idx] = 1
            mask = new_mask
        else: print(f"Warnung: Unbekannte morphologische Operation '{op_type}'.")
    return mask


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("Starte Postprocessing von Attributionskarten (Batch-Modus)...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # --- Parameter aus Config holen ---
    source_xai_run_name = cfg.postprocessing.get("source_xai_run_name", "targeted_analysis_v1")
    source_lrp_aggregation = cfg.postprocessing.get("source_lrp_aggregation", "sum_positive")
    # Klassen, für die das Postprocessing durchgeführt werden soll
    classes_to_process = list(cfg.postprocessing.get("classes_to_process", ["defect_1", "defect_3"]))
    # Welche Postprocessing-Pipeline soll angewendet werden?
    active_pipeline_name = cfg.postprocessing.get("active_pipeline_name", "default_pipeline")
    pipeline_config_node = cfg.postprocessing.pipelines.get(active_pipeline_name)
    
    limit_patches_to_process_per_class = cfg.postprocessing.get("limit_patches_per_class", None) # Optional: Limitiere Anzahl für Test

    if not pipeline_config_node:
        print(f"FEHLER: Pipeline '{active_pipeline_name}' nicht in config/postprocessing/default.yaml gefunden.")
        return
    
    pipeline_config = [] # Konvertiere zu Python Liste von Dicts
    if isinstance(pipeline_config_node, ListConfig):
        for item in pipeline_config_node: pipeline_config.append(dict(item))
    else: pipeline_config = pipeline_config_node


    print(f"Verwende Postprocessing-Pipeline: '{active_pipeline_name}'")
    print(f"Verarbeite Klassen: {classes_to_process}")
    print(f"Lade aggregierte Attributionskarten (Methode: '{source_lrp_aggregation}') aus XAI-Lauf: '{source_xai_run_name}'")

    # Basis-Input-Verzeichnis für aggregierte .npy-Attributionskarten
    base_input_attr_dir = os.path.join(project_root, cfg.output.attributions_dir,
                                       cfg.model.backbone,
                                       f"LRP_{source_xai_run_name}") # Annahme: XAI hat so gespeichert

    # Basis-Output-Verzeichnis für die prozessierten Patch-Masken
    # Diese werden dann von stitch_patch_masks.py verwendet
    base_output_processed_masks_dir = os.path.join(project_root, cfg.output.masks_dir,
                                                   cfg.model.backbone,
                                                   f"LRP_{source_xai_run_name}", # Wichtig: Konsistenter Name
                                                   active_pipeline_name) # Unterordner für diese Pipeline
    os.makedirs(base_output_processed_masks_dir, exist_ok=True)
    print(f"Prozessierte Patch-Masken werden gespeichert in Unterordnern von: {base_output_processed_masks_dir}")

    for class_name in classes_to_process:
        print(f"\n--- Verarbeite Klasse: {class_name} ---")
        
        class_input_dir = os.path.join(base_input_attr_dir, class_name, f"npy_agg_{source_lrp_aggregation}")
        class_output_dir = os.path.join(base_output_processed_masks_dir, class_name) # Eigener Ordner pro Klasse
        os.makedirs(class_output_dir, exist_ok=True)

        if not os.path.exists(class_input_dir):
            print(f"  Warnung: Input-Verzeichnis für Klasse {class_name} nicht gefunden: {class_input_dir}. Überspringe.")
            continue

        patch_counter_for_class = 0
        for npy_filename in os.listdir(class_input_dir):
            if npy_filename.endswith("_attr_agg.npy"): # Nur die aggregierten Karten
                if limit_patches_to_process_per_class is not None and \
                   patch_counter_for_class >= limit_patches_to_process_per_class:
                    print(f"  Limit von {limit_patches_to_process_per_class} Patches für Klasse {class_name} erreicht. Stoppe für diese Klasse.")
                    break

                input_npy_path = os.path.join(class_input_dir, npy_filename)
                patch_base_name = npy_filename.replace("_attr_agg.npy", "") # z.B. "0002cc93b_x896_y0"
                
                print(f"  Verarbeite Patch-Attribution: {patch_base_name}")
                aggregated_attr_map = np.load(input_npy_path)
                current_map = aggregated_attr_map.copy()

                # Visualisierung der Schritte für diesen Patch (optional)
                if cfg.postprocessing.get("save_step_visualizations", False) and patch_counter_for_class < cfg.postprocessing.get("visualize_steps_for_first_n_patches", 2):
                    num_steps = len(pipeline_config)
                    fig, axes = plt.subplots(1, num_steps + 1, figsize=(5 * (num_steps + 1), 4))
                    if not isinstance(axes, np.ndarray): axes = np.array([axes])
                    axes[0].imshow(current_map, cmap='viridis'); axes[0].set_title(f"Orig Attr\n{patch_base_name[:15]}..."); axes[0].axis('off')
                    fig_was_created = True
                else:
                    fig_was_created = False

                step_idx = 1
                for step_cfg_node in pipeline_config:
                    step_cfg = dict(step_cfg_node)
                    step_type = step_cfg.get("type")
                    params = step_cfg.get("params", {})
                    if isinstance(params, DictConfig): params = dict(params)
                    if "operations" in params and isinstance(params["operations"], ListConfig):
                        params["operations"] = [dict(op) for op in params["operations"]]
                    
                    # print(f"    Schritt {step_idx}: {step_type}") # Kann sehr verbose werden
                    if step_type == "normalize_to_01": current_map = normalize_map(current_map)
                    elif step_type == "threshold_fixed": current_map = threshold_fixed(current_map, **params)
                    elif step_type == "threshold_percentile": current_map = threshold_percentile(current_map, **params)
                    elif step_type == "threshold_otsu": current_map = threshold_otsu(current_map)
                    elif step_type == "threshold_gmm": current_map = threshold_gmm(current_map, **params)
                    elif step_type == "morphology": current_map = apply_morphology(current_map, params.get("operations", []))
                    else: print(f"    Warnung: Unbekannter Schritt '{step_type}'.")
                    
                    if fig_was_created and len(axes) > step_idx:
                        is_binary = current_map.ndim == 2 and len(np.unique(current_map)) <= 2
                        axes[step_idx].imshow(current_map, cmap='gray' if is_binary else 'viridis')
                        axes[step_idx].set_title(f"Nach: {step_type[:10]}"); axes[step_idx].axis('off')
                    step_idx += 1
                
                final_patch_mask = current_map.astype(np.uint8) # Sicherstellen, dass es uint8 ist

                # Speichere die finale binäre Patch-Maske
                output_npy_filename = f"{patch_base_name}_processed_mask.npy" # Konsistenter Name für Stitching
                np.save(os.path.join(class_output_dir, output_npy_filename), final_patch_mask)
                
                # Optional: Speichere auch als PNG
                if cfg.postprocessing.get("save_processed_patch_as_png", True):
                    final_mask_pil = Image.fromarray(final_patch_mask * 255, mode='L')
                    final_mask_pil.save(os.path.join(class_output_dir, f"{patch_base_name}_processed_mask.png"))

                if fig_was_created:
                    plt.tight_layout()
                    plt.savefig(os.path.join(class_output_dir, f"{patch_base_name}_steps.png"))
                    plt.close(fig)
                patch_counter_for_class += 1

        print(f"  Klasse {class_name} abgeschlossen. {patch_counter_for_class} Patch-Masken prozessiert und gespeichert in {class_output_dir}")

    print("\nBatch-Postprocessing abgeschlossen.")

if __name__ == "__main__":
    main()