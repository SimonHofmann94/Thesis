# project/stitch_patch_masks.py

import os
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig, ListConfig
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2 # Für Resize, falls nötig

# --- stitch_masks_for_image und parse_patch_filename Funktionen von oben bleiben gleich ---
def stitch_masks_for_image(
    patch_masks_dict: Dict[Tuple[int, int], np.ndarray], 
    original_image_shape_hw: Tuple[int, int],
    patch_size_hw: Tuple[int, int],
    overlap_method: str = "max"
    ) -> np.ndarray:
    orig_h, orig_w = original_image_shape_hw
    patch_h, patch_w = patch_size_hw
    full_stitched_mask = np.zeros(original_image_shape_hw, dtype=np.uint8)

    # Zählkarte für Mittelung bei Überlappung (optional, falls 'average' verwendet wird)
    # count_map = np.zeros(original_image_shape_hw, dtype=np.float32)

    for (y_start, x_start), patch_mask_data in patch_masks_dict.items():
        if patch_mask_data.shape[0] != patch_h or patch_mask_data.shape[1] != patch_w:
            print(f"Warnung: Patch-Maske bei ({y_start},{x_start}) hat unerwartete Größe {patch_mask_data.shape} statt {patch_size_hw}. Überspringe.")
            continue

        y_end = min(y_start + patch_h, orig_h)
        x_end = min(x_start + patch_w, orig_w)
        patch_region_to_copy = patch_mask_data[:y_end-y_start, :x_end-x_start]

        if overlap_method == "max":
            current_region_in_full_mask = full_stitched_mask[y_start:y_end, x_start:x_end]
            full_stitched_mask[y_start:y_end, x_start:x_end] = np.maximum(current_region_in_full_mask, patch_region_to_copy)
        # elif overlap_method == "average":
            # Hier bräuchte man die accumulator_map und count_map Logik
            # full_stitched_mask[y_start:y_end, x_start:x_end] += patch_region_to_copy
            # count_map[y_start:y_end, x_start:x_end] += 1
        else: # Default zu overwrite oder Fehler
            full_stitched_mask[y_start:y_end, x_start:x_end] = patch_region_to_copy
            
    # if overlap_method == "average":
    #     full_stitched_mask = np.divide(full_stitched_mask, count_map, out=np.zeros_like(full_stitched_mask), where=count_map!=0).astype(np.uint8)
    #     # Ggf. hier nochmal thresholden, wenn das Ergebnis nicht binär sein soll
        
    return full_stitched_mask.astype(np.uint8)


def parse_patch_filename(filename: str, suffix_to_remove: str = "_mask.npy") -> Optional[Tuple[str, int, int]]:
    """
    Parst den Originalbild-Identifier und die x,y-Koordinaten aus einem Patch-Masken-Dateinamen.
    Annahme: Dateiname ist z.B. "ORIGINALBILDID_x<X_KOORD>_y<Y_KOORD>_mask.npy"
    """
    # Entferne zuerst den bekannten Suffix
    if filename.endswith(suffix_to_remove):
        base_part = filename[:-len(suffix_to_remove)]
    else: # Falls der Suffix anders ist oder fehlt
        base_part, _ = os.path.splitext(filename)


    # Regex, um ID, x und y zu extrahieren
    match = re.search(r"^(.*?)_x(\d+)_y(\d+)$", base_part)
    if match:
        original_id = match.group(1)
        x_coord = int(match.group(2))
        y_coord = int(match.group(3))
        return original_id, x_coord, y_coord
    else:
        print(f"Warnung: Konnte Patch-Koordinaten nicht aus '{filename}' (Basis: '{base_part}') parsen.")
        return None

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("Starte das Zusammensetzen (Stitching) von prozessierten Patch-Masken...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # --- Pfade und Parameter aus der Config ---
    xai_run_name = cfg.xai.get("run_name", "targeted_analysis") # Sollte dem Run entsprechen, dessen Ergebnisse verarbeitet wurden
    active_postproc_pipeline = cfg.postprocessing.get("active_pipeline_name", "default_pipeline")

    # Basispfad zu den prozessierten binären Patch-Masken (Output von postprocess_attributions.py)
    processed_patch_masks_base_dir = os.path.join(project_root, 
                                               cfg.output.masks_dir, 
                                               cfg.model.backbone,
                                               f"LRP_{xai_run_name}") # Konsistent mit xai.py Output-Struktur

    # Output-Verzeichnis für die zusammengesetzten Masken
    stitched_output_dir = os.path.join(project_root, cfg.output.masks_dir, "stitched_final",
                                        cfg.model.backbone, f"LRP_{xai_run_name}", active_postproc_pipeline)
    os.makedirs(stitched_output_dir, exist_ok=True)
    print(f"Zusammengesetzte Masken werden gespeichert in: {stitched_output_dir}")

    # Patch-Größe (muss mit der Größe der gespeicherten Patch-Masken übereinstimmen)
    patch_cfg = cfg.data.severstal.patch_processing
    patch_size_hw = tuple(patch_cfg.patch_size_hw)

    # Originalbild-Dimensionen
    original_image_height = cfg.data.get("original_image_height_for_stitching", 256)
    original_image_width = cfg.data.get("original_image_width_for_stitching", 1600)
    original_image_shape_hw = (original_image_height, original_image_width)
    
    # Suffix der zu ladenden Patch-Masken-Dateien (von postprocess_attributions.py)
    patch_mask_file_suffix = cfg.postprocessing.get("output_patch_mask_suffix", "_processed_mask.npy")


    # Iteriere durch die Klassen, für die Postprocessing durchgeführt wurde
    # Annahme: Postprocessing hat Unterordner pro Klasse erstellt
    for class_name in SeverstalDataset.DEFECT_CLASS_TO_IDX.keys(): # "defect_1", "defect_2", ...
        print(f"\n--- Stitche Masken für Klasse: {class_name} ---")
        
        # Pfad zu den prozessierten Patch-Masken für diese Klasse und Pipeline
        # z.B. .../LRP_targeted_analysis/defect_1/otsu_forest_clean/
        input_patch_masks_dir = os.path.join(processed_patch_masks_base_dir, class_name, active_postproc_pipeline)

        if not os.path.exists(input_patch_masks_dir):
            print(f"  Input-Verzeichnis für prozessierte Patch-Masken nicht gefunden: {input_patch_masks_dir}. Überspringe.")
            continue

        patches_by_original_image = defaultdict(dict)
        found_files_for_class = False
        for filename in os.listdir(input_patch_masks_dir):
            if filename.endswith(patch_mask_file_suffix):
                found_files_for_class = True
                parsed_info = parse_patch_filename(filename, suffix_to_remove=patch_mask_file_suffix)
                if parsed_info:
                    original_id, x, y = parsed_info
                    try:
                        if filename.endswith(".npy"):
                            patch_mask = np.load(os.path.join(input_patch_masks_dir, filename))
                        # Optional: Unterstützung für .png, falls postprocessing.py das speichert
                        # elif filename.endswith(".png"):
                        #     patch_mask_pil = Image.open(os.path.join(input_patch_masks_dir, filename)).convert('L')
                        #     patch_mask = (np.array(patch_mask_pil) > 127).astype(np.uint8) # Binarisieren
                        
                        # Stelle sicher, dass der Patch die erwartete Größe hat
                        if patch_mask.shape[0] != patch_size_hw[0] or patch_mask.shape[1] != patch_size_hw[1]:
                            print(f"    Warnung: Patch {filename} hat Größe {patch_mask.shape}, erwartet {patch_size_hw}. Resizing...")
                            patch_mask = cv2.resize(patch_mask, (patch_size_hw[1], patch_size_hw[0]), interpolation=cv2.INTER_NEAREST)
                        
                        patches_by_original_image[original_id][(y, x)] = patch_mask
                    except Exception as e:
                        print(f"    Fehler beim Laden der prozessierten Patch-Maske {filename}: {e}")
        
        if not found_files_for_class:
            print(f"  Keine prozessierten Patch-Masken mit Suffix '{patch_mask_file_suffix}' in {input_patch_masks_dir} gefunden.")
            continue
        if not patches_by_original_image:
            print(f"  Konnte keine Patch-Informationen aus Dateinamen parsen für Klasse {class_name}.")
            continue
            
        print(f"  {sum(len(p_dict) for p_dict in patches_by_original_image.values())} prozessierte Patch-Masken für {len(patches_by_original_image)} Originalbilder (Klasse: {class_name}) gefunden.")

        for original_id, coord_mask_dict in patches_by_original_image.items():
            if not coord_mask_dict: continue # Sollte nicht passieren, aber sicher ist sicher

            print(f"    Stitche {len(coord_mask_dict)} Patches für Originalbild: {original_id}, Klasse: {class_name}...")
            
            stitched_mask = stitch_masks_for_image(
                coord_mask_dict, 
                original_image_shape_hw, 
                patch_size_hw,
                overlap_method=cfg.postprocessing.get("stitch_overlap_method", "max")
            )

            # Speichere die zusammengesetzte Maske für diese Klasse
            class_stitched_output_dir = os.path.join(stitched_output_dir, class_name)
            os.makedirs(class_stitched_output_dir, exist_ok=True)
            
            output_filename_png = os.path.join(class_stitched_output_dir, f"{original_id}_stitched_mask.png")
            
            try:
                mask_to_save_pil = Image.fromarray((stitched_mask * 255).astype(np.uint8), mode='L')
                mask_to_save_pil.save(output_filename_png)
                print(f"      Zusammengesetzte Maske gespeichert: {output_filename_png}")
            except Exception as e:
                print(f"      Fehler beim Speichern der zusammengesetzten Maske für {original_id}, Klasse {class_name}: {e}")

    print("\nStitching-Prozess abgeschlossen.")

if __name__ == "__main__":
    main()