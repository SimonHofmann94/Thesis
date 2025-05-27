# find_multiclass_examples.py
import os
import json
import argparse
from typing import List, Set

# Annahme: utils.py ist für create_full_mask_from_png_object nicht zwingend nötig,
# da wir hier nur die Klassentitel aus dem JSON lesen.
# Aber wenn du später die Masken dieser Bilder auch direkt verarbeiten wolltest,
# wäre es gut, die utils-Funktionen verfügbar zu haben.

# Defektklassen-Mapping wie in SeverstalDataset oder anderen Skripten
DEFECT_CLASS_TO_IDX = {
    "defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4
}
IDX_TO_DEFECT_CLASS = {v: k for k, v in DEFECT_CLASS_TO_IDX.items()}


def find_images_with_specific_defects(
    img_dir: str,
    ann_dir: str,
    required_defect_indices: List[int], # z.B. [1, 3]
    num_examples_to_find: int,
    match_logic: str = 'AND', # 'AND': alle müssen da sein, 'OR': mindestens eine
    exclusive: bool = False   # True: NUR diese Klassen dürfen da sein, False: auch andere sind okay
) -> List[dict]:
    """
    Durchsucht Bild- und Annotationsverzeichnisse nach Bildern, die spezifische Defektklassen enthalten.

    Args:
        img_dir: Pfad zum Bildverzeichnis.
        ann_dir: Pfad zum Annotationsverzeichnis.
        required_defect_indices: Liste der Indizes der Defektklassen (1-4), die vorhanden sein sollen.
        num_examples_to_find: Maximale Anzahl der zu findenden Bildbeispiele.
        match_logic: 'AND' (alle required_defect_indices müssen vorhanden sein) oder
                     'OR' (mindestens einer der required_defect_indices muss vorhanden sein).
        exclusive: Wenn True, darf das Bild NUR die in required_defect_indices genannten Klassen enthalten
                   (und keine anderen Defektklassen 1-4). Wenn False, darf es auch andere enthalten.

    Returns:
        Liste von Dictionaries, wobei jedes Dict Infos zu einem gefundenen Bild enthält.
    """
    found_images_info = []
    
    if not os.path.isdir(img_dir):
        print(f"Fehler: Bildverzeichnis nicht gefunden: {img_dir}")
        return found_images_info
    if not os.path.isdir(ann_dir):
        print(f"Fehler: Annotationsverzeichnis nicht gefunden: {ann_dir}")
        return found_images_info

    required_set = set(required_defect_indices)
    if not required_set:
        print("Fehler: Keine Defekt-Indizes zum Suchen angegeben.")
        return found_images_info
    
    print(f"Suche nach {num_examples_to_find} Bildern mit Defekt-Indizes: {required_defect_indices} "
          f"(Logik: {match_logic}, Exklusiv: {exclusive})")

    all_image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    processed_images_count = 0
    for img_filename in all_image_files:
        if len(found_images_info) >= num_examples_to_find:
            break # Genug Beispiele gefunden

        processed_images_count +=1
        if processed_images_count % 200 == 0:
            print(f"  ... {processed_images_count} Bilder geprüft ... {len(found_images_info)} passende gefunden ...")

        base_name_no_ext = os.path.splitext(img_filename)[0]
        ann_path_variants = [
            os.path.join(ann_dir, f"{img_filename}.json"),
            os.path.join(ann_dir, f"{base_name_no_ext}.json")
        ]
        ann_path = None
        for p_variant in ann_path_variants:
            if os.path.exists(p_variant):
                ann_path = p_variant
                break
        
        if not ann_path:
            # print(f"Info: Keine Annotation für {img_filename} gefunden.")
            continue

        try:
            with open(ann_path, 'r') as f:
                annotation_data = json.load(f)
            
            present_defect_indices_in_image: Set[int] = set()
            if "objects" in annotation_data and annotation_data["objects"]:
                for obj in annotation_data["objects"]:
                    class_title = obj.get("classTitle")
                    if class_title in DEFECT_CLASS_TO_IDX:
                        present_defect_indices_in_image.add(DEFECT_CLASS_TO_IDX[class_title])
            
            if not present_defect_indices_in_image and required_set : # Bild hat keine Defekte, aber wir suchen welche
                continue

            # Überprüfe Übereinstimmung basierend auf Logik
            match = False
            if match_logic == 'AND':
                if required_set.issubset(present_defect_indices_in_image): # Alle benötigten sind vorhanden
                    match = True
                    if exclusive: # Wenn exklusiv, dürfen keine *anderen* Defekte (1-4) da sein
                        # other_defects = present_defect_indices_in_image - required_set
                        # Gültige Defekte sind nur die in required_set
                        if not present_defect_indices_in_image.issubset(required_set): # D.h. es gibt Elemente in present, die nicht in required sind
                             match = False
            elif match_logic == 'OR':
                if not required_set.isdisjoint(present_defect_indices_in_image): # Mindestens ein gemeinsames Element
                    match = True
                    if exclusive: # Wenn exklusiv, dürfen NUR Defekte aus required_set vorhanden sein
                        # Alle im Bild vorhandenen Defekte müssen eine Teilmenge der gesuchten sein
                        if not present_defect_indices_in_image.issubset(required_set):
                            match = False
            
            if match:
                found_images_info.append({
                    "filename": img_filename,
                    "path": os.path.join(img_dir, img_filename),
                    "ann_path": ann_path,
                    "present_defect_indices": sorted(list(present_defect_indices_in_image)),
                    "present_defect_names": sorted([IDX_TO_DEFECT_CLASS.get(idx, f"UnknownIdx{idx}") for idx in present_defect_indices_in_image])
                })

        except json.JSONDecodeError:
            print(f"Warnung: Annotationsdatei {ann_path} ist kein valides JSON. Überspringe.")
        except Exception as e:
            print(f"Warnung: Unerwarteter Fehler beim Verarbeiten von {ann_path}: {e}. Überspringe.")
            
    print(f"Suche abgeschlossen. {len(found_images_info)} passende Bilder gefunden (max. {num_examples_to_find} gesucht).")
    return found_images_info


def main():
    parser = argparse.ArgumentParser(description="Findet Bilder mit spezifischen Defektklassen-Kombinationen.")
    parser.add_argument("--img_dir", required=True, help="Pfad zum Bildverzeichnis.")
    parser.add_argument("--ann_dir", required=True, help="Pfad zum Annotationsverzeichnis.")
    parser.add_argument("--defect_indices", required=True, type=int, nargs='+',
                        help="Liste der gewünschten Defekt-Indizes (1-4), z.B. --defect_indices 1 3")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Anzahl der zu findenden Beispiele (Standard: 3).")
    parser.add_argument("--logic", choices=['AND', 'OR'], default='AND',
                        help="Logik für Defektklassen: 'AND' (alle müssen da sein), 'OR' (mindestens eine) (Standard: AND).")
    parser.add_argument("--exclusive", action='store_true',
                        help="Wenn gesetzt, dürfen NUR die angegebenen Defektklassen im Bild sein (und keine anderen der Klassen 1-4).")

    args = parser.parse_args()

    # Überprüfe Gültigkeit der Defekt-Indizes
    for idx in args.defect_indices:
        if not (1 <= idx <= 4):
            parser.error(f"Ungültiger Defekt-Index: {idx}. Muss zwischen 1 und 4 liegen.")
            return
            
    found_images = find_images_with_specific_defects(
        args.img_dir,
        args.ann_dir,
        args.defect_indices,
        args.num_examples,
        args.logic,
        args.exclusive
    )

    if found_images:
        print("\nGefundene Bilder:")
        for info in found_images:
            print(f"  - Datei: {info['filename']}")
            print(f"    Pfad: {info['path']}")
            # print(f"    Ann Pfad: {info['ann_path']}") # Optional
            print(f"    Gefundene Defekt-Namen im Bild: {info['present_defect_names']}")
            print(f"    Gefundene Defekt-Indizes im Bild: {info['present_defect_indices']}")
            print("-" * 20)
    else:
        print("Keine Bilder gefunden, die den Kriterien entsprechen.")

if __name__ == "__main__":
    main()