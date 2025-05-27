# analyze_annotations.py
import os
import json
import argparse
from collections import Counter
from typing import Optional

# Defektklassen-Mapping wie in deinen anderen Skripten
DEFECT_CLASS_TO_IDX = {
    "defect_1": 1, "defect_2": 2, "defect_3": 3, "defect_4": 4
}
IDX_TO_DEFECT_CLASS = {v: k for k, v in DEFECT_CLASS_TO_IDX.items()}

def analyze_annotation_directory(ann_dir: str, img_dir: Optional[str] = None):
    """
    Analysiert JSON-Annotationsdateien in einem Verzeichnis und gibt Statistiken aus.

    Args:
        ann_dir (str): Pfad zum Verzeichnis mit den JSON-Annotationsdateien.
        img_dir (str, optional): Pfad zum Bildverzeichnis. Wenn angegeben, werden nur
                                 Annotationen geprüft, für die auch ein Bild existiert.
    """
    if not os.path.isdir(ann_dir):
        print(f"Fehler: Annotationsverzeichnis nicht gefunden: {ann_dir}")
        return

    print(f"Analysiere Annotationen im Verzeichnis: {ann_dir}")
    if img_dir and not os.path.isdir(img_dir):
        print(f"Warnung: Bildverzeichnis {img_dir} nicht gefunden. Prüfe alle Annotationen ohne Bildabgleich.")
        img_dir = None # Deaktiviere Bildabgleich

    json_files = sorted([f for f in os.listdir(ann_dir) if f.lower().endswith('.json')])
    
    total_json_files = len(json_files)
    files_with_any_objects = 0
    files_with_known_defect_objects = 0
    defect_class_counter = Counter() # Zählt Vorkommen von Defektklassen 1-4 über alle Dateien
    images_with_no_corresponding_annotation_in_loop = 0 # Nur relevant, wenn img_dir gegeben

    # Wenn img_dir gegeben ist, erstelle eine Liste der Bild-Basisnamen für den Abgleich
    image_basenames = set()
    if img_dir:
        for img_f in os.listdir(img_dir):
            if img_f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_basenames.add(os.path.splitext(img_f)[0])
                # Berücksichtige auch den Fall, dass die JSON-Datei die Bildendung enthält, z.B. "bild.jpg.json"
                image_basenames.add(img_f)


    print(f"Gefunden: {total_json_files} JSON-Dateien im Annotationsverzeichnis.")
    if img_dir:
        print(f"Gefunden: {len(image_basenames)} potenzielle Bild-Basisnamen für den Abgleich.")

    processed_files = 0
    for json_filename in json_files:
        processed_files += 1
        if processed_files % 500 == 0:
            print(f"  ... {processed_files}/{total_json_files} JSON-Dateien verarbeitet ...")

        # Abgleich mit Bildverzeichnis, falls angegeben
        if img_dir:
            # Versuche, einen passenden Bild-Basisnamen zu finden
            # Fall 1: annotation.json -> bild.jpg (ann_base = annotation, img_base = bild)
            # Fall 2: bild.jpg.json -> bild.jpg (ann_base = bild.jpg, img_base = bild.jpg)
            ann_basename_no_ext = os.path.splitext(json_filename)[0] # z.B. "image123" oder "image123.jpg"
            
            # Wir prüfen, ob 'ann_basename_no_ext' oder (wenn es '.jpg' enthält) 'ann_basename_no_ext' ohne das letzte '.jpg'
            # in unseren image_basenames ist.
            # Beispiel: json_filename = "foo.jpg.json" -> ann_basename_no_ext = "foo.jpg"
            #           json_filename = "foo.json"     -> ann_basename_no_ext = "foo"
            
            match_found = False
            if ann_basename_no_ext in image_basenames: # z.B. foo.jpg (ann) in {..., foo.jpg (img), ...}
                match_found = True
            else:
                # Vielleicht ist der JSON-Dateiname nur der Basisname ohne Bildendung
                # z.B. ann_basename_no_ext = "foo" und wir suchen nach "foo" in {..., "foo" (von foo.jpg), ...}
                # Diese Logik ist schon durch den Aufbau von image_basenames abgedeckt,
                # da wir dort os.path.splitext(img_f)[0] hinzugefügt haben.
                pass # Bereits abgedeckt

            if not match_found:
                images_with_no_corresponding_annotation_in_loop +=1 # Falsch benannt, sollte sein: Annotations without corresponding image
                # print(f"Info: Keine passende Bilddatei für Annotation {json_filename} im Bildverzeichnis gefunden. Überspringe.")
                continue
        
        file_path = os.path.join(ann_dir, json_filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            has_any_objects_in_file = False
            has_known_defect_in_file = False
            
            if "objects" in data and isinstance(data["objects"], list) and data["objects"]:
                files_with_any_objects += 1
                has_any_objects_in_file = True
                
                temp_defects_in_this_file = set() # Um Doppelzählungen pro Datei zu vermeiden
                for obj in data["objects"]:
                    if isinstance(obj, dict) and "classTitle" in obj:
                        class_title = obj["classTitle"]
                        if class_title in DEFECT_CLASS_TO_IDX:
                            has_known_defect_in_file = True
                            temp_defects_in_this_file.add(DEFECT_CLASS_TO_IDX[class_title])
                
                if has_known_defect_in_file:
                    files_with_known_defect_objects +=1
                
                for defect_idx in temp_defects_in_this_file:
                     defect_class_counter[defect_idx] += 1 # Zählt, in wie vielen Dateien jede Klasse vorkommt

        except json.JSONDecodeError:
            print(f"Warnung: {json_filename} ist kein valides JSON. Übersprungen.")
        except Exception as e:
            print(f"Warnung: Fehler beim Verarbeiten von {json_filename}: {e}. Übersprungen.")

    print("\n--- Analyseergebnis ---")
    print(f"Verarbeitete JSON-Dateien (die ggf. ein Bild im img_dir hatten): {processed_files - images_with_no_corresponding_annotation_in_loop if img_dir else total_json_files}")
    if img_dir:
        print(f"JSON-Dateien ohne korrespondierendes Bild im img_dir (übersprungen): {images_with_no_corresponding_annotation_in_loop}")
    print(f"Dateien mit mindestens einem Eintrag in der 'objects'-Liste: {files_with_any_objects}")
    print(f"Dateien mit mindestens einem bekannten Defektobjekt (defect_1 bis defect_4): {files_with_known_defect_objects}")
    
    print("\nAnzahl der Dateien, in denen jede Defektklasse vorkommt:")
    if not defect_class_counter:
        print("  Keine bekannten Defektklassen in den Annotationen gefunden.")
    else:
        for i in range(1, 5): # Für Defektklassen 1 bis 4
            class_name = IDX_TO_DEFECT_CLASS.get(i, f"Unbekannte Klasse {i}")
            count = defect_class_counter.get(i, 0)
            print(f"  {class_name} (Index {i}): {count} Dateien")
    print("-----------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Analysiert den Inhalt von Annotationsverzeichnissen.")
    parser.add_argument("--ann_dir", required=True, help="Pfad zum Annotationsverzeichnis.")
    parser.add_argument("--img_dir", default=None, help="Optional: Pfad zum zugehörigen Bildverzeichnis für den Abgleich.")
    
    args = parser.parse_args()
    analyze_annotation_directory(args.ann_dir, args.img_dir)

if __name__ == "__main__":
    main()