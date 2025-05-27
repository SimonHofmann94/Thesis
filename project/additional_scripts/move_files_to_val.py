# move_files_to_val.py
import os
import shutil
import argparse
from typing import List, Optional

# Mögliche Bild- und Annotationserweiterungen
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
ANNOTATION_EXTENSION = '.json'

def find_actual_filename(directory: str, base_name: str, extensions: List[str]) -> Optional[str]:
    """
    Findet den tatsächlichen Dateinamen im Verzeichnis für einen Basisnamen und mögliche Erweiterungen.
    Gibt den ersten gefundenen vollen Dateinamen zurück.
    """
    for ext in extensions:
        full_name = base_name + ext
        if os.path.exists(os.path.join(directory, full_name)):
            return full_name
        # Berücksichtige auch Großbuchstaben-Erweiterungen, falls vorhanden
        full_name_upper = base_name + ext.upper()
        if os.path.exists(os.path.join(directory, full_name_upper)):
            return full_name_upper
    return None

def move_files(
    image_base_names: List[str],
    source_train_dir: str,
    dest_val_dir: str
):
    """
    Verschiebt die angegebenen Bilder und ihre zugehörigen Annotationen
    vom Trainings- in das Validierungsverzeichnis.

    Args:
        image_base_names (List[str]): Liste der Basisnamen der Bilder (ohne Erweiterung).
        source_train_dir (str): Basispfad zum Trainingsverzeichnis (z.B. '../data/Severstal/train').
        dest_val_dir (str): Basispfad zum Ziel-Validierungsverzeichnis (z.B. '../data/Severstal/val').
    """

    # Definiere Quell- und Zielpfade für Bilder und Annotationen
    source_img_dir = os.path.join(source_train_dir, "img")
    source_ann_dir = os.path.join(source_train_dir, "ann")
    dest_img_dir = os.path.join(dest_val_dir, "img")
    dest_ann_dir = os.path.join(dest_val_dir, "ann")

    # Stelle sicher, dass die Zielverzeichnisse existieren
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_ann_dir, exist_ok=True)
    print(f"Ziel-Bildverzeichnis: {os.path.abspath(dest_img_dir)}")
    print(f"Ziel-Annotationsverzeichnis: {os.path.abspath(dest_ann_dir)}")

    moved_count = 0
    skipped_count = 0

    for base_name in image_base_names:
        print(f"\nVerarbeite Bild-Basisnamen: {base_name}")

        # 1. Finde und verschiebe das Bild
        actual_img_filename = find_actual_filename(source_img_dir, base_name, IMAGE_EXTENSIONS)
        
        if not actual_img_filename:
            print(f"  FEHLER: Bilddatei für '{base_name}' nicht in '{source_img_dir}' gefunden. Überspringe.")
            skipped_count += 1
            continue

        source_img_path = os.path.join(source_img_dir, actual_img_filename)
        dest_img_path = os.path.join(dest_img_dir, actual_img_filename)

        # 2. Finde und verschiebe die Annotation(en)
        #    Wir müssen beide gängigen Annotationstypen berücksichtigen:
        #    a) {base_name}.json
        #    b) {actual_img_filename}.json (z.B. image123.jpg.json)
        
        annotation_files_to_move = []
        
        # Typ a: {base_name}.json
        ann_file_type_a = base_name + ANNOTATION_EXTENSION
        source_ann_path_a = os.path.join(source_ann_dir, ann_file_type_a)
        if os.path.exists(source_ann_path_a):
            annotation_files_to_move.append((source_ann_path_a, os.path.join(dest_ann_dir, ann_file_type_a)))
            print(f"  Annotationsdatei Typ A gefunden: {ann_file_type_a}")

        # Typ b: {actual_img_filename}.json
        ann_file_type_b = actual_img_filename + ANNOTATION_EXTENSION
        source_ann_path_b = os.path.join(source_ann_dir, ann_file_type_b)
        if os.path.exists(source_ann_path_b) and source_ann_path_b not in [src for src, dst in annotation_files_to_move]: # Verhindere Doppelbewegung
            annotation_files_to_move.append((source_ann_path_b, os.path.join(dest_ann_dir, ann_file_type_b)))
            print(f"  Annotationsdatei Typ B gefunden: {ann_file_type_b}")

        if not annotation_files_to_move:
            print(f"  WARNUNG: Keine passende Annotationsdatei für '{base_name}' (Bild: {actual_img_filename}) in '{source_ann_dir}' gefunden.")
            # Hier könntest du entscheiden, ob du das Bild trotzdem verschieben möchtest oder nicht.
            # Fürs Erste: Wir verschieben das Bild nur, wenn auch eine Annotation da ist, um Konsistenz zu wahren.
            # Wenn du das Bild auch ohne Annotation verschieben willst, kommentiere die nächste Zeile aus und passe die Logik an.
            # print(f"  Überspringe Verschieben von Bild '{actual_img_filename}', da keine Annotation gefunden wurde.")
            # skipped_count += 1
            # continue # Entferne dies, wenn das Bild auch ohne Annotation verschoben werden soll
            pass # Lasse zu, dass das Bild auch ohne Annotation verschoben wird, aber die Warnung bleibt.


        # Verschiebe das Bild
        try:
            print(f"  Verschiebe Bild: '{source_img_path}' -> '{dest_img_path}'")
            shutil.move(source_img_path, dest_img_path)
        except Exception as e:
            print(f"  FEHLER beim Verschieben des Bildes '{actual_img_filename}': {e}")
            skipped_count +=1
            continue # Wenn das Bild nicht verschoben werden kann, macht es keinen Sinn, die Annotation zu verschieben.

        # Verschiebe die Annotation(en)
        for source_ann_path, dest_ann_path in annotation_files_to_move:
            try:
                print(f"  Verschiebe Annotation: '{source_ann_path}' -> '{dest_ann_path}'")
                shutil.move(source_ann_path, dest_ann_path)
            except Exception as e:
                print(f"  FEHLER beim Verschieben der Annotation '{os.path.basename(source_ann_path)}': {e}")
                # Hier könntest du überlegen, das bereits verschobene Bild zurückzuverschieben für Konsistenz.
                # Fürs Erste belassen wir es bei einer Fehlermeldung.

        moved_count += 1

    print(f"\n--- Zusammenfassung ---")
    print(f"{moved_count} Bild-Basisnamen erfolgreich (teilweise) verarbeitet und Dateien verschoben.")
    print(f"{skipped_count} Bild-Basisnamen konnten nicht (vollständig) verarbeitet werden (siehe Fehler oben).")


def main():
    parser = argparse.ArgumentParser(description="Verschiebt Bilder und ihre Annotationen vom Train- ins Val-Verzeichnis.")
    parser.add_argument("--image_names", required=True, nargs='+',
                        help="Liste der Basisnamen der Bilder (ohne Erweiterung), die verschoben werden sollen. Z.B. image1 image2")
    parser.add_argument("--train_dir", required=True,
                        help="Basispfad zum Quell-Trainingsverzeichnis (z.B. ../data/Severstal/train). Muss 'img' und 'ann' Unterordner enthalten.")
    parser.add_argument("--val_dir", required=True,
                        help="Basispfad zum Ziel-Validierungsverzeichnis (z.B. ../data/Severstal/val). 'img' und 'ann' Unterordner werden erstellt, falls nicht vorhanden.")
    
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.train_dir, "img")):
        parser.error(f"Quell-Bildverzeichnis nicht gefunden: {os.path.join(args.train_dir, 'img')}")
    if not os.path.isdir(os.path.join(args.train_dir, "ann")):
        parser.error(f"Quell-Annotationsverzeichnis nicht gefunden: {os.path.join(args.train_dir, 'ann')}")

    move_files(args.image_names, args.train_dir, args.val_dir)

if __name__ == "__main__":
    main()