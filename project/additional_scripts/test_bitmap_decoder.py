import base64, os
import zlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError 
from typing import Optional, Tuple, List, Dict
import io 

# ------------------------------------------------------------------------------------
# DATEN VON EINEM BEISPIEL-JSON-OBJEKT (aus deinem Severstal-Datensatz)
# Ersetze diese Werte mit einem echten Beispiel aus einer deiner JSON-Dateien
# ------------------------------------------------------------------------------------
# Beispiel 1 (defect_3 aus deinem vorherigen Post)
EXAMPLE_BITMAP_DATA_BASE64 = "eJwBPgHB/olQTkcNChoKAAAADUlIRFIAAACiAAAAPgEDAAAAkphSYAAAAAZQTFRFAAAA////pdmf3QAAAAF0Uk5TAEDm2GYAAADmSURBVHicldTBEYQgDAXQOBw8UoLbCaVpaZRiCRw5MGTdoBjIvywXZ55Oko8okayQCKydC1SOVpnBw8uljPWc1f00E32sXiXG+VbR6ArQ7OugXrQGBspTlK7DdNujCWqGWv7QCpWxHlAjVB2jZxtivJqhFqiMNUJVQ6yvZqgFodofVVbtxKb1hNozB60FlX3S7SO2dtuErV2YNaMCrd2MrZ3VeH8opp2zmsb9etVbzSCEjIY1WK0gmqhcjwVouiPWe06pIFu3tFPTVZ0hajeuyWhewf5V5DUeQB1Uwhqg+nmqVhgqxS8tbXN+2ddjDgAAAABJRU5ErkJggiqZh3I="
EXAMPLE_ORIGIN_XY = [737, 39] # [x_koordinate_spalte, y_koordinate_zeile]
EXAMPLE_CLASS_TITLE = "defect_3"

# Du kannst hier weitere Beispiele hinzufügen und sie abwechselnd testen:
# EXAMPLE_BITMAP_DATA_BASE64 = "DEIN_ANDERER_BASE64_STRING"
# EXAMPLE_ORIGIN_XY = [DEIN_ANDERES_X, DEIN_ANDERES_Y]
# EXAMPLE_CLASS_TITLE = "DEINE_ANDERE_KLASSE"

# Gesamtbild-Dimensionen (wie deine Severstal-Bilder)
FULL_IMAGE_HEIGHT = 256
FULL_IMAGE_WIDTH = 1600

# Pfad zu einem Beispielbild (optional, nur für die Visualisierung im Hintergrund)
# Wenn du kein echtes Bild hast, erstellen wir ein leeres.
# ERSETZE DIESEN PFAD, WENN DU EIN PASSENDES BILD ZUM OBIGEN JSON HAST
# Z.B. das Bild, aus dem die obige Annotation stammt.
EXAMPLE_IMAGE_PATH = "../data/Severstal/train/img/b3ff13646.jpg" # z.B. "data/Severstal/train/img/DEIN_BEISPIELBILD.jpg"
# Wenn EXAMPLE_IMAGE_PATH = None ist, wird ein leeres graues Bild verwendet.
# ------------------------------------------------------------------------------------

def base64_zlib_to_pil_image(base64_str: str) -> Optional[Image.Image]:
    """
    Dekodiert einen Base64-String, dekomprimiert zlib und lädt das Ergebnis als PIL Image (PNG).
    """
    try:
        decoded_bytes = base64.b64decode(base64_str)
        decompressed_bytes = zlib.decompress(decoded_bytes)
        
        # Erstelle einen In-Memory Byte-Stream für PIL
        bytes_io = io.BytesIO(decompressed_bytes)
        pil_image = Image.open(bytes_io)
        return pil_image
    except UnidentifiedImageError:
        print("Fehler: Konnte die Daten nicht als bekanntes Bildformat (PNG) identifizieren.")
        # Gib die ersten Bytes aus, um zu sehen, ob es wirklich PNG ist
        # print(f"  Erste Bytes nach Dekompression (Hex): {' '.join(f'{b:02x}' for b in decompressed_bytes[:16])}")
        return None
    except Exception as e:
        print(f"Fehler beim Laden der dekomprimierten Bytes als PIL Image: {e}")
        return None

def create_full_mask_from_png_object(
    base64_data_str: str, 
    origin_xy: List[int], # [x_col, y_row]
    target_image_h: int, 
    target_image_w: int
) -> Optional[np.ndarray]:
    """
    Erstellt eine binäre Maske für das Gesamtbild aus einer PNG-Annotation.
    """
    print(f"\nVerarbeite PNG-Bitmap mit Origin: {origin_xy}")
    
    small_mask_pil = base64_zlib_to_pil_image(base64_data_str)

    if small_mask_pil is None:
        print("  Fehler: Konnte die kleine Maske nicht als PIL Image laden.")
        return None

    # Konvertiere das PIL Image der kleinen Maske in ein NumPy Array.
    # PNG-Masken sind oft im Modus 'P' (palettenbasiert) oder 'L' (Graustufen) oder '1' (binär).
    # Wir wollen eine binäre Maske (0 oder 1).
    # Wenn es 'RGBA' ist, nimm den Alpha-Kanal oder prüfe auf nicht-transparente Pixel.
    # Für typische Masken sollte ein einfacher Helligkeits-Threshold funktionieren.
    
    print(f"  PIL Image der kleinen Maske: Modus={small_mask_pil.mode}, Größe={small_mask_pil.size}")

    # Konvertiere zu Graustufen, dann zu NumPy Array und normalisiere zu 0 oder 1
    # Annahme: In der Masken-PNG sind Defektpixel nicht schwarz (0).
    # Wenn Defektpixel weiß (255) sind und Hintergrund schwarz (0):
    try:
        if small_mask_pil.mode == 'RGBA': # Falls es einen Alpha-Kanal gibt
            # Option 1: Nimm den Alpha-Kanal als Maske (wenn Alpha > 0 bedeutet maskiert)
            # small_mask_np_gray = np.array(small_mask_pil.split()[-1]) # Letzter Kanal ist Alpha
            # Option 2: Konvertiere zu RGB und dann zu Graustufen
            small_mask_pil = small_mask_pil.convert('L') # Konvertiere zu Graustufen
            small_mask_np_gray = np.array(small_mask_pil)

        elif small_mask_pil.mode == 'P' or small_mask_pil.mode == 'L':
            small_mask_np_gray = np.array(small_mask_pil.convert('L')) # Sicherstellen, dass es Graustufen ist
        elif small_mask_pil.mode == '1': # Bereits binär
             small_mask_np_gray = np.array(small_mask_pil) * 255 # Umwandeln 0/1 zu 0/255 für Konsistenz
        else:
            print(f"  Unbehandelter PIL Modus: {small_mask_pil.mode}. Versuche Konvertierung zu 'L'.")
            small_mask_np_gray = np.array(small_mask_pil.convert('L'))

        # Erstelle eine binäre Maske: Pixel > 0 (oder einem anderen Threshold) werden zu 1, sonst 0
        small_mask_2d_binary = (small_mask_np_gray > 0).astype(np.uint8) 
        h_bitmap, w_bitmap = small_mask_2d_binary.shape
        print(f"  Kleine binäre 2D-Maske (Shape {small_mask_2d_binary.shape}) erfolgreich erstellt.")

    except Exception as e:
        print(f"  Fehler bei der Konvertierung des PIL Images zu einer binären NumPy-Maske: {e}")
        return None

    if h_bitmap == 0 or w_bitmap == 0:
        print("  Fehler: Die dekodierte kleine Maske hat Dimensionen von Null.")
        return None

    # Erstelle die leere Gesamtmaske
    full_target_mask = np.zeros((target_image_h, target_image_w), dtype=np.uint8)
    
    x_origin_col, y_origin_row = origin_xy[0], origin_xy[1]

    y_start_target = y_origin_row
    y_end_target = min(y_origin_row + h_bitmap, target_image_h)
    x_start_target = x_origin_col
    x_end_target = min(x_origin_col + w_bitmap, target_image_w)

    h_to_copy_from_small = y_end_target - y_start_target
    w_to_copy_from_small = x_end_target - x_start_target

    if h_to_copy_from_small > 0 and w_to_copy_from_small > 0:
        full_target_mask[y_start_target:y_end_target, x_start_target:x_end_target] = \
            small_mask_2d_binary[:h_to_copy_from_small, :w_to_copy_from_small]
        print(f"  Maske platziert auf Zielmaske im Bereich y:[{y_start_target}:{y_end_target}], x:[{x_start_target}:{x_end_target}]")
    else:
        print("  Warnung: PNG-Bitmap liegt komplett außerhalb des Zielbildbereichs oder hat Größe 0 nach Clipping.")
        
    return full_target_mask

# --- Hauptteil des Testskripts ---
if __name__ == "__main__":
    print("Starte PNG-Bitmap-Dekodierungstest...")

    # --- KORREKTUR HIER: background_image_pil initialisieren ---
    background_image_pil = Image.new('RGB', (FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT), color='lightgray')
    print("Standard-Hintergrundbild (grau) initialisiert.")
    # --- ENDE KORREKTUR ---

    if EXAMPLE_IMAGE_PATH and os.path.exists(EXAMPLE_IMAGE_PATH):
        try:
            loaded_image = Image.open(EXAMPLE_IMAGE_PATH).convert("RGB")
            background_image_pil = loaded_image.resize((FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)) # Überschreibe Dummy
            print(f"Beispielbild '{EXAMPLE_IMAGE_PATH}' geladen und auf Zielgröße ({FULL_IMAGE_WIDTH}x{FULL_IMAGE_HEIGHT}) angepasst.")
        except Exception as e:
            print(f"Fehler beim Laden des Beispielbildes '{EXAMPLE_IMAGE_PATH}': {e}. Verwende graues Dummy-Bild.")
            # background_image_pil bleibt das graue Dummy-Bild
    else:
        if EXAMPLE_IMAGE_PATH:
             print(f"Beispielbild '{EXAMPLE_IMAGE_PATH}' nicht gefunden. Verwende graues Dummy-Bild.")
        else:
             print("Kein Beispielbildpfad angegeben. Verwende graues Dummy-Bild.")
    # print("Verwende Hintergrundbild für Visualisierung.") # Kann jetzt weg oder angepasst werden


    generated_mask = create_full_mask_from_png_object(
        EXAMPLE_BITMAP_DATA_BASE64,
        EXAMPLE_ORIGIN_XY,
        FULL_IMAGE_HEIGHT,
        FULL_IMAGE_WIDTH
    )

    if generated_mask is not None:
        print(f"\nGenerierte Maske hat Shape: {generated_mask.shape}, Datentyp: {generated_mask.dtype}")
        print(f"Anzahl der maskierten Pixel (Wert > 0): {np.sum(generated_mask > 0)}")

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        ax[0].imshow(background_image_pil)
        ax[0].set_title(f"Bild (oder Dummy) für {EXAMPLE_CLASS_TITLE}") # Titel angepasst
        ax[0].axis('off')

        ax[1].imshow(background_image_pil)
        ax[1].imshow(generated_mask, alpha=0.5, cmap='viridis') 
        ax[1].plot(EXAMPLE_ORIGIN_XY[0], EXAMPLE_ORIGIN_XY[1], 'ro', markersize=5) 
        ax[1].set_title(f"Bild mit generierter PNG-Maske (Origin: {EXAMPLE_ORIGIN_XY})")
        ax[1].axis('off')
        
        plt.tight_layout()
        
        plot_filename = "bitmap_test_visualization.png"
        plt.savefig(plot_filename)
        print(f"Visualisierung gespeichert als: {plot_filename}")
        plt.close(fig) 

    else:
        print("Konnte keine Maske generieren.")

    print("\nTestskript beendet.")
