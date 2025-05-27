import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from PIL import Image, UnidentifiedImageError 
import base64
import zlib
import io 

def set_seed(seed: int):
    """
    Setzt den Random Seed für Python, NumPy und Torch (inkl. CUDA) auf `seed` für Reproduzierbarkeit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Für deterministische CUDNN-Gewährleistung
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Focal Loss for multi-label classification.
        Args:
            alpha (Optional[torch.Tensor]): Weighting factor for each class. Tensor of shape (num_classes,).
                                             If None, no alpha weighting is applied.
            gamma (float): Focusing parameter. Higher values give more weight to hard examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid), shape (N, C).
            targets (torch.Tensor): Ground truth labels, shape (N, C), with 0s and 1s.
        Returns:
            torch.Tensor: Calculated focal loss.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        #pt = torch.exp(-BCE_loss) # Wahrscheinlichkeit des korrekten Labels
        # Stattdessen verwenden wir die direkte Berechnung von p_t
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets) # Wahrscheinlichkeit des Ground-Truth-Labels

        F_loss = BCE_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            if self.alpha.type() != inputs.type():
                self.alpha = self.alpha.type(inputs.type()) # Ensure same dtype
            # Alpha-Gewichtung anwenden: alpha für positive Klasse, (1-alpha) für negative Klasse
            # Für Multi-Label ist es oft einfacher, alpha direkt auf die positiven Klassen anzuwenden.
            # Hier verwenden wir eine übliche Form für Multi-Label:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha_t * F_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else: # 'none'
            return F_loss



class EarlyStopping:
    """Early stops the training if validation loss/metric doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, mode='min'): # mode hinzugefügt
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the quantity 
                        monitored has stopped decreasing; in 'max' mode it will stop when the 
                        quantity monitored has stopped increasing. Default: 'min'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min_max = np.Inf if mode == 'min' else -np.Inf # Anpassung hier
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode # mode speichern

    def step(self, val_metric, model=None):      # NEUE Version: umbenannt zu step
        # Der Inhalt der Methode bleibt genau gleich wie in meinem vorherigen Beispiel
        # für die __call__-Methode (mit der mode-Logik)

        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta: # Logik für Verbesserung
            self.counter += 1
            if self.verbose: # Nur ausgeben, wenn verbose=True
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0
        
        return self.early_stop


    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation metric improves.'''
        if model is not None and self.verbose: # Nur speichern und loggen, wenn Modell übergeben und verbose
            # Du könntest das Speichern des Modells auch außerhalb der EarlyStopping-Klasse handhaben
            # torch.save(model.state_dict(), self.path) 
            # Das Speichern passiert jetzt im train.py basierend auf best_f1_micro_val
            self.trace_func(f'Validation metric ({self.mode}imized) improved ({self.val_metric_min_max:.6f} --> {val_metric:.6f}).')
        self.val_metric_min_max = val_metric


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
    # print(f"\nVerarbeite PNG-Bitmap mit Origin: {origin_xy}")
    
    small_mask_pil = base64_zlib_to_pil_image(base64_data_str)

    if small_mask_pil is None:
        print("  Fehler: Konnte die kleine Maske nicht als PIL Image laden.")
        return None

    # Konvertiere das PIL Image der kleinen Maske in ein NumPy Array.
    # PNG-Masken sind oft im Modus 'P' (palettenbasiert) oder 'L' (Graustufen) oder '1' (binär).
    # Wir wollen eine binäre Maske (0 oder 1).
    # Wenn es 'RGBA' ist, nimm den Alpha-Kanal oder prüfe auf nicht-transparente Pixel.
    # Für typische Masken sollte ein einfacher Helligkeits-Threshold funktionieren.
    
    # print(f"  PIL Image der kleinen Maske: Modus={small_mask_pil.mode}, Größe={small_mask_pil.size}")

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
        # print(f"  Kleine binäre 2D-Maske (Shape {small_mask_2d_binary.shape}) erfolgreich erstellt.")

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
        # print(f"  Maske platziert auf Zielmaske im Bereich y:[{y_start_target}:{y_end_target}], x:[{x_start_target}:{x_end_target}]")
    else:
        print("  Warnung: PNG-Bitmap liegt komplett außerhalb des Zielbildbereichs oder hat Größe 0 nach Clipping.")
        
    return full_target_mask