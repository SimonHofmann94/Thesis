# demo_xai_methods.py
"""
Demo-Script um die verschiedenen XAI-Methoden zu testen.
Zeigt die Verwendung der neuen modularen XAI-Struktur.
Erweitert um umfassende Vergleichsvisualisierungen aller XAI-Methoden.
"""

import os, cv2
import sys
import torch
import json
import glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw
import time

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.xai_methods import get_available_methods, create_xai_method
from project.models import ClassifierModel
try:
    from current.train_5_Classes_blackout import get_transforms
except ImportError:
    print("WARNUNG: get_transforms nicht verfügbar. Verwende Fallback.")
    from torchvision import transforms
    def get_transforms(cfg_augment_dict, img_h_config, img_w_config, is_train, for_strip_mode):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
from project.utils import create_full_mask_from_png_object

# Defect class mapping
DEFECT_CLASS_TO_IDX = {
    "Defect 1": 1,
    "Defect 2": 2, 
    "Defect 3": 3,
    "Defect 4": 4
}

def load_full_gt_mask(ann_path: str, img_height: int, img_width: int) -> Optional[np.ndarray]:
    """Lädt die Ground Truth Maske aus einer JSON-Annotation."""
    if not os.path.exists(ann_path):
        return None
    
    gt_mask = np.zeros((img_height, img_width), dtype=np.uint8)
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
                                b64_data, origin, img_height, img_width
                            )
                            if single_obj_mask_np is not None:
                                gt_mask[single_obj_mask_np == 1] = class_idx_for_mask
        
        return gt_mask
    except Exception as e:
        print(f"Fehler beim Laden der Annotation {ann_path}: {e}")
        return None

def demo_xai_methods():
    """Demonstriert die verfügbaren XAI-Methoden."""
    
    print("=== Demo: XAI-Methoden ===")
    print()
    
    # Zeige verfügbare Methoden
    available_methods = get_available_methods()
    print(f"Verfügbare XAI-Methoden: {available_methods}")
    print()
    
    # Erstelle ein einfaches Dummy-Modell für Tests
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = torch.nn.Linear(64, 5)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    model = DummyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Teste jede verfügbare Methode
    for method_name in available_methods:
        print(f"--- Teste {method_name} ---")
        try:
            xai_method = create_xai_method(method_name, model, device)
            print(f"✓ {method_name} erfolgreich erstellt: {xai_method.get_method_name()}")
            
            # Teste mit Dummy-Input
            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            
            try:
                attribution = xai_method.attribute(dummy_input, target=1)
                print(f"✓ Attribution erfolgreich berechnet. Shape: {attribution.shape}")
            except Exception as e:
                print(f"⚠ Fehler bei Attribution: {e}")
                
        except Exception as e:
            print(f"✗ Fehler beim Erstellen von {method_name}: {e}")
        
        print()

def run_eval_example():
    """Zeigt ein Beispiel für die Verwendung des eval_model_with_xai Scripts."""
    
    print("=== Beispiel: eval_model_with_xai Verwendung ===")
    print()
    
    # Beispiele für verschiedene XAI-Methoden
    base_command = "python project/eval_model_with_xai.py"
    
    examples = [
        {
            "method": "lrp",
            "description": "Layer-wise Relevance Propagation"
        },
        {
            "method": "gradcam++",
            "description": "GradCAM++ für visuelle Erklärungen"
        },
        {
            "method": "scorecam",
            "description": "ScoreCAM (gradientenfreie Methode)"
        },
        {
            "method": "xrai",
            "description": "XRAI für regionale Aggregation"
        },
        {
            "method": "integratedgradients",
            "description": "Integrated Gradients"
        },
        {
            "method": "occlusion",
            "description": "Occlusion-basierte Analyse"
        },
        {
            "method": "shap",
            "description": "SHAP (SHapley Additive exPlanations)"
        }
    ]
    
    print("Beispiel-Commands:")
    print()

    for example in examples:
        cmd = f"{base_command} \\"
        cmd += f"\n  --img_dir data/img \\"
        cmd += f"\n  --ann_dir data/ann \\"
        cmd += f"\n  --model_path weights/best_model_severstal_wide_strips.pt \\"
        cmd += f"\n  --output_dir outputs/xai_{example['method']} \\"
        cmd += f"\n  --xai_method {example['method']} \\"
        cmd += f"\n  --limit_images 5"
        
        print(f"# {example['description']}")
        print(cmd)
        print()

def compare_all_xai_methods(
    test_images_dir: str = "data/img",
    annotations_dir: str = "data/ann", 
    model_path: str = "weights/best_model_severstal_wide_strips.pt",
    output_dir: str = "outputs/xai_comparison",
    limit_images: Optional[int] = None,
    methods_to_test: Optional[List[str]] = None
):
    """
    Erstellt umfassende Vergleichsvisualisierungen aller XAI-Methoden.
    
    Args:
        test_images_dir: Pfad zu den Testbildern
        annotations_dir: Pfad zu den Annotationen  
        model_path: Pfad zum trainierten Modell
        output_dir: Ausgabeverzeichnis für Vergleichsbilder
        limit_images: Optional: Limitiere Anzahl der Bilder
        methods_to_test: Optional: Spezifische Methoden testen
    """
    
    # Konfiguration
    NUM_CLASSES = 5
    MODEL_BACKBONE = 'densenet121'
    STRIP_HEIGHT = 224
    CLASS_NAMES = ["No Defect", "Defect 1", "Defect 2", "Defect 3", "Defect 4"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔬 XAI-Methoden Vergleichsanalyse")
    print("=" * 60)
    
    # Erstelle Output-Verzeichnis
    os.makedirs(output_dir, exist_ok=True)
      # Lade Modell
    print("📦 Lade Modell...")
    model = ClassifierModel(backbone_name=MODEL_BACKBONE, pretrained=False, num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"✓ Modell erfolgreich geladen: {model_path}")
    except Exception as e:
        print(f"✗ Fehler beim Laden des Modells: {e}")
        return
    
    # Verfügbare XAI-Methoden (dummy ausschließen für Produktivanalyse)
    available_methods = get_available_methods()
    if methods_to_test is None:
        methods_to_test = [m for m in available_methods if m != "dummy"]  # dummy ausschließen
    else:
        methods_to_test = [m for m in methods_to_test if m in available_methods and m != "dummy"]
    
    print(f"📋 Verfügbare Methoden: {available_methods}")
    print(f"🎯 Zu testende Methoden: {methods_to_test}")
      # Finde Testbilder
    image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg"))
    if limit_images:
        image_paths = image_paths[:limit_images]
    
    print(f"📸 Gefundene Bilder: {len(image_paths)}")
    
    # Transformations
    cfg_augment_dict = {"enable": False, "verbose_transforms": False}
    transform = get_transforms(
        cfg_augment_dict=cfg_augment_dict,
        img_h_config=STRIP_HEIGHT,
        img_w_config=-1,  # Variable Breite
        is_train=False,
        for_strip_mode=True
    )
      # Verarbeite jedes Bild
    total_start_time = time.time()
    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f"\n🖼️ Verarbeite Bild {i+1}/{len(image_paths)}: {image_name}")
        image_start_time = time.time()
        
        # Lade und verarbeite Bild
        result = load_image_and_annotation(
            image_path, annotations_dir, transform, model, DEVICE, CLASS_NAMES
        )
        
        if result is None:
            continue
            
        image_pil, image_tensor, gt_mask, pred_class, pred_conf, pred_probs = result
        
        # Berechne XAI-Attributionen für alle Methoden
        attributions = {}
        computation_times = {}
        
        # Wähle Zielklasse (bevorzuge Prediction falls nicht "No Defect")
        target_class = pred_class if pred_class > 0 else 1
        
        print(f"   🎯 Zielklasse für XAI: {target_class} ({CLASS_NAMES[target_class]})")
        
        for method_name in methods_to_test:
            print(f"   🔍 Berechne {method_name}...")
            
            try:
                start_time = time.time()
                
                # Methodenspezifische Parameter
                kwargs = get_method_kwargs(method_name)
                xai_method = create_xai_method(method_name, model, DEVICE, **kwargs)
                
                # Berechne Attribution
                attribution = xai_method.attribute(image_tensor, target_class)

                # Prüfe ob Attribution None ist (SHAP Fehlerfall)
                if attribution is None:
                    print(f"      ⚠️ {method_name} übersprungen (Fehler aufgetreten)")
                    continue  # Überspringe diese Methode
                
                computation_times[method_name] = time.time() - start_time
                attributions[method_name] = attribution
                
                comp_time = computation_times[method_name]
                if comp_time >= 60:
                    time_str = f"{comp_time/60:.1f}min"
                else:
                    time_str = f"{comp_time:.2f}s"
                print(f"      ✓ Erfolgreich ({time_str})")
                
            except Exception as e:
                print(f"      ✗ Fehler: {str(e)[:50]}...")
                attributions[method_name] = None
                computation_times[method_name] = 0
        
        # Erstelle Vergleichsvisualisierung
        create_comprehensive_comparison(
            image_pil=image_pil,
            gt_mask=gt_mask,
            attributions=attributions,
            computation_times=computation_times,
            image_name=image_name,
            pred_class=pred_class,
            pred_conf=pred_conf,
            target_class=target_class,
            class_names=CLASS_NAMES,
            output_dir=output_dir
        )
        
        image_total_time = time.time() - image_start_time
        if image_total_time >= 60:
            time_str = f"{image_total_time/60:.1f}min"
        else:
            time_str = f"{image_total_time:.1f}s"
        print(f"   💾 Vergleichsbild gespeichert (Gesamt: {time_str})")
    
    total_time = time.time() - total_start_time
    if total_time >= 60:
        time_str = f"{total_time/60:.1f}min"
    else:
        time_str = f"{total_time:.1f}s"
    print(f"\n✅ Alle Bilder verarbeitet! Gesamtzeit: {time_str}")


def load_image_and_annotation(
    image_path: str, 
    annotations_dir: str, 
    transform, 
    model, 
    device,
    class_names: List[str]
) -> Optional[Tuple]:
    """Lädt Bild, Annotation und macht Modell-Vorhersage."""
    
    try:
        # Lade Bild
        image_pil = Image.open(image_path).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
          # Lade Ground Truth Maske
        image_name = os.path.basename(image_path)
        ann_path = os.path.join(annotations_dir, image_name + ".json")
        gt_mask = None
        if os.path.exists(ann_path):
            try:
                img_width, img_height = image_pil.size
                gt_mask = load_full_gt_mask(ann_path, img_height, img_width)
            except Exception as e:
                print(f"      ⚠ Konnte GT-Maske nicht laden: {e}")
        
        # Modell-Vorhersage
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_conf = probs[0, pred_class].item()
        
        return image_pil, image_tensor, gt_mask, pred_class, pred_conf, probs
        
    except Exception as e:
        print(f"      ✗ Fehler beim Laden: {e}")
        return None


def get_method_kwargs(method_name: str) -> Dict:
    """Gibt methodenspezifische Parameter zurück."""
    method_kwargs = {
        "rise": {"n_masks": 1000},
        "occlusion": {"sliding_window_shapes": (3, 15, 15), "strides": (1, 8, 8)},  # Strides korrigiert
        "felzenszwalb": {"scale": 50, "sigma": 0.8, "min_size": 20},
        "watershed": {"compactness": 0.01},
        "reciprocam": {"epsilon": 1e-7, "gamma": 2.0},
        "customxrai": {"n_segments": 100, "compactness": 10.0},
        "shap": {"masker_type": "partition", "n_samples": 100, "max_evals": 500}  # SHAP Parameter hinzugefügt
    }
    return method_kwargs.get(method_name, {})


def create_comprehensive_comparison(
    image_pil: Image.Image,
    gt_mask: Optional[np.ndarray],
    attributions: Dict[str, torch.Tensor],
    computation_times: Dict[str, float],
    image_name: str,
    pred_class: int,
    pred_conf: float,
    target_class: int,
    class_names: List[str],
    output_dir: str
):
    """Erstellt eine umfassende Vergleichsvisualisierung aller XAI-Methoden."""
    
    # Filtere erfolgreiche Attributionen
    valid_attributions = {k: v for k, v in attributions.items() if v is not None}
    
    # Berechne Grid-Layout
    n_methods = len(valid_attributions)
    n_base_images = 2 if gt_mask is not None else 1  # Original + GT Mask (optional)
    n_total = n_base_images + n_methods
    
    # Grid-Dimensionen (bevorzuge 4-5 Spalten)
    n_cols = min(5, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols
    
    # Erstelle Figure
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    # Titel
    title = f"{image_name}\nPrediction: {class_names[pred_class]} ({pred_conf:.3f}) | XAI Target: {class_names[target_class]}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plot_idx = 0
    
    # 1. Original Bild
    ax = fig.add_subplot(gs[plot_idx // n_cols, plot_idx % n_cols])
    ax.imshow(image_pil)
    ax.set_title("Original Image", fontweight='bold')
    ax.axis('off')
    plot_idx += 1
    
    # 2. Ground Truth Maske (falls vorhanden)
    if gt_mask is not None:
        ax = fig.add_subplot(gs[plot_idx // n_cols, plot_idx % n_cols])
        ax.imshow(gt_mask, cmap='hot', alpha=0.7)
        ax.imshow(image_pil, alpha=0.3)
        ax.set_title("Ground Truth", fontweight='bold')
        ax.axis('off')
        plot_idx += 1
    
    # 3. XAI-Attributionen
    for method_name, attribution in valid_attributions.items():
        ax = fig.add_subplot(gs[plot_idx // n_cols, plot_idx % n_cols])
        
        # Konvertiere Attribution zu Heatmap
        heatmap = attribution_to_heatmap(attribution)
        
        # Zeige Overlay
        ax.imshow(image_pil, alpha=0.5)
        im = ax.imshow(heatmap, cmap='jet', alpha=0.7)
          # Titel mit Computation Time
        comp_time = computation_times.get(method_name, 0)
        if comp_time >= 60:
            time_str = f"{comp_time/60:.1f}min"
        else:
            time_str = f"{comp_time:.2f}s"
        title = f"{method_name.upper()}\n({time_str})"
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')
        
        # Colorbar für jeden Plot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plot_idx += 1
    
    # Fülle leere Subplots
    while plot_idx < n_rows * n_cols:
        ax = fig.add_subplot(gs[plot_idx // n_cols, plot_idx % n_cols])
        ax.axis('off')
        plot_idx += 1
    
    # Speichere
    output_filename = f"{image_name.split('.')[0]}_xai_comparison.jpg"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', format='jpg')
    plt.close()


def attribution_to_heatmap(attribution: torch.Tensor) -> np.ndarray:
    """Konvertiert Attribution zu normalisierten Heatmap."""
    
    # Konvertiere zu numpy
    attr_np = attribution.detach().cpu().numpy()
    
    # Handle verschiedene Shapes
    if len(attr_np.shape) == 4:  # (N, C, H, W)
        attr_np = attr_np[0]  # Erstes Sample
    
    if len(attr_np.shape) == 3:  # (C, H, W)
        if attr_np.shape[0] == 3:  # RGB
            attr_np = np.mean(attr_np, axis=0)  # Mittelwert über Kanäle
        else:
            attr_np = attr_np[0]  # Erster Kanal
    
    # Normalisierung
    attr_min, attr_max = attr_np.min(), attr_np.max()
    if attr_max > attr_min:
        attr_np = (attr_np - attr_min) / (attr_max - attr_min)
    else:
        attr_np = np.zeros_like(attr_np)
    
    return attr_np

def main():
    """Hauptfunktion der Demo."""
    print("XAI-Methoden Demo & Vergleichsanalyse")
    print("=" * 60)
    print()
    
    # Menü für Benutzer
    print("Wählen Sie eine Option:")
    print("1. Demo der verfügbaren Methoden (schnell)")
    print("2. Beispiel-Commands anzeigen")
    print("3. Umfassende XAI-Vergleichsanalyse (alle Methoden)")
    print("4. XAI-Vergleichsanalyse (nur ausgewählte Methoden)")
    print()
    
    choice = input("Ihre Wahl (1-4): ").strip()
    
    if choice == "1":
        print("=" * 50)
        demo_xai_methods()
        
    elif choice == "2":
        print("=" * 50)
        run_eval_example()
        
    elif choice == "3":
        print("=" * 50)
        print("🚀 Starte umfassende XAI-Vergleichsanalyse...")
        
        # Parameter abfragen
        img_dir = input("Bildverzeichnis [data/img]: ").strip() or "data/img"
        ann_dir = input("Annotationsverzeichnis [data/ann]: ").strip() or "data/ann"
        model_path = input("Modellpfad [weights/best_model_severstal_wide_strips.pt]: ").strip() or "weights/best_model_severstal_wide_strips.pt"
        output_dir = input("Ausgabeverzeichnis [outputs/xai_comparison]: ").strip() or "outputs/xai_comparison"
        
        limit_str = input("Anzahl Bilder limitieren [alle]: ").strip()
        limit_images = int(limit_str) if limit_str.isdigit() else None
        
        compare_all_xai_methods(
            test_images_dir=img_dir,
            annotations_dir=ann_dir,
            model_path=model_path,
            output_dir=output_dir,
            limit_images=limit_images
        )
        
    elif choice == "4":
        print("=" * 50)
        print("🎯 Starte selektive XAI-Vergleichsanalyse...")
        
        # Zeige verfügbare Methoden
        available_methods = get_available_methods()
        print(f"\nVerfügbare Methoden: {', '.join(available_methods)}")
        
        methods_input = input("Methoden auswählen (kommagetrennt): ").strip()
        selected_methods = [m.strip() for m in methods_input.split(",") if m.strip()]
        
        if not selected_methods:
            print("❌ Keine Methoden ausgewählt!")
            return
        
        # Parameter abfragen
        img_dir = input("Bildverzeichnis [data/img]: ").strip() or "data/img"
        ann_dir = input("Annotationsverzeichnis [data/ann]: ").strip() or "data/ann"
        model_path = input("Modellpfad [weights/best_model_severstal_wide_strips.pt]: ").strip() or "weights/best_model_severstal_wide_strips.pt"
        output_dir = input("Ausgabeverzeichnis [outputs/xai_comparison_selected]: ").strip() or "outputs/xai_comparison_selected"
        
        limit_str = input("Anzahl Bilder limitieren [alle]: ").strip()
        limit_images = int(limit_str) if limit_str.isdigit() else None
        
        compare_all_xai_methods(
            test_images_dir=img_dir,
            annotations_dir=ann_dir,
            model_path=model_path,
            output_dir=output_dir,
            limit_images=limit_images,
            methods_to_test=selected_methods
        )
        
    else:
        print("❌ Ungültige Auswahl!")
        return
    
    print("\n" + "=" * 60)
    print("🎉 Fertig!")
    print("\nTipps:")
    print("- Verwenden Sie --limit_images für schnelle Tests")
    print("- GradCAM++ funktioniert am besten mit CNNs")
    print("- ScoreCAM ist langsamer, aber gradientenfrei")
    print("- XRAI kann bei großen Bildern länger dauern")
    print("- LRP ist besonders gut für tiefe Netzwerke")
    print("- ReciproCAM und GuidedGradCAM sind die neuesten Methoden")
    print("- Occlusion zeigt direkt welche Regionen wichtig sind")
    print("- SHAP bietet theoretisch fundierte Erklärungen")


if __name__ == "__main__":
    main()
