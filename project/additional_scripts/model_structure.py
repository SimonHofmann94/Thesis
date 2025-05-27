# visualize_model_architecture.py
import torch
import torch.nn as nn
from torchvision.models import ( # Direkt aus torchvision.models importieren
    VGG11_Weights, ResNet18_Weights, EfficientNet_B0_Weights, DenseNet121_Weights
)
import torchvision.models as tv_models # Alias für torchvision.models

# --- Deine ClassifierModel Definition ---
# (Ich kopiere sie hier hinein, damit das Skript eigenständig ist.
# Alternativ: Importiere sie aus deiner models.py Datei)
class ClassifierModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'densenet121', # Standard auf densenet121 gesetzt
        pretrained: bool = True, # Für die Architektur-Visualisierung ist pretrained nicht entscheidend
        num_classes: int = 5,   # Standardmäßig 5 Klassen
    ):
        super().__init__()
        if backbone_name == 'vgg11':
            weights = VGG11_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.vgg11(weights=weights)
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            in_features = backbone.classifier[-1].in_features
            classifier_layers = list(backbone.classifier.children())[:-1]
            classifier_layers.append(nn.Linear(in_features, num_classes))
            self.classifier = nn.Sequential(*classifier_layers)
        elif backbone_name in ['resnet18', 'resnet50']:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None # Beispiel für ResNet18
            if backbone_name == 'resnet50': # Passende Weights für ResNet50, falls verwendet
                 weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = getattr(tv_models, backbone_name)(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.features = backbone
            self.avgpool = nn.Identity() # Bei ResNet wird oft AdaptiveAvgPool2d am Ende des Feature Extractors verwendet
            self.classifier = nn.Linear(in_features, num_classes)
        elif backbone_name.startswith('efficientnet'):
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None # Beispiel für B0
            # Für andere EfficientNets ggf. anpassen
            backbone = getattr(tv_models, backbone_name)(weights=weights)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            self.features = backbone
            self.avgpool = nn.Identity() # EfficientNet hat oft ein AdaptiveAvgPool2d eingebaut
            self.classifier = nn.Linear(in_features, num_classes)
        elif backbone_name == 'densenet121':
            weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.densenet121(weights=weights)
            in_features = backbone.classifier.in_features
            # Wichtig: Das ursprüngliche Classifier-Layer von DenseNet entfernen oder ersetzen
            backbone.classifier = nn.Identity() # Entfernt das ursprüngliche fc-Layer
            self.features = backbone.features # DenseNet's features sind der Hauptteil
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Standard-Pooling nach DenseNet Features
            self.classifier = nn.Linear(in_features, num_classes) # Dein eigener Classifier
        else:
            raise ValueError(f"Unbekannter Backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x
# --- Ende der ClassifierModel Definition ---

def main():
    # --- Modellparameter (entsprechend deinem trainierten Modell) ---
    model_backbone = 'densenet121'
    num_classes = 5  # Severstal: NoDefect + 4 Defect Classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Beispiel-Input-Größe (entsprechend deiner Patch-Größe) ---
    # (Batch_size, Channels, Height, Width)
    batch_size = 1 # Für die Visualisierung reicht ein Batch von 1
    input_channels = 3
    patch_h, patch_w = 256, 256
    example_input_size_torchinfo = (batch_size, input_channels, patch_h, patch_w)
    example_input_tensor_torchviz = torch.randn(example_input_size_torchinfo).to(device)


    # 1. Modell instanziieren
    # Für die reine Architekturanzeige ist `pretrained=False` okay, da wir keine Gewichte laden.
    # Wenn du ein spezifisches, trainiertes Modell visualisieren willst, das geladen wird, ist das auch okay.
    model = ClassifierModel(backbone_name=model_backbone, pretrained=False, num_classes=num_classes)
    model.to(device)
    model.eval() # Wichtig, falls Dropout etc. vorhanden sind

    print(f"Modell: {model_backbone} mit {num_classes} Klassen auf Device: {device}\n")

    # 2. `print(model)` - Einfache Textausgabe
    print("--- Modellarchitektur (print(model)) ---")
    print(model)
    print("-" * 50 + "\n")

    # 3. `torchinfo` für eine detaillierte Zusammenfassung
    try:
        from torchinfo import summary
        print("--- Modellzusammenfassung (torchinfo) ---")
        # col_names: welche Spalten angezeigt werden sollen
        # depth: wie tief in verschachtelte Module (Sequential, ModuleList) eingedrungen wird
        summary(model, input_size=example_input_size_torchinfo, device=str(device),
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                depth=5) # Experimentiere mit dem depth-Wert für DenseNet
        print("-" * 50 + "\n")
    except ImportError:
        print("torchinfo ist nicht installiert. Überspringe torchinfo-Zusammenfassung.")
        print("Installiere mit: pip install torchinfo")
    except Exception as e:
        print(f"Fehler bei der torchinfo-Zusammenfassung: {e}")


    # 4. `torchviz` für eine grafische Darstellung des Berechnungsgraphen
    try:
        from torchviz import make_dot

        # Führe einen Forward-Pass mit dem Beispiel-Input durch
        output_tensor = model(example_input_tensor_torchviz)

        # Erstelle den Graphen
        # params=dict(model.named_parameters()) fügt Parameter-Knoten hinzu (optional, macht den Graphen größer)
        dot_graph = make_dot(output_tensor, params=None) # Ohne Parameter für einen saubereren Graphen
        
        # Speichere den Graphen als PDF (oder .png, etc.)
        output_graph_filename = f"{model_backbone}_architecture"
        dot_graph.render(output_graph_filename, format='pdf', cleanup=True) # cleanup=True löscht die .dot Quelldatei
        print(f"--- Modellgraph (torchviz) ---")
        print(f"Graph der Modellarchitektur wurde als '{output_graph_filename}.pdf' gespeichert.")
        print("Hinweis: Für DenseNet121 kann dieser Graph sehr groß und detailliert sein.")
        print("-" * 50 + "\n")

    except ImportError:
        print("torchviz ist nicht installiert. Überspringe torchviz-Grapherstellung.")
        print("Installiere mit: pip install torchviz")
    except NameError as ne: # Fängt Fehler ab, wenn Graphviz nicht im Systempfad ist
        if "dot" in str(ne).lower():
             print("Fehler bei der torchviz-Grapherstellung: Graphviz 'dot' Kommando nicht gefunden.")
             print("Stelle sicher, dass Graphviz installiert und zum System-PATH hinzugefügt ist.")
        else:
            print(f"Unerwarteter NameError bei torchviz: {ne}")
    except Exception as e:
        print(f"Fehler bei der torchviz-Grapherstellung: {e}")


if __name__ == "__main__":
    main()