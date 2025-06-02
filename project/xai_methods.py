# xai_methods.py
"""
Modulare XAI-Methoden für das eval_model_with_xai Script.
Jede XAI-Methode wird als separate Klasse implementiert mit einheitlicher Schnittstelle.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List
import warnings
from skimage.segmentation import slic
from sklearn.linear_model import LinearRegression

# Captum Imports
try:
    from captum.attr import (
        LRP, DeepLift, IntegratedGradients, GradientShap,
        LayerGradCam, LayerAttribution, LayerActivation,
        NoiseTunnel, Occlusion, InputXGradient, GuidedGradCam,
        GuidedBackprop
    )
    # Für neuere Captum Versionen (falls verfügbar)
    try:
        from captum.attr import XRAI
        XRAI_AVAILABLE = True
    except ImportError:
        XRAI_AVAILABLE = False
        print("WARNUNG: XRAI nicht in dieser Captum-Version verfügbar. Verwende Custom-Implementation.")
    
    CAPTUM_AVAILABLE = True
    print("INFO: Captum erfolgreich importiert.")
except ImportError:
    print("WARNUNG: Captum nicht gefunden. XAI-Funktionalität wird eingeschränkt sein.")
    CAPTUM_AVAILABLE = False
    XRAI_AVAILABLE = False

# SHAP Imports
try:
    import shap
    SHAP_AVAILABLE = True
    print("INFO: SHAP erfolgreich importiert.")
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNUNG: SHAP nicht gefunden. SHAP-basierte Methoden werden nicht verfügbar sein.")

# Zusätzliche Bibliotheken für ScoreCAM (falls benötigt)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("INFO: OpenCV nicht verfügbar, einige XAI-Methoden könnten eingeschränkt sein.")

# Zusätzliche Bibliotheken für XRAI-Implementation
try:
    from skimage.segmentation import slic, felzenszwalb, watershed
    from skimage.filters import sobel
    from sklearn.linear_model import LinearRegression
    from scipy import ndimage as ndi
    # peak_local_maxima optional - nicht alle scikit-image Versionen haben es
    try:
        from skimage.feature import peak_local_maxima
    except ImportError:
        print("INFO: peak_local_maxima nicht verfügbar, verwende scipy.ndimage Alternativen.")
    SEGMENTATION_AVAILABLE = True
    print("INFO: scikit-image und scikit-learn erfolgreich importiert.")
except ImportError as e:
    SEGMENTATION_AVAILABLE = False
    print(f"INFO: scikit-image oder scikit-learn nicht verfügbar: {e}")
    print("erweiterte Segmentierungsmethoden werden eingeschränkt sein.")


class BaseXAIMethod(ABC):
    """
    Abstrakte Basisklasse für alle XAI-Methoden.
    Definiert einheitliche Schnittstelle für verschiedene Explainability-Algorithmen.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @abstractmethod
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        """
        Berechnet Attributionen für gegebene Eingaben und Zielklasse.
        
        Args:
            inputs: Eingabetensor (N, C, H, W)
            target: Zielklassen-Index
            **kwargs: Methodenspezifische Parameter
            
        Returns:
            Attribution tensor gleicher Größe wie inputs
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Gibt den Namen der XAI-Methode zurück."""
        pass
    
    def prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Bereitet Eingaben für XAI-Methode vor (Gradient requirements etc.)."""
        return inputs.clone().detach().requires_grad_(True)


class DummyXAIMethod(BaseXAIMethod):
    """Fallback XAI-Methode für Testzwecke."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        super().__init__(model, device)
        print("INFO: DummyXAIMethod initialisiert.")
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        print(f"INFO: DummyXAIMethod.attribute für Target {target}. Gebe Zufalls-Attributionen zurück.")
        return torch.rand_like(inputs)
    
    def get_method_name(self) -> str:
        return "DummyXAI"


class LRPMethod(BaseXAIMethod):
    """Layer-wise Relevance Propagation."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für LRP erforderlich.")
        self.lrp = LRP(model)
        print("INFO: LRP-Methode initialisiert.")
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        return self.lrp.attribute(inputs_prepared, target=target)
    
    def get_method_name(self) -> str:
        return "LRP"


class DeepLiftMethod(BaseXAIMethod):
    """DeepLift Attribution."""
    
    def __init__(self, model: nn.Module, device: torch.device = None, baseline: str = "zero"):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für DeepLift erforderlich.")
        self.deeplift = DeepLift(model)
        self.baseline = baseline
        print("INFO: DeepLift-Methode initialisiert.")
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        
        # Baseline erstellen
        if self.baseline == "zero":
            baselines = torch.zeros_like(inputs_prepared)
        elif self.baseline == "random":
            baselines = torch.rand_like(inputs_prepared)
        else:
            baselines = None
            
        return self.deeplift.attribute(inputs_prepared, baselines=baselines, target=target)
    
    def get_method_name(self) -> str:
        return "DeepLift"


class IntegratedGradientsMethod(BaseXAIMethod):
    """Integrated Gradients Attribution."""
    
    def __init__(self, model: nn.Module, device: torch.device = None, n_steps: int = 50):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für Integrated Gradients erforderlich.")
        self.ig = IntegratedGradients(model)
        self.n_steps = n_steps
        print("INFO: Integrated Gradients-Methode initialisiert.")
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        baselines = torch.zeros_like(inputs_prepared)
        return self.ig.attribute(inputs_prepared, baselines=baselines, target=target, n_steps=self.n_steps)
    
    def get_method_name(self) -> str:
        return "IntegratedGradients"


class GradCAMPlusMethod(BaseXAIMethod):
    """GradCAM++ Implementation using LayerGradCam from Captum."""
    
    def __init__(self, model: nn.Module, target_layer: str = None, device: torch.device = None):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für GradCAM++ erforderlich.")
            
        # Automatische Layer-Erkennung für gängige Architekturen
        if target_layer is None:
            target_layer = self._find_target_layer()
        
        self.target_layer = self._get_layer_by_name(target_layer)
        self.gradcam = LayerGradCam(model, self.target_layer)
        print(f"INFO: GradCAM++-Methode initialisiert mit Layer: {target_layer}")
    
    def _find_target_layer(self) -> str:
        """Findet automatisch einen geeigneten Ziellayer."""
        # Für DenseNet
        if hasattr(self.model, 'features'):
            if hasattr(self.model.features, 'denseblock4'):
                return "features.denseblock4"
            elif hasattr(self.model.features, 'layer4'):
                return "features.layer4"
        
        # Für ResNet
        if hasattr(self.model, 'layer4'):
            return "layer4"
              # Fallback: letztes Convolutional Layer finden
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                conv_layers.append(name)
        if conv_layers:
            return conv_layers[-1]
        
        raise ValueError("Konnte keinen geeigneten Layer für GradCAM++ finden. Bitte explizit angeben.")
    
    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Holt Layer-Objekt anhand des Namens."""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        attribution = self.gradcam.attribute(inputs_prepared, target=target)
        
        # Interpoliere auf Eingabegröße
        if attribution.shape[-2:] != inputs_prepared.shape[-2:]:
            attribution = torch.nn.functional.interpolate(
                attribution, size=inputs_prepared.shape[-2:], mode='bilinear', align_corners=False
            )
        
        return attribution
    
    def get_method_name(self) -> str:
        return "GradCAMPlus"


class GuidedGradCAMMethod(BaseXAIMethod):
    """
    Guided Grad-CAM Implementation using Captum.
    Combines GradCAM with Guided Backpropagation for high-resolution explanations.
    """
    
    def __init__(self, model: nn.Module, target_layer: str = None, device: torch.device = None):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für Guided Grad-CAM erforderlich.")
            
        # Automatische Layer-Erkennung (gleiche Logik wie GradCAM++)
        if target_layer is None:
            target_layer = self._find_target_layer(model)
        
        try:
            self.target_layer = self._get_layer_by_name(model, target_layer)
        except Exception as e:
            print(f"WARNUNG: Konnte Layer '{target_layer}' nicht finden: {e}")
            print("Verwende letzten Convolutional Layer als Fallback.")
            self.target_layer = self._find_last_conv_layer(model)
        
        self.guided_gradcam = GuidedGradCam(model, self.target_layer)
        print(f"INFO: Guided Grad-CAM-Methode initialisiert mit Layer: {target_layer}")
    
    def _find_target_layer(self, model: nn.Module) -> str:
        """Findet automatisch einen geeigneten Ziellayer (gleiche Logik wie GradCAM++)."""
        if hasattr(model, 'features'):
            if hasattr(model.features, 'denseblock4'):  # DenseNet
                return "features.denseblock4.denselayer16.conv2"
            elif hasattr(model.features, 'layer4'):  # ResNet-ähnlich
                return "features.layer4"
            else:
                return "features"
        elif hasattr(model, 'layer4'):  # Direct ResNet
            return "layer4"
        else:
            raise ValueError("Konnte keinen geeigneten Layer für Guided Grad-CAM finden. Bitte explizit angeben.")
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Holt einen Layer per Namen (dotted notation)."""
        layer = model
        for attr in layer_name.split('.'):
            layer = getattr(layer, attr)
        return layer
    
    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Fallback: Findet den letzten Convolutional Layer."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv if last_conv else list(model.children())[-1]
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        try:
            attribution = self.guided_gradcam.attribute(inputs_prepared, target=target)
            return attribution
        except Exception as e:
            print(f"WARNUNG: Guided Grad-CAM fehlgeschlagen: {e}")
            # Fallback: Normale GradCAM
            gradcam = LayerGradCam(self.model, self.target_layer)
            attribution = gradcam.attribute(inputs_prepared, target=target)
            if attribution.shape[-2:] != inputs_prepared.shape[-2:]:
                attribution = torch.nn.functional.interpolate(
                    attribution, size=inputs_prepared.shape[-2:], mode='bilinear', align_corners=False
                )
            return attribution
    
    def get_method_name(self) -> str:
        return "GuidedGradCAM"


class ReciproCAMMethod(BaseXAIMethod):
    """
    Recipro-CAM Implementation.
    A novel CAM method that uses reciprocal gradients to address vanishing gradient issues.
    """
    
    def __init__(self, model: nn.Module, target_layer: str = None, device: torch.device = None, 
                 epsilon: float = 1e-7, gamma: float = 2.0):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für Recipro-CAM erforderlich.")
            
        # Automatische Layer-Erkennung
        if target_layer is None:
            target_layer = self._find_target_layer(model)
        
        try:
            self.target_layer = self._get_layer_by_name(model, target_layer)
        except Exception as e:
            print(f"WARNUNG: Konnte Layer '{target_layer}' nicht finden: {e}")
            print("Verwende letzten Convolutional Layer als Fallback.")
            self.target_layer = self._find_last_conv_layer(model)
        
        self.epsilon = epsilon  # Verhindert Division durch Null
        self.gamma = gamma      # Potenz für Reciprocal-Transformation
        
        print(f"INFO: Recipro-CAM-Methode initialisiert mit Layer: {target_layer}, epsilon: {epsilon}, gamma: {gamma}")
    
    def _find_target_layer(self, model: nn.Module) -> str:
        """Findet automatisch einen geeigneten Ziellayer."""
        if hasattr(model, 'features'):
            if hasattr(model.features, 'denseblock4'):  # DenseNet
                return "features.denseblock4.denselayer16.conv2"
            elif hasattr(model.features, 'layer4'):  # ResNet-ähnlich
                return "features.layer4"
            else:
                return "features"
        elif hasattr(model, 'layer4'):  # Direct ResNet
            return "layer4"
        else:
            raise ValueError("Konnte keinen geeigneten Layer für Recipro-CAM finden. Bitte explizit angeben.")
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Holt einen Layer per Namen (dotted notation)."""
        layer = model
        for attr in layer_name.split('.'):
            layer = getattr(layer, attr)
        return layer
    
    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Fallback: Findet den letzten Convolutional Layer."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv if last_conv else list(model.children())[-1]
    
    def _reciprocal_transform(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Anwendung der Reciprocal-Transformation auf Gradienten.
        Formel: reciprocal_grad = sign(grad) * (1 / (|grad| + epsilon))^gamma
        """
        abs_grad = torch.abs(gradients)
        sign_grad = torch.sign(gradients)
        reciprocal_grad = sign_grad * torch.pow(1.0 / (abs_grad + self.epsilon), self.gamma)
        
        # Normalisierung um extreme Werte zu vermeiden
        reciprocal_grad = torch.clamp(reciprocal_grad, -1e6, 1e6)
        
        return reciprocal_grad
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        inputs_prepared.requires_grad_(True)
        
        # Hook für Feature-Maps und Gradienten
        feature_maps = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            if grad_output[0] is not None:
                gradients = grad_output[0]
        
        # Registriere Hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        
        try:
            # Forward Pass
            self.model.eval()
            outputs = self.model(inputs_prepared)
            
            # Backward Pass für spezifische Klasse
            if target is not None:
                target_output = outputs[:, target].sum()
            else:
                target_output = outputs.max(1)[0].sum()
            
            # Gradienten berechnen
            self.model.zero_grad()
            target_output.backward(retain_graph=True)
            
            if gradients is None or feature_maps is None:
                raise RuntimeError("Konnte Gradienten oder Feature-Maps nicht erfassen")
            
            # Reciprocal-Transformation auf Gradienten anwenden
            reciprocal_gradients = self._reciprocal_transform(gradients)
            
            # Global Average Pooling der transformierten Gradienten
            weights = torch.mean(reciprocal_gradients, dim=(2, 3), keepdim=True)
            
            # Gewichtete Kombination der Feature-Maps
            cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
            
            # ReLU anwenden (nur positive Aktivierungen)
            cam = torch.relu(cam)
            
            # Normalisierung
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Interpolation auf Eingabegröße
            if cam.shape[-2:] != inputs_prepared.shape[-2:]:
                cam = torch.nn.functional.interpolate(
                    cam, size=inputs_prepared.shape[-2:], mode='bilinear', align_corners=False
                )
            
            # Auf 3 Kanäle erweitern für Konsistenz
            cam = cam.repeat(1, 3, 1, 1)
            
            return cam
            
        except Exception as e:
            print(f"WARNUNG: Recipro-CAM fehlgeschlagen: {e}")
            # Fallback zu GradCAM
            gradcam = LayerGradCam(self.model, self.target_layer)
            attribution = gradcam.attribute(inputs_prepared, target=target)
            if attribution.shape[-2:] != inputs_prepared.shape[-2:]:
                attribution = torch.nn.functional.interpolate(
                    attribution, size=inputs_prepared.shape[-2:], mode='bilinear', align_corners=False
                )
            return attribution
            
        finally:
            # Hooks entfernen
            forward_handle.remove()
            backward_handle.remove()
    
    def get_method_name(self) -> str:
        return "ReciproCAM"


class XRAIMethod(BaseXAIMethod):
    """XRAI (eXplanation through Regional Aggregation of Importance) Implementation."""
    
    def __init__(self, model: nn.Module, device: torch.device = None, n_steps: int = 50):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE or not XRAI_AVAILABLE:
            raise ImportError("Captum mit XRAI ist für diese Methode erforderlich.")
        
        # XRAI benötigt oft eine Basismethode (z.B. Integrated Gradients)
        self.base_method = IntegratedGradients(model)
        self.xrai = XRAI(model)
        self.n_steps = n_steps
        print("INFO: XRAI-Methode initialisiert.")
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        baselines = torch.zeros_like(inputs_prepared)
        
        try:
            return self.xrai.attribute(
                inputs_prepared, 
                target=target, 
                baselines=baselines,
                n_steps=self.n_steps
            )
        except Exception as e:
            print(f"WARNUNG: XRAI fehlgeschlagen, verwende Integrated Gradients als Fallback: {e}")
            return self.base_method.attribute(inputs_prepared, baselines=baselines, target=target)
    
    def get_method_name(self) -> str:
        return "XRAI"


class ScoreCAMMethod(BaseXAIMethod):
    """
    ScoreCAM Implementation.
    Vereinfachte Implementation, da ScoreCAM nicht direkt in Captum verfügbar ist.
    """
    
    def __init__(self, model: nn.Module, target_layer: str = None, device: torch.device = None, batch_size: int = 32):
        super().__init__(model, device)
        
        if target_layer is None:
            target_layer = self._find_target_layer()
        
        self.target_layer_name = target_layer
        self.target_layer = self._get_layer_by_name(target_layer)
        self.batch_size = batch_size
        self.activations = None
        self._register_hooks()
        print(f"INFO: ScoreCAM-Methode initialisiert mit Layer: {target_layer}")
    
    def _find_target_layer(self) -> str:
        """Findet automatisch einen geeigneten Ziellayer (gleiche Logik wie GradCAM++)."""
        # Für DenseNet
        if hasattr(self.model, 'features'):
            if hasattr(self.model.features, 'denseblock4'):
                return "features.denseblock4"
        
        # Für ResNet
        if hasattr(self.model, 'layer4'):
            return "layer4"
            
        # Fallback
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                conv_layers.append(name)
        
        if conv_layers:
            return conv_layers[-1]
        
        raise ValueError("Konnte keinen geeigneten Layer für ScoreCAM finden.")
    
    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Holt Layer-Objekt anhand des Namens."""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def _register_hooks(self):
        """Registriert Forward-Hook für Aktivationen."""
        def hook_fn(module, input, output):
            self.activations = output.detach()
        
        self.target_layer.register_forward_hook(hook_fn)
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = inputs.to(self.device)
        
        # Forward pass um Aktivationen zu erhalten
        with torch.no_grad():
            _ = self.model(inputs_prepared)
        
        if self.activations is None:
            print("WARNUNG: Keine Aktivationen erhalten, verwende Zufallsattribution.")
            return torch.rand_like(inputs_prepared)
        
        # ScoreCAM Algorithmus (vereinfacht)
        activations = self.activations  # (N, C, H, W)
        N, C, H, W = activations.shape
        input_size = inputs_prepared.shape[-2:]
        
        # Normalisiere Aktivationen
        activations_norm = torch.zeros_like(activations)
        for i in range(C):
            act_map = activations[0, i]  # Nehme ersten Batch
            if act_map.max() > act_map.min():
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())
            activations_norm[0, i] = act_map
        
        # Skaliere Aktivationskarten auf Eingabegröße
        activations_upsampled = torch.nn.functional.interpolate(
            activations_norm, size=input_size, mode='bilinear', align_corners=False
        )
        
        # Berechne Scores für jede Aktivationskarte
        scores = []
        for i in range(C):
            # Maskiere Input mit Aktivationskarte
            masked_input = inputs_prepared * activations_upsampled[:, i:i+1]
            
            with torch.no_grad():
                output = self.model(masked_input)
                prob = torch.softmax(output, dim=1)
                score = prob[0, target].item()
            
            scores.append(score)
        
        scores = torch.tensor(scores, device=self.device)
        
        # Gewichtete Kombination der Aktivationskarten
        final_attribution = torch.zeros_like(inputs_prepared)
        for i in range(C):
            final_attribution += scores[i] * activations_upsampled[:, i:i+1]
        
        return final_attribution
    
    def get_method_name(self) -> str:
        return "ScoreCAM"


class InputXGradientMethod(BaseXAIMethod):
    """Input × Gradient Attribution Method."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für Input×Gradient erforderlich.")
        self.input_x_gradient = InputXGradient(model)
        print("INFO: Input×Gradient-Methode initialisiert.")
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        return self.input_x_gradient.attribute(inputs_prepared, target=target)
    
    def get_method_name(self) -> str:
        return "InputXGradient"


class CustomXRAIMethod(BaseXAIMethod):
    """
    Custom XRAI-like Implementation using superpixel segmentation.
    XRAI (eXplanation through Regional Aggregation of Importance) 
    aggregates pixel-level attributions into coherent regions.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None, 
                 base_method: str = "IntegratedGradients", n_segments: int = 100,
                 compactness: float = 10.0):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für Custom XRAI erforderlich.")
        if not SEGMENTATION_AVAILABLE:
            raise ImportError("scikit-image und scikit-learn sind für Custom XRAI erforderlich.")
        
        # Basis-Attributionsmethode
        if base_method.lower() == "integratedgradients":
            self.base_method = IntegratedGradients(model)
            self.baseline_needed = True
        elif base_method.lower() == "inputxgradient":
            self.base_method = InputXGradient(model)
            self.baseline_needed = False
        else:
            self.base_method = IntegratedGradients(model)
            self.baseline_needed = True
            
        self.n_segments = n_segments
        self.compactness = compactness
        print(f"INFO: Custom XRAI-Methode initialisiert mit {base_method} als Basis.")
    
    def _get_superpixels(self, image_np: np.ndarray) -> np.ndarray:
        """Erstellt Superpixel-Segmentierung."""
        # Konvertiere (C, H, W) zu (H, W, C) für SLIC
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Stelle sicher, dass Werte im Bereich [0, 1] sind
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # SLIC Segmentierung
        segments = slic(image_np, n_segments=self.n_segments, 
                       compactness=self.compactness, start_label=1)
        return segments
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        
        # Basis-Attributionen berechnen
        if self.baseline_needed:
            baselines = torch.zeros_like(inputs_prepared)
            base_attributions = self.base_method.attribute(
                inputs_prepared, baselines=baselines, target=target
            )
        else:
            base_attributions = self.base_method.attribute(inputs_prepared, target=target)
        
        # Für jedes Bild im Batch
        batch_size = inputs_prepared.shape[0]
        final_attributions = torch.zeros_like(base_attributions)
        
        for i in range(batch_size):
            # Konvertiere zu numpy für Segmentierung
            image_np = inputs_prepared[i].detach().cpu().numpy()
            attr_np = base_attributions[i].detach().cpu().numpy()
            
            # Superpixel-Segmentierung
            segments = self._get_superpixels(image_np)
            
            # Aggregiere Attributionen für jeden Superpixel
            aggregated_attr = np.zeros_like(attr_np)
            
            for segment_id in np.unique(segments):
                if segment_id == 0:  # Hintergrund überspringen
                    continue
                
                # Maske für aktuellen Superpixel
                mask = segments == segment_id
                
                # Berechne mittlere Attribution für diesen Superpixel
                if np.any(mask):
                    for c in range(attr_np.shape[0]):
                        segment_mean = np.mean(attr_np[c][mask])
                        aggregated_attr[c][mask] = segment_mean
            
            final_attributions[i] = torch.from_numpy(aggregated_attr).to(self.device)
        
        return final_attributions
    
    def get_method_name(self) -> str:
        return "CustomXRAI"

class SHAPMethod(BaseXAIMethod):
    """
    SHAP (SHapley Additive exPlanations) for Computer Vision.
    Uses SHAP's Partition explainer with superpixel masking.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None, 
                 masker_type: str = "partition", n_samples: int = 100,
                 max_evals: int = 50):  # Reduziert für bessere Performance
        super().__init__(model, device)
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP ist für SHAP-Methode erforderlich.")
        
        self.masker_type = masker_type
        self.n_samples = n_samples
        self.max_evals = max_evals
        print(f"INFO: SHAP-Methode initialisiert mit {masker_type} masker.")
    
    def _create_model_wrapper(self, target_class: int):
        """Erstellt einen Wrapper für das Modell, der nur die Zielklasse zurückgibt."""
        def model_wrapper(images):
            # Konvertiere numpy zu torch falls nötig
            if isinstance(images, np.ndarray):
                # SHAP sendet oft Batches von Bildern
                if images.ndim == 4:  # (batch, H, W, C)
                    # Konvertiere zu (batch, C, H, W)
                    images = np.transpose(images, (0, 3, 1, 2))
                elif images.ndim == 3:  # (H, W, C)
                    # Konvertiere zu (1, C, H, W)
                    images = np.transpose(images, (2, 0, 1))
                    images = images[np.newaxis, :]
                elif images.ndim == 1:  # Flacher Array vom Masker
                    # Rekonstruiere die ursprüngliche Form - das ist der Schlüssel!
                    # images sollte die Form (H*W*C,) haben
                    images = images.reshape(self.original_height, self.original_width, 3)
                    images = np.transpose(images, (2, 0, 1))
                    images = images[np.newaxis, :]
                
                images = torch.from_numpy(images).float().to(self.device)
            
            # Normalisierung (SHAP gibt [0,255] zurück, Modell erwartet [0,1])
            if images.max() > 1.0:
                images = images / 255.0
            
            with torch.no_grad():
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                # Gib die Wahrscheinlichkeit für die Zielklasse zurück
                result = probs[:, target_class].cpu().numpy()
                
                if len(result) == 1:
                    return result[0]
                else:
                    return result
        
        return model_wrapper
    
    def _preprocess_image_for_shap(self, image_np: np.ndarray):
        """Bereitet das Bild KORREKT für SHAP vor."""
        # Input ist (C, H, W) = (3, 224, 1400)
        # SHAP braucht (H, W, C) = (224, 1400, 3)
        
        if image_np.shape[0] in [1, 3]:  # Kanäle sind erste Dimension
            image_np = np.transpose(image_np, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        
        # Handle Grayscale
        if image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
        
        # Normalisiere zu [0, 255] für SHAP ImageMasker
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # Speichere Dimensionen für Model-Wrapper
        self.original_height, self.original_width = image_np.shape[:2]
        
        return image_np
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        # Nur erstes Bild im Batch verarbeiten (SHAP ist rechenintensiv)
        input_single = inputs[0:1]
        
        # Konvertiere zu numpy für SHAP
        image_np = input_single[0].detach().cpu().numpy()
        
        # Bereite Bild vor
        image_prepared = self._preprocess_image_for_shap(image_np)
        
        try:
            # Erstelle Masker - verwende das Bild selbst als Referenz
            if self.masker_type == "partition":
                # Verwende das Bild direkt für den Masker
                masker = shap.maskers.Image("inpaint_telea", image_prepared)
            elif self.masker_type == "pixel":
                masker = shap.maskers.Image("blur(1,1)", image_prepared)
            else:
                masker = shap.maskers.Image("inpaint_telea", image_prepared)
            
            # Erstelle Model-Wrapper
            model_wrapper = self._create_model_wrapper(target)
            
            # Erstelle Explainer
            explainer = shap.Explainer(model_wrapper, masker, 
                                     output_names=[f"Class_{target}"])
            
            # Berechne SHAP-Werte mit stark reduzierter max_evals für Geschwindigkeit
            shap_values = explainer(
                image_prepared, 
                max_evals=self.max_evals,
                batch_size=1
            )
            
            # Extrahiere Attribution
            if hasattr(shap_values, 'values'):
                attribution_np = shap_values.values
            else:
                attribution_np = shap_values
            
            # Handle verschiedene SHAP output formats
            if isinstance(attribution_np, list):
                attribution_np = attribution_np[0]
            
            # Stelle sicher, dass es ein numpy array ist
            if not isinstance(attribution_np, np.ndarray):
                attribution_np = np.array(attribution_np)
            
            # Konvertiere zurück zu (C, H, W) Format
            if attribution_np.ndim == 3 and attribution_np.shape[-1] in [1, 3]:
                attribution_np = np.transpose(attribution_np, (2, 0, 1))
            elif attribution_np.ndim == 2:
                attribution_np = attribution_np[np.newaxis, :, :]
            
            # Konvertiere zu torch tensor
            attribution_tensor = torch.from_numpy(attribution_np).float().to(self.device)
            
            # Erweitere auf Batch-Größe
            if attribution_tensor.ndim == 3:
                attribution_tensor = attribution_tensor.unsqueeze(0)
            
            # Erweitere für fehlende Kanäle falls nötig
            if attribution_tensor.shape[1] < inputs.shape[1]:
                attribution_tensor = attribution_tensor.repeat(1, inputs.shape[1], 1, 1)
            
            # Stelle sicher, dass die Größe stimmt
            if attribution_tensor.shape[-2:] != inputs.shape[-2:]:
                attribution_tensor = torch.nn.functional.interpolate(
                    attribution_tensor, size=inputs.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
            
            return attribution_tensor
            
        except Exception as e:
            print(f"WARNUNG: SHAP-Attribution fehlgeschlagen: {e}")
            print("SHAP wird übersprungen - kein Fallback.")
            # Gib None zurück statt Fallback
            return None
            
    def get_method_name(self) -> str:
        return "SHAP"

class RISEMethod(BaseXAIMethod):
    """
    RISE (Randomized Input Sampling for Explanation).
    Perturbation-basierte XAI-Methode die zufällige Masken verwendet.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None, 
                 n_masks: int = 2000, p1: float = 0.1, s: int = 8):
        super().__init__(model, device)
        self.n_masks = n_masks
        self.p1 = p1  # Wahrscheinlichkeit, dass ein Pixel in der Maske 1 ist
        self.s = s    # Upsampling-Faktor für die Masken
        print(f"INFO: RISE-Methode initialisiert mit {n_masks} Masken.")
    
    def _generate_random_masks(self, input_size: tuple, n_masks: int) -> torch.Tensor:
        """Generiert zufällige binäre Masken für RISE."""
        _, _, h, w = input_size
        
        # Erstelle kleinere Masken die dann upsampled werden
        small_h, small_w = h // self.s, w // self.s
        
        # Generiere zufällige Masken
        masks = torch.rand(n_masks, 1, small_h, small_w, device=self.device) < self.p1
        masks = masks.float()
        
        # Upsample zu ursprünglicher Größe
        masks = torch.nn.functional.interpolate(
            masks, size=(h, w), mode='bilinear', align_corners=False
        )
        
        return masks
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = inputs.to(self.device)
        batch_size, channels, height, width = inputs_prepared.shape
        
        # Generiere zufällige Masken
        masks = self._generate_random_masks(inputs_prepared.shape, self.n_masks)
        
        # Sammle Scores für alle Masken
        scores = []
        
        for i in range(self.n_masks):
            # Wende Maske auf Input an
            masked_input = inputs_prepared * masks[i:i+1]
            
            with torch.no_grad():
                output = self.model(masked_input)
                prob = torch.softmax(output, dim=1)
                score = prob[0, target].item()
            
            scores.append(score)
        
        scores = torch.tensor(scores, device=self.device)
        
        # Berechne gewichtete Summe der Masken
        attribution = torch.zeros_like(inputs_prepared)
        for i in range(self.n_masks):
            attribution += scores[i] * masks[i:i+1]
        
        # Normalisiere durch Anzahl der Masken
        attribution = attribution / self.n_masks
        
        return attribution
    
    def get_method_name(self) -> str:
        return "RISE"


class OcclusionMethod(BaseXAIMethod):
    """
    Occlusion-basierte XAI-Methode using Captum.
    Schiebt ein Occlusionsfenster über das Bild und misst den Einfluss.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None,
                 sliding_window_shapes: tuple = (3, 15, 15), strides: tuple = (1, 8, 8)):
        super().__init__(model, device)
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für Occlusion erforderlich.")
        
        self.occlusion = Occlusion(model)
        self.sliding_window_shapes = sliding_window_shapes
        self.strides = strides
        print(f"INFO: Occlusion-Methode initialisiert mit Fenster {sliding_window_shapes} und Strides {strides}.")
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        
        try:
            # Occlusion ohne baselines Parameter - der wird automatisch als Nullen verwendet
            attribution = self.occlusion.attribute(
                inputs_prepared,
                target=target,
                sliding_window_shapes=self.sliding_window_shapes,
                strides=self.strides
            )
            return attribution
        except Exception as e:
            print(f"WARNUNG: Occlusion fehlgeschlagen: {e}")
            # Besserer Fallback: erstelle eine sinnvolle Dummy-Attribution
            return torch.zeros_like(inputs_prepared)
    
    def get_method_name(self) -> str:
        return "Occlusion"


class FelzenszwalbXAIMethod(BaseXAIMethod):
    """
    XAI-Methode basierend auf Felzenszwalb-Segmentierung.
    Verwendet Felzenszwalb-Algorithmus für semantische Segmentierung 
    und aggregiert Attributionen basierend auf den Segmenten.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None,
                 scale: float = 100, sigma: float = 0.5, min_size: int = 50,
                 base_method: str = "IntegratedGradients"):
        super().__init__(model, device)
        if not SEGMENTATION_AVAILABLE:
            raise ImportError("scikit-image ist für Felzenszwalb erforderlich.")
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für Felzenszwalb erforderlich.")
        
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        
        # Basis-Attributionsmethode
        if base_method.lower() == "integratedgradients":
            self.base_method = IntegratedGradients(model)
            self.baseline_needed = True
        elif base_method.lower() == "inputxgradient":
            self.base_method = InputXGradient(model)
            self.baseline_needed = False
        else:
            self.base_method = IntegratedGradients(model)
            self.baseline_needed = True
            
        print(f"INFO: Felzenszwalb XAI-Methode initialisiert mit {base_method} als Basis.")
    
    def _get_felzenszwalb_segments(self, image_np: np.ndarray) -> np.ndarray:
        """Erstellt Felzenszwalb-Segmentierung."""
        # Konvertiere (C, H, W) zu (H, W, C) für Felzenszwalb
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Grayscale falls nur ein Kanal
        if image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = image_np.squeeze(-1)
        
        # Stelle sicher, dass Werte im Bereich [0, 1] sind
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # Felzenszwalb Segmentierung
        segments = felzenszwalb(
            image_np, 
            scale=self.scale, 
            sigma=self.sigma, 
            min_size=self.min_size
        )
        return segments
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        
        # Basis-Attributionen berechnen
        if self.baseline_needed:
            baselines = torch.zeros_like(inputs_prepared)
            base_attributions = self.base_method.attribute(
                inputs_prepared, baselines=baselines, target=target
            )
        else:
            base_attributions = self.base_method.attribute(inputs_prepared, target=target)
        
        # Für jedes Bild im Batch
        batch_size = inputs_prepared.shape[0]
        final_attributions = torch.zeros_like(base_attributions)
        
        for i in range(batch_size):
            # Konvertiere zu numpy für Segmentierung
            image_np = inputs_prepared[i].detach().cpu().numpy()
            attr_np = base_attributions[i].detach().cpu().numpy()
            
            # Felzenszwalb-Segmentierung
            segments = self._get_felzenszwalb_segments(image_np)
            
            # Aggregiere Attributionen für jedes Segment
            aggregated_attr = np.zeros_like(attr_np)
            
            for segment_id in np.unique(segments):
                # Maske für aktuelles Segment
                mask = segments == segment_id
                
                # Berechne mittlere Attribution für dieses Segment
                if np.any(mask):
                    for c in range(attr_np.shape[0]):
                        segment_mean = np.mean(attr_np[c][mask])
                        aggregated_attr[c][mask] = segment_mean
            
            final_attributions[i] = torch.from_numpy(aggregated_attr).to(self.device)
        
        return final_attributions
    
    def get_method_name(self) -> str:
        return "Felzenszwalb"


class WatershedXAIMethod(BaseXAIMethod):
    """
    XAI-Methode basierend auf Watershed-Segmentierung.
    Verwendet Watershed-Algorithmus für intensitätsbasierte Segmentierung
    und aggregiert Attributionen basierend auf den Segmenten.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None,
                 markers: int = None, compactness: float = 0.001,
                 base_method: str = "IntegratedGradients"):
        super().__init__(model, device)
        if not SEGMENTATION_AVAILABLE:
            raise ImportError("scikit-image ist für Watershed erforderlich.")
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum ist für Watershed erforderlich.")
        
        self.markers = markers
        self.compactness = compactness
        
        # Basis-Attributionsmethode
        if base_method.lower() == "integratedgradients":
            self.base_method = IntegratedGradients(model)
            self.baseline_needed = True
        elif base_method.lower() == "inputxgradient":
            self.base_method = InputXGradient(model)
            self.baseline_needed = False
        else:
            self.base_method = IntegratedGradients(model)
            self.baseline_needed = True
            
        print(f"INFO: Watershed XAI-Methode initialisiert mit {base_method} als Basis.")
    
    def _get_watershed_segments(self, image_np: np.ndarray) -> np.ndarray:
        """Erstellt Watershed-Segmentierung."""
        # Konvertiere (C, H, W) zu (H, W, C) für Watershed
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Konvertiere zu Grayscale für Watershed
        if image_np.ndim == 3:
            if image_np.shape[2] == 3:  # RGB zu Grayscale
                image_gray = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140])
            else:  # Bereits single channel
                image_gray = image_np.squeeze(-1)
        else:
            image_gray = image_np
        
        # Stelle sicher, dass Werte im Bereich [0, 1] sind
        if image_gray.max() > 1.0:
            image_gray = image_gray / 255.0
        
        # Berechne Gradienten mit Sobel
        gradient = sobel(image_gray)
          # Finde lokale Maxima als Marker
        if self.markers is None:
            # Automatische Marker-Erkennung
            try:
                # Versuche peak_local_maxima falls verfügbar
                from skimage.feature import peak_local_maxima
                local_maxima = peak_local_maxima(image_gray, min_distance=20)
                markers = np.zeros_like(image_gray, dtype=int)
                markers[tuple(local_maxima.T)] = np.arange(1, len(local_maxima) + 1)
            except ImportError:
                # Fallback: Verwende scipy.ndimage für lokale Maxima
                from scipy.ndimage import maximum_filter
                local_maxima = maximum_filter(image_gray, size=20) == image_gray
                markers = np.zeros_like(image_gray, dtype=int)
                markers[local_maxima] = np.arange(1, np.sum(local_maxima) + 1)
        else:
            # Verwende feste Anzahl von Markern
            markers = self.markers
        
        # Watershed-Segmentierung
        segments = watershed(
            gradient, 
            markers=markers, 
            compactness=self.compactness
        )
        
        return segments
    
    def attribute(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs_prepared = self.prepare_inputs(inputs)
        
        # Basis-Attributionen berechnen
        if self.baseline_needed:
            baselines = torch.zeros_like(inputs_prepared)
            base_attributions = self.base_method.attribute(
                inputs_prepared, baselines=baselines, target=target
            )
        else:
            base_attributions = self.base_method.attribute(inputs_prepared, target=target)
        
        # Für jedes Bild im Batch
        batch_size = inputs_prepared.shape[0]
        final_attributions = torch.zeros_like(base_attributions)
        
        for i in range(batch_size):
            # Konvertiere zu numpy für Segmentierung
            image_np = inputs_prepared[i].detach().cpu().numpy()
            attr_np = base_attributions[i].detach().cpu().numpy()
            
            try:
                # Watershed-Segmentierung
                segments = self._get_watershed_segments(image_np)
                
                # Aggregiere Attributionen für jedes Segment
                aggregated_attr = np.zeros_like(attr_np)
                
                for segment_id in np.unique(segments):
                    if segment_id == 0:  # Hintergrund überspringen
                        continue
                    
                    # Maske für aktuelles Segment
                    mask = segments == segment_id
                    
                    # Berechne mittlere Attribution für dieses Segment
                    if np.any(mask):
                        for c in range(attr_np.shape[0]):
                            segment_mean = np.mean(attr_np[c][mask])
                            aggregated_attr[c][mask] = segment_mean
                
                final_attributions[i] = torch.from_numpy(aggregated_attr).to(self.device)
                
            except Exception as e:
                print(f"WARNUNG: Watershed-Segmentierung fehlgeschlagen: {e}")
                # Fallback: verwende Basis-Attributionen
                final_attributions[i] = base_attributions[i]
        
        return final_attributions
    
    def get_method_name(self) -> str:
        return "Watershed"


def create_xai_method(method_name: str, model: nn.Module, device: torch.device = None, **kwargs) -> BaseXAIMethod:
    """
    Factory-Funktion zur Erstellung von XAI-Methoden.
    
    Args:
        method_name: Name der XAI-Methode
        model: PyTorch-Modell
        device: Gerät für Berechnungen
        **kwargs: Methodenspezifische Parameter
        
    Returns:
        Instanz der entsprechenden XAI-Methode
    """
    method_name_lower = method_name.lower()
    
    if method_name_lower == "dummy":
        return DummyXAIMethod(model, device)
    elif method_name_lower == "lrp":
        return LRPMethod(model, device)
    elif method_name_lower == "deeplift":
        return DeepLiftMethod(model, device, **kwargs)
    elif method_name_lower == "integratedgradients" or method_name_lower == "ig":
        return IntegratedGradientsMethod(model, device, **kwargs)
    elif method_name_lower == "gradcam++" or method_name_lower == "gradcamplus":
        return GradCAMPlusMethod(model, device=device, **kwargs)
    elif method_name_lower == "guidedgradcam" or method_name_lower == "guided_gradcam":
        return GuidedGradCAMMethod(model, device=device, **kwargs)
    elif method_name_lower == "reciprocam" or method_name_lower == "recipro_cam":
        return ReciproCAMMethod(model, device=device, **kwargs)
    elif method_name_lower == "xrai":
        return XRAIMethod(model, device, **kwargs)
    elif method_name_lower == "scorecam":
        return ScoreCAMMethod(model, device=device, **kwargs)
    elif method_name_lower == "inputxgradient":
        return InputXGradientMethod(model, device)
    elif method_name_lower == "customxrai":
        return CustomXRAIMethod(model, device, **kwargs)
    elif method_name_lower == "shap":
        return SHAPMethod(model, device, **kwargs)
    elif method_name_lower == "rise":
        return RISEMethod(model, device, **kwargs)
    elif method_name_lower == "occlusion":
        return OcclusionMethod(model, device, **kwargs)
    elif method_name_lower == "felzenszwalb":
        return FelzenszwalbXAIMethod(model, device, **kwargs)
    elif method_name_lower == "watershed":
        return WatershedXAIMethod(model, device, **kwargs)
    else:
        available_methods = ["dummy", "lrp", "deeplift", "integratedgradients", "gradcam++", "guidedgradcam", "reciprocam", "xrai", "scorecam", "inputxgradient", "customxrai", "shap", "rise", "occlusion", "felzenszwalb", "watershed"]
        raise ValueError(f"Unbekannte XAI-Methode: {method_name}. Verfügbar: {available_methods}")


def get_available_methods() -> List[str]:
    """Gibt Liste der verfügbaren XAI-Methoden zurück."""
    base_methods = ["dummy"]
    
    if CAPTUM_AVAILABLE:
        base_methods.extend(["lrp", "deeplift", "integratedgradients", "gradcam++", "guidedgradcam", "reciprocam", "scorecam", "inputxgradient", "occlusion", "rise"])
        
        if XRAI_AVAILABLE:
            base_methods.append("xrai")
        else:
            base_methods.append("customxrai")
            
        if SEGMENTATION_AVAILABLE:
            base_methods.extend(["felzenszwalb", "watershed"])
    
    if SHAP_AVAILABLE:
        base_methods.append("shap")
    
    return base_methods
