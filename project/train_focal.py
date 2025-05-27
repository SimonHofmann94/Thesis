# project/train2.py - Version mit Focal Loss anstelle von BCEWithLogitsLoss

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import hydra
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from omegaconf import DictConfig, ListConfig
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score 
# multilabel_confusion_matrix können wir später für detailliertere Analyse hinzufügen

from datasets import SeverstalDataset # Wird jetzt als Patch-Dataset verwendet
from models import ClassifierModel
from utils import set_seed, EarlyStopping
from collections import Counter, defaultdict 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

# ----- Neue Focal Loss Klasse -----
class FocalLoss(nn.Module):
    """
    Implementierung des Focal Loss für Multi-Label-Klassifikation.
    FL(p) = -alpha * (1-p)^gamma * log(p) für positive Klassen
    FL(p) = -(1-alpha) * p^gamma * log(1-p) für negative Klassen
    
    Args:
        alpha (float): Gewichtungsfaktor für positive Klassen (ähnlich wie pos_weight)
        gamma (float): Parameter zur Reduzierung des Einflusses einfacher Beispiele
        pos_weight (Tensor): Klassenspezifische Gewichte für positive Beispiele 
        reduction (str): 'mean' oder 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.eps = 1e-6  # Kleine Konstante für numerische Stabilität
        
    def forward(self, inputs, targets):
        # Anwenden der Sigmoid-Funktion für Wahrscheinlichkeiten
        probs = torch.sigmoid(inputs)
        
        # Vermeiden von 0 und 1 für numerische Stabilität
        probs = torch.clamp(probs, min=self.eps, max=1.0-self.eps)
        
        # Loss-Berechnung für positive und negative Beispiele
        # BCE = -[y * log(p) + (1 - y) * log(1 - p)]
        bce_loss = -targets * torch.log(probs) - (1.0 - targets) * torch.log(1.0 - probs)
        
        # Fokale Gewichtung hinzufügen
        focal_weights_pos = (1 - probs) ** self.gamma
        focal_weights_neg = probs ** self.gamma
        
        # Kombinierte Gewichtung mit Alpha für Klassenbalance
        focal_weighted_bce = torch.zeros_like(bce_loss)
        
        # Positive Beispiele
        pos_mask = targets > 0
        if self.pos_weight is not None:
            focal_weighted_bce[pos_mask] = bce_loss[pos_mask] * focal_weights_pos[pos_mask] * self.alpha * self.pos_weight.view(1, -1).expand_as(inputs)[pos_mask]
        else:
            focal_weighted_bce[pos_mask] = bce_loss[pos_mask] * focal_weights_pos[pos_mask] * self.alpha
            
        # Negative Beispiele
        neg_mask = targets <= 0
        focal_weighted_bce[neg_mask] = bce_loss[neg_mask] * focal_weights_neg[neg_mask] * (1 - self.alpha)
        
        # Aggregation
        if self.reduction == 'mean':
            return torch.mean(focal_weighted_bce)
        elif self.reduction == 'sum':
            return torch.sum(focal_weighted_bce)
        else:  # 'none'
            return focal_weighted_bce


def analyze_dataset_composition(subset_dataset: torch.utils.data.Subset, dataset_name: str, underlying_patch_dataset_info: list):
    """
    Analysiert und gibt die Zusammensetzung eines Datasets (Subset von Patches) aus,
    basierend auf den Originalbildern und der Anzahl der Patches pro Bild.

    Args:
        subset_dataset (torch.utils.data.Subset): Das zu analysierende Dataset (z.B. train_dataset).
        dataset_name (str): Name des Datasets für die Ausgabe (z.B. "Trainingsset").
        underlying_patch_dataset_info (list): Die 'patches_info'-Liste des ursprünglichen SeverstalDataset.
    """
    print(f"\n--- Analyse der Zusammensetzung des {dataset_name} ---")
    
    image_patch_counts = defaultdict(lambda: {"total": 0, "no_defect": 0, "defect_1": 0, "defect_2": 0, "defect_3": 0, "defect_4": 0})
    
    if not subset_dataset.indices:
        print(f"Das {dataset_name} enthält keine Patches.")
        return

    for patch_idx_in_original_list in subset_dataset.indices:
        patch_info = underlying_patch_dataset_info[patch_idx_in_original_list]
        image_basename = os.path.basename(patch_info["image_path"])
        label_vector = patch_info["label_vector"] # Ist ein Tensor

        image_patch_counts[image_basename]["total"] += 1
        if label_vector[SeverstalDataset.NO_DEFECT_IDX].item() == 1.0:
            image_patch_counts[image_basename]["no_defect"] += 1
        if label_vector[SeverstalDataset.DEFECT_CLASS_TO_IDX["defect_1"]].item() == 1.0:
            image_patch_counts[image_basename]["defect_1"] += 1
        if label_vector[SeverstalDataset.DEFECT_CLASS_TO_IDX["defect_2"]].item() == 1.0:
            image_patch_counts[image_basename]["defect_2"] += 1
        if label_vector[SeverstalDataset.DEFECT_CLASS_TO_IDX["defect_3"]].item() == 1.0:
            image_patch_counts[image_basename]["defect_3"] += 1
        if label_vector[SeverstalDataset.DEFECT_CLASS_TO_IDX["defect_4"]].item() == 1.0:
            image_patch_counts[image_basename]["defect_4"] += 1
            
    print(f"Das {dataset_name} wurde aus {len(image_patch_counts)} einzigartigen Originalbildern generiert.")
    print(f"Gesamtzahl der Patches im {dataset_name}: {len(subset_dataset.indices)}")
    
    print("\nDetails pro Originalbild (Bild: Gesamt-Patches [K0, K1, K2, K3, K4]):")
    # Sortiere nach Bildnamen für konsistente Ausgabe
    for img_name, counts in sorted(image_patch_counts.items()):
        print(f"  {img_name}: {counts['total']} Patches "
              f"[{counts['no_defect']}, {counts['defect_1']}, {counts['defect_2']}, {counts['defect_3']}, {counts['defect_4']}]")
    
    print(f"--- Ende der Analyse für {dataset_name} ---\n")


def get_transforms(cfg: DictConfig, img_h_config: int, img_w_config: int, is_train: bool = True):
    """
    Erzeugt Transformationspipelines.
    'is_train' steuert, ob Augmentierungen angewendet werden.
    img_h_config, img_w_config sind jetzt die Patch-Dimensionen.
    """
    aug = cfg.augment
    pil_transforms_list = []

    if is_train:
        if aug.get("use_resizedcrop", False):
            pil_transforms_list.append(transforms.RandomResizedCrop((img_h_config, img_w_config), scale=(0.8, 1.0), ratio=(0.9, 1.1)))
        else:
            pil_transforms_list.append(transforms.Resize((img_h_config, img_w_config))) # Muss da sein, wenn kein RRC
        if aug.get("use_hflip", False):
            pil_transforms_list.append(transforms.RandomHorizontalFlip())
        if aug.get("use_vflip", False):
            pil_transforms_list.append(transforms.RandomVerticalFlip())
        if aug.get("use_rotation", False):
            pil_transforms_list.append(transforms.RandomRotation(aug.get("rotation_degree", 15)))
        if aug.get("use_colorjitter", False):
            pil_transforms_list.append(
                transforms.ColorJitter(brightness=aug.get("cj_brightness", 0.2), 
                                       contrast=aug.get("cj_contrast", 0.2),
                                       saturation=aug.get("cj_saturation", 0.2),
                                       hue=aug.get("cj_hue", 0.1))
            )
        if aug.get("use_perspective", False):
            pil_transforms_list.append(
                transforms.RandomPerspective(distortion_scale=aug.get("perspective_distortion_scale",0.5),
                                             p=aug.get("perspective_p", 0.5))
            )
    else: # Für Validierung/Test
        pil_transforms_list.append(transforms.Resize((img_h_config, img_w_config)))

    pil_transforms_list.append(transforms.ToTensor())

    tensor_transforms_list = []
    if is_train and aug.get("use_blur", False): # Blur nur im Training
        tensor_transforms_list.append(transforms.GaussianBlur(kernel_size=tuple(aug.get("blur_kernel", [3,3])), 
                                                              sigma=tuple(aug.get("blur_sigma", [0.1, 2.0]))))
    if is_train and aug.get("use_erasing", False): # Erasing nur im Training
        tensor_transforms_list.append(transforms.RandomErasing(p=aug.get("erasing_p",0.5), 
                                                              scale=tuple(aug.get("erasing_scale", [0.02, 0.33])), 
                                                              ratio=tuple(aug.get("erasing_ratio", [0.3, 3.3]))))
    
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_transforms_list.append(normalize_transform)

    return transforms.Compose(pil_transforms_list + tensor_transforms_list)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Verwende Device: {device}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Projekt-Root: {project_root}")

    dataset_name_for_log = cfg.data.get("name", "unknown_dataset") # Für Logging
    tb_log_abs_dir = os.path.join(project_root, cfg.train.tb_log_dir, cfg.model.backbone, dataset_name_for_log + "_focal_loss")  # Geändert für Focal Loss
    os.makedirs(tb_log_abs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_abs_dir)
    print(f"TensorBoard Logs in: {tb_log_abs_dir}")

    # --- Patch- und Bildparameter ---
    patch_cfg = cfg.data.severstal.patch_processing
    if not patch_cfg.get("enabled", False):
        print("Patch-Verarbeitung nicht in Config aktiviert. Train.py erwartet Patch-Dataset.")
        writer.close()
        return
        
    patch_size_hw = tuple(patch_cfg.patch_size_hw)
    stride_hw = tuple(patch_cfg.stride_hw) # Für Dataset-Initialisierung
    max_neg_p = patch_cfg.get("max_neg_patches_per_image", 5)
    min_pos_p = patch_cfg.get("min_positive_pixel_percentage", 0.01)
    debug_orig_imgs = patch_cfg.get("debug_limit_images_patching", None)

    print(f"Patch-Parameter: Size={patch_size_hw}, Stride={stride_hw}")
    patch_h, patch_w = patch_size_hw[0], patch_size_hw[1]

    # --- Transformationen ---
    train_tf = get_transforms(cfg, img_h_config=patch_h, img_w_config=patch_w, is_train=True)
    val_tf = get_transforms(cfg, img_h_config=patch_h, img_w_config=patch_w, is_train=False)

    # --- Dataset und DataLoader für Severstal (Patch-basiert) ---
    severstal_img_dir = os.path.join(project_root, cfg.data.severstal.img_dir)
    severstal_ann_dir = os.path.join(project_root, cfg.data.severstal.ann_dir)

    print("Initialisiere Severstal Patch Dataset für Index-Splitting (mit train_tf)...")
    dataset_for_splitting_indices = SeverstalDataset(
        img_dir=severstal_img_dir, ann_dir=severstal_ann_dir,
        patch_size_hw=patch_size_hw, stride_hw=stride_hw, transform=train_tf,
        max_neg_patches_per_image=max_neg_p, min_positive_pixel_percentage=min_pos_p,
        debug_limit_images=debug_orig_imgs
    )

    if len(dataset_for_splitting_indices) == 0:
        print("Severstal Patch Dataset ist leer. Abbruch.")
        writer.close()
        return

    train_ratio = cfg.data.severstal.get("severstal_train_split_ratio", 0.8) # Aus Config holen
    n_total_patches = len(dataset_for_splitting_indices)
    n_train_patches = int(n_total_patches * train_ratio)
    n_val_patches = n_total_patches - n_train_patches

    print(f"Gesamtzahl extrahierter Patches: {n_total_patches}. Split: {n_train_patches} Training, {n_val_patches} Validierung.")
    
    train_indices, val_indices = random_split(
        range(n_total_patches), [n_train_patches, n_val_patches], # Teile die Indizes
        generator=torch.Generator().manual_seed(cfg.train.seed)
    )

    # Da `SeverstalDataset` das `transform` im Konstruktor nimmt und anwendet:
    train_dataset = torch.utils.data.Subset(dataset_for_splitting_indices, train_indices.indices)
    
    print("Initialisiere Val-Dataset-Instanz mit Validierungs-Transformationen...")
    val_dataset_instance_with_val_tf = SeverstalDataset(
        img_dir=severstal_img_dir, ann_dir=severstal_ann_dir,
        patch_size_hw=patch_size_hw, stride_hw=stride_hw, transform=val_tf, # HIER val_tf
        max_neg_patches_per_image=max_neg_p, min_positive_pixel_percentage=min_pos_p,
        debug_limit_images=debug_orig_imgs 
    )
    val_dataset = torch.utils.data.Subset(val_dataset_instance_with_val_tf, val_indices.indices)

    # Stelle sicher, dass SeverstalDataset in diesem Scope bekannt ist (sollte durch Import am Anfang der Datei der Fall sein)
    analyze_dataset_composition(train_dataset, "Trainingsset", dataset_for_splitting_indices.patches_info)
    analyze_dataset_composition(val_dataset, "Validierungsset", val_dataset_instance_with_val_tf.patches_info)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True # drop_last kann bei BCE helfen, wenn letzter Batch zu klein
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True
    )
    print(f"Train DataLoader: {len(train_loader)} Batches. Val DataLoader: {len(val_loader)} Batches.")

    # --- Klassen-Gewichte für Focal Loss berechnen ---
    print("Berechne Klassen-Gewichte für Focal Loss basierend auf dem Trainings-Subset der Patches...")
    num_labels = SeverstalDataset.NUM_TOTAL_CLASSES
    class_positive_counts_train = torch.zeros(num_labels, device=device)
    
    underlying_patch_dataset = train_dataset.dataset # Die Instanz von SeverstalDataset
    training_patch_indices = train_dataset.indices

    print(f"Iteriere durch {len(training_patch_indices)} Trainings-Patch-Labels für Klassen-Gewichte...")
    for i, patch_idx_in_original_list in enumerate(training_patch_indices):
        patch_info = underlying_patch_dataset.patches_info[patch_idx_in_original_list]
        label_vector = patch_info["label_vector"]
        class_positive_counts_train += label_vector.to(device)
        
        if i % 5000 == 0 and i > 0:
            print(f"  ... {i} Patch-Labels für Gewichte geprüft.")

    n_train_total_patches = len(train_dataset)
    pos_weight_tensor = torch.zeros(num_labels, device=device)
    for c in range(num_labels):
        num_positive_for_c = class_positive_counts_train[c]
        num_negative_for_c = n_train_total_patches - num_positive_for_c
        if num_positive_for_c > 0 and num_negative_for_c > 0:
            pos_weight_tensor[c] = num_negative_for_c / num_positive_for_c
        else:
            pos_weight_tensor[c] = 1.0
    
    # Optional: Clipping der pos_weights
    MAX_POS_WEIGHT = cfg.train.get("max_pos_weight", 30.0)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, max=MAX_POS_WEIGHT)
    print(f"Clipping pos_weights bei max={MAX_POS_WEIGHT}")

    print(f"Verwendete Trainings-Patches für Zählung: {n_train_total_patches}")
    print(f"Anzahl positiver Patch-Labels pro Klasse (0=NoDefect, ...): {class_positive_counts_train.cpu().numpy()}")
    print(f"Berechnete (und ggf. geclippte) pos_weights: {pos_weight_tensor.cpu().numpy()}")

    # --- Modell ---
    model = ClassifierModel(
        backbone_name=cfg.model.backbone, pretrained=cfg.model.pretrained,
        num_classes=SeverstalDataset.NUM_TOTAL_CLASSES
    ).to(device)
    print(f"Modell: {cfg.model.backbone} mit {SeverstalDataset.NUM_TOTAL_CLASSES} Ausgabeklassen geladen.")    # --- Loss-Funktion festlegen (Focal Loss oder BCEWithLogitsLoss) ---
    use_focal = cfg.train.get("use_focal", True)  # Prüfen, ob Focal Loss verwendet werden soll
    
    if use_focal:
        # Hole Focal Loss Parameter aus der Config
        focal_gamma = cfg.train.get("focal_gamma", 2.0)  # Default: 2.0
        focal_alpha = cfg.train.get("focal_alpha", 0.25) # Default: 0.25
        
        criterion = FocalLoss(
            alpha=focal_alpha, 
            gamma=focal_gamma, 
            pos_weight=pos_weight_tensor, 
            reduction='mean'
        )
        print(f"Verwende Verlustfunktion: Focal Loss mit Alpha={focal_alpha}, Gamma={focal_gamma} und berechneten pos_weights")
    else:
        # Fallback auf BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Verwende Verlustfunktion: BCEWithLogitsLoss mit berechneten pos_weights")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.max_epochs)
    print(f"Optimizer: AdamW, LR: {cfg.train.lr}, Scheduler: CosineAnnealingLR")

    # --- Trainings-Loop ---
    best_f1_micro_val = 0.0
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience, mode="max", verbose=True)

    print(f"\nStarte Training für {cfg.train.max_epochs} Epochen...")
    for epoch in range(cfg.train.max_epochs):
        model.train()
        train_loss_epoch = 0.0
        for batch_idx, (patches, labels, _) in enumerate(train_loader):
            patches, labels = patches.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            if (batch_idx + 1) % cfg.train.get("log_interval", 50) == 0:
                print(f"  Epoch {epoch+1}/{cfg.train.max_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss_epoch / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        print(f"Epoch {epoch+1} abgeschlossen. Avg Train Loss: {avg_train_loss:.4f}")

        # Validierung
        model.eval()
        val_loss_epoch = 0.0
        all_labels_val_list = [] # Um Listen von Tensoren zu sammeln
        all_probs_val_list = []
        
        with torch.no_grad():
            for patches, labels, _ in val_loader:
                patches = patches.to(device)
                labels = labels.to(device).float()
                outputs = model(patches)
                loss = criterion(outputs, labels)
                val_loss_epoch += loss.item()
                probs = torch.sigmoid(outputs)
                all_probs_val_list.append(probs.cpu())
                all_labels_val_list.append(labels.cpu())
        
        avg_val_loss = val_loss_epoch / len(val_loader)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        all_labels_val_np = torch.cat(all_labels_val_list).numpy()
        all_probs_val_np = torch.cat(all_probs_val_list).numpy()
        
        thresholds = np.arange(0.1, 0.91, 0.05)
        current_epoch_best_f1_micro = 0.0
        current_epoch_best_threshold = 0.5
        
        for t in thresholds:
            preds_val = (all_probs_val_np > t).astype(int)
            f1_t = f1_score(all_labels_val_np, preds_val, average='micro', zero_division=0)
            if f1_t > current_epoch_best_f1_micro:
                current_epoch_best_f1_micro = f1_t
                current_epoch_best_threshold = t
        
        final_preds_val = (all_probs_val_np > current_epoch_best_threshold).astype(int)
        precision_val_micro = precision_score(all_labels_val_np, final_preds_val, average='micro', zero_division=0)
        recall_val_micro = recall_score(all_labels_val_np, final_preds_val, average='micro', zero_division=0)
        f1_per_class_val = f1_score(all_labels_val_np, final_preds_val, average=None, zero_division=0)

        print(f"[Val] Epoch {epoch+1} - Loss: {avg_val_loss:.4f}, Best Thresh: {current_epoch_best_threshold:.2f} -> F1_micro: {current_epoch_best_f1_micro:.4f}, P_micro: {precision_val_micro:.4f}, R_micro: {recall_val_micro:.4f}")
        f1_class_strings = [f"{f1:.3f}" for f1 in f1_per_class_val]
        print(f"  F1 per class (0..4): [{', '.join(f1_class_strings)}]")

        writer.add_scalar("Val/F1_micro", current_epoch_best_f1_micro, epoch)
        writer.add_scalar("Val/Precision_micro", precision_val_micro, epoch)
        writer.add_scalar("Val/Recall_micro", recall_val_micro, epoch)
        writer.add_scalar("Val/Best_Threshold", current_epoch_best_threshold, epoch)
        for i, f1_class in enumerate(f1_per_class_val):
            writer.add_scalar(f"Val_F1_Class/Class_{i}", f1_class, epoch)

        if current_epoch_best_f1_micro > best_f1_micro_val:
            best_f1_micro_val = current_epoch_best_f1_micro
            best_model_ckpt_dir = os.path.join(project_root, cfg.train.ckpt_dir, cfg.model.backbone, dataset_name_for_log + "_focal")
            os.makedirs(best_model_ckpt_dir, exist_ok=True)
            best_model_path = os.path.join(best_model_ckpt_dir, f"best_model_severstal_patches_focal.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Neues bestes Modell gespeichert: {best_model_path} (F1_micro: {best_f1_micro_val:.4f})")
        
        if stopper.step(current_epoch_best_f1_micro):
            print("Early stopping ausgelöst.")
            break
        scheduler.step()

    print(f"\nTraining abgeschlossen. Bestes F1_micro auf Validierung: {best_f1_micro_val:.4f}")
    writer.close()
    print("TensorBoard Writer geschlossen.")

if __name__ == "__main__":
    main()
