# project/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import hydra
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from omegaconf import DictConfig, ListConfig,  OmegaConf # ListConfig importiert
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score 
# multilabel_confusion_matrix können wir später für detailliertere Analyse hinzufügen

from project.datasets import SeverstalDataset, SeverstalStripDataset  # Wird jetzt als Patch-Dataset verwendet
from project.models import ClassifierModel
from project.utils import set_seed, EarlyStopping, FocalLoss
from collections import Counter, defaultdict 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


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


def get_transforms(cfg_augment_dict: dict, img_h_config: int, img_w_config: int, is_train: bool = True):
    """
    Erzeugt Transformationspipelines.
    'is_train' steuert, ob Augmentierungen angewendet werden.
    img_h_config, img_w_config sind jetzt die Patch-Dimensionen.
    """
    aug = cfg_augment_dict
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

    # Pfade zu Daten (angenommen, sie sind relativ zum Projekt-Root in der Config)
    img_dir_abs = os.path.join(project_root, cfg.data.severstal.img_dir)
    ann_dir_abs = os.path.join(project_root, cfg.data.severstal.ann_dir)

    # Tensorboard Setup
    dataset_name_for_log = cfg.data.get("name", "severstal")
    current_training_mode = cfg.data.severstal.get("training_mode", "patches")
    tb_log_subdir = f"{cfg.model.backbone}/{dataset_name_for_log}_{current_training_mode}"
    tb_log_abs_dir = os.path.join(project_root, cfg.train.tb_log_dir, tb_log_subdir)
    os.makedirs(tb_log_abs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_abs_dir)
    print(f"TensorBoard Logs in: {tb_log_abs_dir}")

    # Gemeinsame Debug-Limit für Bilder (aus Patch-Config nehmen oder eigenen für Strips definieren)
    # Hier nehmen wir an, patch_processing.debug_limit_images_patching gilt für beide Modi, wenn nichts anderes da ist.
    debug_orig_imgs = cfg.data.severstal.patch_processing.get("debug_limit_images_patching", None)
    if current_training_mode == "wide_strips":
        # Überschreibe mit spezifischem Limit für Strips, falls vorhanden
        debug_orig_imgs = cfg.data.severstal.wide_strip_processing.get("debug_limit_images_strips", debug_orig_imgs)


    train_dataset = None
    val_dataset = None
    num_labels_for_model = SeverstalDataset.NUM_TOTAL_CLASSES # Standard, wird ggf. überschrieben

    # --- Transformations-Konfiguration ---
    # get_transforms erwartet ein dict für cfg.augment, also extrahieren wir es.
    # OmegaConf.to_container konvertiert DictConfig zu python dicts/lists
    cfg_augment_dict = OmegaConf.to_container(cfg.augment, resolve=True) if cfg.augment else {}


    if current_training_mode == "patches":
        print("\n--- Modus: Training mit Patches ---")
        patch_cfg = cfg.data.severstal.patch_processing
        if not patch_cfg.get("enabled", False):
            print("Patch-Verarbeitung nicht aktiviert. Abbruch.")
            writer.close(); return
            
        patch_size_hw = tuple(patch_cfg.patch_size_hw)
        stride_hw = tuple(patch_cfg.stride_hw)
        max_neg_p = patch_cfg.get("max_neg_patches_per_image", 5)
        min_pos_p = patch_cfg.get("min_positive_pixel_percentage", 0.01)
        
        print(f"Patch-Parameter: Size={patch_size_hw}, Stride={stride_hw}")
        patch_h, patch_w = patch_size_hw[0], patch_size_hw[1]

        train_tf = get_transforms(cfg_augment_dict, img_h_config=patch_h, img_w_config=patch_w, is_train=True)
        val_tf = get_transforms(cfg_augment_dict, img_h_config=patch_h, img_w_config=patch_w, is_train=False)

        print("Initialisiere SeverstalDataset für Patch-Splitting...")
        dataset_for_splitting_indices = SeverstalDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs,
            patch_size_hw=patch_size_hw, stride_hw=stride_hw, transform=train_tf, # transform hier für Subset wichtig
            max_neg_patches_per_image=max_neg_p, min_positive_pixel_percentage=min_pos_p,
            debug_limit_images=debug_orig_imgs
        )
        if len(dataset_for_splitting_indices) == 0:
            print("Severstal Patch Dataset ist leer. Abbruch."); writer.close(); return

        n_total_items = len(dataset_for_splitting_indices)
        n_train_items = int(n_total_items * cfg.data.severstal.severstal_train_split_ratio)
        n_val_items = n_total_items - n_train_items
        print(f"Gesamtzahl extrahierter Patches: {n_total_items}. Split: {n_train_items} Training, {n_val_items} Validierung.")
        
        train_indices, val_indices_obj = random_split( # val_indices ist ein Dataset-Objekt, wir brauchen .indices
            range(n_total_items), [n_train_items, n_val_items],
            generator=torch.Generator().manual_seed(cfg.train.seed)
        )
        train_dataset = Subset(dataset_for_splitting_indices, train_indices.indices)
        
        # Val-Dataset mit val_tf
        val_dataset_instance_with_val_tf = SeverstalDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs,
            patch_size_hw=patch_size_hw, stride_hw=stride_hw, transform=val_tf,
            max_neg_patches_per_image=max_neg_p, min_positive_pixel_percentage=min_pos_p,
            debug_limit_images=debug_orig_imgs 
        )
        val_dataset = Subset(val_dataset_instance_with_val_tf, val_indices_obj.indices)

        # Analyse der Patch-Dataset-Zusammensetzung
        analyze_dataset_composition(train_dataset, "Trainingsset (Patches)", dataset_for_splitting_indices.patches_info)
        analyze_dataset_composition(val_dataset, "Validierungsset (Patches)", val_dataset_instance_with_val_tf.patches_info)
        
        # pos_weight Berechnung für Patches
        print("Berechne pos_weight für Patch-Modus...")
        num_labels_for_model = SeverstalDataset.NUM_TOTAL_CLASSES
        class_positive_counts_train = torch.zeros(num_labels_for_model, device=device)
        # ... (deine bestehende pos_weight Logik für Patches, die patches_info verwendet) ...
        underlying_patch_dataset = train_dataset.dataset 
        training_patch_indices = train_dataset.indices
        for i, patch_idx_in_original_list in enumerate(training_patch_indices):
            patch_info = underlying_patch_dataset.patches_info[patch_idx_in_original_list]
            label_vector = patch_info["label_vector"] 
            class_positive_counts_train += label_vector.to(device)
        
        n_train_total_items = len(train_dataset)


    elif current_training_mode == "wide_strips":
        print("\n--- Modus: Training mit Wide Strips ---")
        strip_cfg = cfg.data.severstal.wide_strip_processing
        if not strip_cfg.get("enabled", False):
            print("Wide Strip-Verarbeitung nicht aktiviert. Abbruch.")
            writer.close(); return

        strip_h = strip_cfg.strip_height
        # Für Streifen ist die Breite oft variabel oder eine feste große Breite
        # Das Modell (insb. avgpool) muss damit umgehen können oder Resize in get_transforms muss es handhaben
        strip_w_config_for_transform = strip_cfg.get("random_crop_width", -1) # -1 bedeutet variable Breite / nur Höhe resizen

        train_tf = get_transforms(cfg_augment_dict, img_h_config=strip_h, img_w_config=strip_w_config_for_transform, is_train=True)
        val_tf = get_transforms(cfg_augment_dict, img_h_config=strip_h, img_w_config=strip_w_config_for_transform, is_train=False)

        # Da SeverstalStripDataset die Augmentierung (inkl. Blackout) in __getitem__ macht,
        # teilen wir die Liste der Bilddateien.
        # Erstelle eine temporäre Instanz, um die gefilterte Liste der Bilddateien zu bekommen.
        # Wichtig: Hier wird angenommen, dass img_dir und ann_dir für wide_strips nur Bilder enthalten,
        # die bereits Defekte haben (deine Vorfilterung).
        temp_strip_dataset_for_filelist = SeverstalStripDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs,
            strip_height=strip_h, # Minimale Parameter
            debug_limit_images=debug_orig_imgs
        )
        all_strip_image_files = temp_strip_dataset_for_filelist.image_files # Liste der Strings (Dateinamen)
        
        if not all_strip_image_files:
            print("Keine Bilder im (ggf. debug-limitierten) Strip-Dataset gefunden. Abbruch."); writer.close(); return

        n_total_items = len(all_strip_image_files)
        n_train_items = int(n_total_items * cfg.data.severstal.severstal_train_split_ratio)
        n_val_items = n_total_items - n_train_items
        print(f"Gesamtzahl Originalbilder für Streifen: {n_total_items}. Split: {n_train_items} Training, {n_val_items} Validierung.")

        # Teile die Indizes der Dateiliste
        indices_for_file_split = list(range(n_total_items))
        train_file_indices_list, val_file_indices_list = random_split(
            indices_for_file_split, [n_train_items, n_val_items],
            generator=torch.Generator().manual_seed(cfg.train.seed)
        )

        train_image_filenames_to_use = [all_strip_image_files[i] for i in train_file_indices_list.indices]
        val_image_filenames_to_use = [all_strip_image_files[i] for i in val_file_indices_list.indices]
        # Erstelle die finalen Dataset-Instanzen. Hierfür muss SeverstalStripDataset
        # eine optionale Liste von zu verwendenden Bilddateinamen im __init__ akzeptieren.
        # ANNAHME: Du hast SeverstalStripDataset.__init__ so angepasst, dass es einen
        # Parameter `image_filenames_to_use: Optional[List[str]] = None` akzeptiert.
        # Wenn nicht, muss die Logik hier anders sein (z.B. zwei volle Instanzen und Subset).
        
        # Baue die Listen der zu verwendenden Dateinamen
        train_image_filenames_to_use = [all_strip_image_files[i] for i in train_file_indices_list.indices]
        val_image_filenames_to_use = [all_strip_image_files[i] for i in val_file_indices_list.indices]

        # Parameter für SeverstalStripDataset aus Config holen
        use_blackout = strip_cfg.get("use_defect_blackout", False)
        blackout_prob = strip_cfg.get("defect_blackout_prob", 0.5)
        target_no_defect = strip_cfg.get("target_no_defect_ratio", None)
        instance_blackout_p = strip_cfg.get("instance_blackout_prob_selective", 0.5)
        blackout_min_px = strip_cfg.get("blackout_min_pixels", 10)
        verbose_strip_debug = strip_cfg.get("verbose_strip_dataset_debug", False)

        print(f"\nErstelle Trainings-Dataset für Streifen mit {len(train_image_filenames_to_use)} spezifischen Bildern...")
        
        train_dataset = SeverstalStripDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs,
            strip_height=strip_h, transform=train_tf,
            use_defect_blackout=use_blackout, 
            defect_blackout_prob=blackout_prob, # Globale Prob.
            target_no_defect_ratio=target_no_defect,
            instance_blackout_prob_selective=instance_blackout_p,
            blackout_min_pixels=blackout_min_px,
            verbose_strip_dataset_debug=verbose_strip_debug,
            image_filenames_to_use=train_image_filenames_to_use # << ÜBERGABE DER LISTE
            # debug_limit_images wird hier nicht mehr explizit benötigt, da die Liste bereits final ist
        )
        
        print(f"\nErstelle Validierungs-Dataset für Streifen mit {len(val_image_filenames_to_use)} spezifischen Bildern...")
        val_dataset = SeverstalStripDataset(
            img_dir=img_dir_abs, ann_dir=ann_dir_abs,
            strip_height=strip_h, transform=val_tf,
            use_defect_blackout=False, # Wichtig: KEIN Blackout für Validierung
            # Die folgenden Blackout-Parameter sind für val_dataset irrelevant, wenn use_defect_blackout=False
            defect_blackout_prob=0.0, 
            target_no_defect_ratio=None,
            instance_blackout_prob_selective=0.0,
            blackout_min_pixels=blackout_min_px, # Kann bleiben, wird nicht genutzt
            verbose_strip_dataset_debug=verbose_strip_debug, # Kann für Debugging an bleiben
            image_filenames_to_use=val_image_filenames_to_use # << ÜBERGABE DER LISTE
        )

        # pos_weight Berechnung für Streifen (basierend auf Originalbild-Labels im Trainingsset)
        # Diese Logik muss die Annotationen der Bilder im train_image_filenames_to_use analysieren.
        print("Berechne pos_weight für Streifen-Modus (basierend auf Original-Annotationen der Trainingsbilder)...")
        num_labels_for_model = SeverstalStripDataset.NUM_TOTAL_CLASSES
        class_positive_counts_train = torch.zeros(num_labels_for_model, device=device)
        
        # Temporäre Instanz oder Hilfsfunktion, um an die Original-Labels der Trainingsbilder zu kommen
        # Da die Labels in SeverstalStripDataset dynamisch sind, brauchen wir die statischen Original-Bild-Labels
        for img_filename_in_train in train_image_filenames_to_use:
            # Lade die Annotation für dieses Bild und bestimme seine Defektklassen
            # Dies ist eine vereinfachte Annahme, dass _load_combined_gt_mask
            # direkt die Defekte zählt oder wir eine ähnliche Funktion haben.
            # Für eine genaue Zählung müsste _load_combined_gt_mask hier verwendet werden
            # um die Maske zu laden und dann die unique classes zu zählen.
            
            # Provisorische Zählung (besser: _load_combined_gt_mask verwenden und auswerten)
            temp_mask = temp_strip_dataset_for_filelist._load_combined_gt_mask(
                img_filename_in_train, 
                256, # Dummy Höhe/Breite, da wir nur die Klassen aus JSON wollen
                1600
            )
            temp_label_vector = torch.zeros(num_labels_for_model)
            unique_classes_in_img = np.unique(temp_mask)
            has_defect_in_img = False
            for class_val in unique_classes_in_img:
                if 1 <= class_val <= 4 and np.sum(temp_mask == class_val) > 0 : # Mindestens 1 Pixel
                    temp_label_vector[class_val] = 1.0
                    has_defect_in_img = True
            if not has_defect_in_img: # Sollte nicht passieren wegen Vorfilterung
                 temp_label_vector[SeverstalStripDataset.NO_DEFECT_IDX] = 1.0 # Als Fallback
            
            class_positive_counts_train += temp_label_vector.to(device)
        
        n_train_total_items = len(train_dataset) # Anzahl der Trainingsbilder/-streifen


    else: # Unbekannter Modus
        print(f"FEHLER: Unbekannter training_mode: {current_training_mode}")
        writer.close(); return

    if train_dataset is None or val_dataset is None or len(train_dataset) == 0:
        print("Fehler bei der Dataset-Erstellung oder leeres Dataset. Abbruch.")
        writer.close(); return
        
    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, # Shuffle ist wichtig!
        num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True
    )
    print(f"Train DataLoader: {len(train_loader)} Batches ({len(train_dataset)} Samples). Val DataLoader: {len(val_loader)} Batches ({len(val_dataset)} Samples).")


    # --- `pos_weight` oder `alpha` für Focal Loss Berechnung ---
    print(f"\nBerechne Gewichte für Verlustfunktion (Modus: {current_training_mode})...")
    # class_positive_counts_train und n_train_total_items werden im jeweiligen Modus-Block oben gesetzt.
    
    pos_weight_tensor = None
    focal_loss_alpha_tensor = None

    if n_train_total_items > 0:
        if cfg.train.loss_function == "bce":
            pos_weight_tensor = torch.zeros(num_labels_for_model, device=device)
            for c in range(num_labels_for_model):
                num_positive_for_c = class_positive_counts_train[c].item() # .item() für Skalar
                num_negative_for_c = n_train_total_items - num_positive_for_c
                if num_positive_for_c > 0 and num_negative_for_c > 0:
                    pos_weight_tensor[c] = num_negative_for_c / num_positive_for_c
                else:
                    pos_weight_tensor[c] = 1.0
            
            MAX_POS_WEIGHT = cfg.train.get("max_pos_weight", 30.0)
            pos_weight_tensor = torch.clamp(pos_weight_tensor, min=1.0, max=MAX_POS_WEIGHT)
            print(f"Für BCE: Berechnete pos_weights (geclippt max={MAX_POS_WEIGHT}, min=1.0): {pos_weight_tensor.cpu().numpy()}")

        elif cfg.train.loss_function == "focal":
            focal_cfg = cfg.train.focal_loss
            if focal_cfg.alpha_mode == "calculated":
                # Berechne Alpha ähnlich zu pos_weight, aber oft wird alpha direkt als
                # Gewicht für die positive Klasse verwendet. Eine gängige Methode ist:
                # alpha[c] = (Anteil der negativen Samples für Klasse c) / (Gesamtanteil) ODER
                # alpha[c] = (1 - Anteil der positiven Samples für Klasse c)
                # Hier verwenden wir eine einfache Umkehrung der Häufigkeit, normalisiert,
                # sodass seltenere Klassen ein höheres Alpha bekommen (bis zu ~0.99)
                # und häufigere ein niedrigeres (bis zu ~0.01), oft wird 0.25 für häufig, 0.75 für selten genutzt.
                # Für Multi-Label ist die Alpha-Definition komplexer.
                # Einfacher Ansatz: (1 - (positive_counts / total_samples)), dann clippen.
                # Oder: alpha_c = N_neg / (N_pos + N_neg)
                # Wichtig: FocalLoss erwartet alpha oft als Gewicht für die positive Klasse.
                # Wenn alpha[i] das Gewicht für Klasse i ist.
                
                # Simplifizierter Ansatz für Alpha (ähnlich wie pos_weight, aber anders skaliert)
                # Man könnte hier auch die `pos_weight_tensor` Berechnung nehmen und dann
                # alpha = pos_weight / (1 + pos_weight) ableiten oder eine andere Heuristik.
                # Fürs Erste: Deaktiviere Alpha-Berechnung und verwende None, es sei denn, es ist "fixed"
                # alpha_values = [] 
                # for c in range(num_labels_for_model):
                #     pos_c = class_positive_counts_train[c].item()
                #     neg_c = n_train_total_items - pos_c
                #     # alpha_c = neg_c / n_train_total_items if n_train_total_items > 0 else 0.5 # Gewichtet positive Klasse
                #     # Für Focal Loss wird alpha oft als Gewicht für die positive Klasse verstanden
                #     # Ein häufigerer Wert ist 0.25 für die "einfache" Klasse und 0.75 für die "schwere/seltene" Klasse.
                #     # Da wir mehrere Klassen haben, ist ein Vektor nötig.
                #     # Hier ein Beispiel, das seltenere Klassen höher gewichtet:
                #     if pos_c > 0:
                #         alpha_val = 1.0 - (pos_c / n_train_total_items) # Höher für seltenere Klassen
                #         alpha_values.append(min(max(alpha_val, 0.01), 0.99)) # Clippen
                #     else:
                #         alpha_values.append(0.99) # Sehr hohes Alpha, wenn Klasse nie vorkommt
                # focal_loss_alpha_tensor = torch.tensor(alpha_values, device=device, dtype=torch.float32)
                
                # Alternativ: Wenn `alpha` in FocalLoss ein Skalar sein soll (oft als Gewicht für positive Klasse)
                # oder ein Tensor der Klassengewichte. Für Multi-Label wird alpha oft pro Klasse gesetzt.
                # Wenn alpha=None, wird keine Alpha-Gewichtung angewendet.
                focal_loss_alpha_tensor = None # Standardmäßig keine Alpha-Gewichtung in FocalLoss
                print("Für Focal Loss: Alpha-Berechnung (calculated) noch nicht implementiert/verfeinert. Verwende alpha=None.")

            elif focal_cfg.alpha_mode == "fixed":
                if focal_cfg.get("alpha_fixed") is not None:
                    alpha_list = OmegaConf.to_container(focal_cfg.alpha_fixed, resolve=True)
                    if len(alpha_list) == num_labels_for_model:
                        focal_loss_alpha_tensor = torch.tensor(alpha_list, device=device, dtype=torch.float32)
                        print(f"Für Focal Loss: Verwende feste Alpha-Werte: {focal_loss_alpha_tensor.cpu().numpy()}")
                    else:
                        print(f"WARNUNG: Länge von focal_loss.alpha_fixed ({len(alpha_list)}) "
                              f"stimmt nicht mit num_classes ({num_labels_for_model}) überein. Verwende alpha=None.")
                        focal_loss_alpha_tensor = None
                else:
                    focal_loss_alpha_tensor = None # Kein alpha_fixed Wert in Config
            else: # alpha_mode = None oder unbekannt
                focal_loss_alpha_tensor = None
            
            if focal_loss_alpha_tensor is None:
                 print("Für Focal Loss: Keine Alpha-Gewichtung wird verwendet.")


    else: # Kein Trainings-Set
        print("WARNUNG: Kein Trainings-Set für Gewichts-Berechnung vorhanden.")
        # Standard-Gewichte, falls das Training trotzdem gestartet wird (sollte nicht passieren)
        if cfg.train.loss_function == "bce":
            pos_weight_tensor = torch.ones(num_labels_for_model, device=device)
        # Für Focal Loss bleibt alpha dann None

    print(f"Verwendete Trainings-Samples für Zählung: {n_train_total_items}")
    print(f"Anzahl positiver Labels pro Klasse (basierend auf Modus): {class_positive_counts_train.cpu().numpy()}")


    # --- Modell, Verlustfunktion, Optimizer (gemeinsam) ---
    model = ClassifierModel(
        backbone_name=cfg.model.backbone, pretrained=cfg.model.pretrained,
        num_classes=num_labels_for_model # NUM_TOTAL_CLASSES der jeweiligen Dataset-Klasse
    ).to(device)
    print(f"Modell: {cfg.model.backbone} mit {num_labels_for_model} Ausgabeklassen geladen.")

    # --- Verlustfunktion basierend auf Config ---
    criterion = None
    if cfg.train.loss_function == "focal":
        focal_cfg = cfg.train.focal_loss
        criterion = FocalLoss(
            alpha=focal_loss_alpha_tensor, # Kann None sein
            gamma=focal_cfg.get("gamma", 2.0)
        )
        print(f"Verwende Verlustfunktion: FocalLoss (gamma={criterion.gamma}, alpha={criterion.alpha.cpu().numpy() if criterion.alpha is not None else 'None'})")
    elif cfg.train.loss_function == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Verwende Verlustfunktion: BCEWithLogitsLoss (pos_weight={pos_weight_tensor.cpu().numpy() if pos_weight_tensor is not None else 'None'})")
    else:
        print(f"FEHLER: Unbekannte loss_function: {cfg.train.loss_function}. Verwende BCEWithLogitsLoss als Fallback.")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) # Fallback

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
                loss = criterion(outputs, labels) # Kann auch für Val geloggt werden
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
            best_model_ckpt_dir = os.path.join(project_root, cfg.train.ckpt_dir, cfg.model.backbone, dataset_name_for_log)
            os.makedirs(best_model_ckpt_dir, exist_ok=True)
            best_model_path = os.path.join(best_model_ckpt_dir, f"best_model_severstal_patches.pt")
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