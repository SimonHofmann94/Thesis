import os
import numpy as np
import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

from project.datasets import BSDataset
from project.models import ClassifierModel
from project.utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("cwd", lambda: os.getcwd())


@hydra.main(config_path="../config", config_name="config",version_base=None)
def main(cfg: DictConfig):
    """
    Evaluation-Skript für das Klassifikationsmodell (BSData binary) auf dem echten Test-Set.
    """
    # Reproduzierbarkeit
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    # Pfade
    root = hydra.utils.get_original_cwd()
    data_dir  = os.path.join(root, cfg.data.bsdata.data_dir)
    label_dir = os.path.join(root, cfg.data.bsdata.label_dir)
    ckpt_path = os.path.join(root, cfg.train.ckpt_dir, f"model_{cfg.model.backbone}.pth")

    # Transform für Test-Set
    test_tf = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset laden
    full_bs = BSDataset(
        data_dir=data_dir,
        label_dir=label_dir,
        transform=test_tf
    )

    # Stratifiziertes Test-Subset mit Klassenbalance 50/50
    labels = [full_bs[i][1] for i in range(len(full_bs))]
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cfg.data.bsdata.test_split,
        random_state=cfg.train.seed
    )
    _, test_idx = next(sss.split(range(len(full_bs)), labels))
    bs_test     = Subset(full_bs, test_idx)

    test_loader = DataLoader(
        bs_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )
    # Modell laden
    model = ClassifierModel(
        backbone_name=cfg.model.backbone,
        pretrained=False,
        num_classes=cfg.model.num_classes
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Nur binary supported
    if cfg.model.num_classes != 2:
        print("Multiclass-Evaluation noch nicht implementiert. (Nur binary unterstützt.)")
        return

    # Inferenz auf Test-Set
    y_true, y_pred, y_prob = [], [], []
    # Konfigurierbarer Threshold – hier aus Val-Evaluation übernehmen!
    best_threshold = cfg.evaluation.threshold  # ← kannst du ggf. als cfg.eval.threshold setzen
    
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # Wahrsch. für Klasse "positiv"
    
            y_true.extend(labels.numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    
    # Threshold aus config
    best_threshold = cfg.evaluation.threshold
    y_pred = [1 if p > best_threshold else 0 for p in y_prob]

    # Grundmetriken
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    # Klassifikationsergebnisse ausgeben
    print("\n=== Test Set Evaluation Results ===")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Optional: Threshold Sweep zur Diagnose
    print("\n=== Threshold Sweep ===")
    all_labels = np.array(y_true)
    all_probs = np.array(y_prob)
    best_f1, best_thresh = 0.0, 0.5
    best_p, best_r = 0.0, 0.0

    for t in np.linspace(0.1, 0.9, 17):
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds)
        p = precision_score(all_labels, preds)
        r = recall_score(all_labels, preds)
        print(f"Threshold: {t:.2f} | F1: {f1:.4f} | P: {p:.4f} | R: {r:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_p = p
            best_r = r

    print(f"\n=== Best Threshold on Test (diagnostisch) ===")
    print(f"Threshold: {best_thresh:.2f}")
    print(f"F1       : {best_f1:.4f}")
    print(f"Precision: {best_p:.4f}")
    print(f"Recall   : {best_r:.4f}")

    # Score-Verteilung plotten
    import matplotlib.pyplot as plt
    pos_probs = [p for p, y in zip(y_prob, y_true) if y == 1]
    neg_probs = [p for p, y in zip(y_prob, y_true) if y == 0]

    plt.figure(figsize=(10, 6))
    plt.hist(pos_probs, bins=40, alpha=0.6, label="Positive (Defekt)", color="crimson")
    plt.hist(neg_probs, bins=40, alpha=0.6, label="Negative (kein Defekt)", color="steelblue")
    plt.axvline(x=best_threshold, color='black', linestyle='--', label=f"Threshold {best_threshold:.2f}")
    plt.xlabel("Modell-Score für Klasse 'Defekt'")
    plt.ylabel("Häufigkeit")
    plt.title("Score-Verteilung im Testset")
    plt.legend()
    plt.grid(True)

    scoreplot_path = os.path.join(root, cfg.train.ckpt_dir, "test_score_distribution.png")
    os.makedirs(os.path.dirname(scoreplot_path), exist_ok=True)
    plt.savefig(scoreplot_path)
    print(f"Score-Verteilung gespeichert unter: {scoreplot_path}")

    # Optionale Speicherung der Wahrscheinlichkeiten
    np.save("outputs/y_prob_test.npy", np.array(y_prob))
    np.save("outputs/y_true_test.npy", np.array(y_true))


 
    # Metriken berechnen
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    roc_auc = roc_auc_score(y_true, y_prob)
    cm      = confusion_matrix(y_true, y_pred)

    # Ergebnisse ausgeben
    print("\n=== Test Set Evaluation Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
