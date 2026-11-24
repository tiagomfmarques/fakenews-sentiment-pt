import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

RESULTS_DIR = Path("src/results")
NOMES_CLASSES = ["true", "fake"]   # label 0 = verdadeira, 1 = falsa


def calcular_metricas(y_true, y_pred, prob_fake=None) -> dict:
    """Métricas de classificação binária (classe positiva = fake)."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if prob_fake is not None and len(set(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, prob_fake)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in m.items()}


def guardar_resultados(nome: str, config: dict, metricas_val: dict, metricas_teste: dict,
                       df_teste: pd.DataFrame, y_pred, prob_fake=None,
                       results_dir: Path = RESULTS_DIR):
    """Grava JSON de métricas, CSV de previsões e figura da matriz de confusão."""
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    saida = {
        "modelo": nome,
        "data": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "val": metricas_val,
        "test": metricas_teste,
    }
    caminho_json = results_dir / "metrics" / f"fakenews_{nome}.json"
    caminho_json.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    preds = df_teste[["doc_id", "pair_id", "label"]].copy()
    preds["pred"] = y_pred
    if prob_fake is not None:
        preds["prob_fake"] = np.round(prob_fake, 4)
    preds["correto"] = preds["label"] == preds["pred"]
    preds.to_csv(results_dir / "metrics" / f"fakenews_{nome}_preds.csv", index=False)

    guardar_matriz_confusao(metricas_teste["confusion_matrix"], nome,
                            results_dir / "figures" / f"fakenews_{nome}_cm.png")

    print(f"\n[{nome}] TESTE  acc={metricas_teste['accuracy']:.4f}  "
          f"f1_macro={metricas_teste['f1_macro']:.4f}  f1_fake={metricas_teste['f1_fake']:.4f}")
    print(f"Resultados gravados em {caminho_json}")


def guardar_matriz_confusao(cm, nome: str, caminho: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["verdadeira", "falsa"])
    ax.set_yticks([0, 1], labels=["verdadeira", "falsa"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(nome)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def carregar_split(versao: str, split: str, data_dir: str = "data/processed") -> pd.DataFrame:
    """Lê data/processed/fakebr_<versao>_<split>.csv (procura também em subpastas)."""
    nome = f"fakebr_{versao}_{split}.csv"
    candidatos = [Path(data_dir) / nome, *Path(data_dir).glob(f"*/{nome}")]
    for c in candidatos:
        if c.exists():
            return pd.read_csv(c)
    raise FileNotFoundError(f"Não encontrei {nome} em {data_dir} (nem em subpastas)")
