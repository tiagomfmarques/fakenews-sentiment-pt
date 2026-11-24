import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import RESULTS_DIR

COLUNAS_POR_TAREFA = {
    "fakenews": ["accuracy", "precision_fake", "recall_fake", "f1_fake", "f1_macro", "roc_auc"],
    "sentiment": ["f1_micro", "f1_macro", "f1_weighted", "precision_micro", "recall_micro",
                  "exact_match"],
}
TITULOS = {"fakenews": ("Deteção de notícias falsas", "Resultados no conjunto de teste do Fake.Br"),
           "sentiment": ("Reconhecimento de emoções (28 classes)",
                         "Resultados no conjunto de teste do GoEmotions PT")}


def carregar_resultados(tarefa: str = "fakenews", results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    COLUNAS = COLUNAS_POR_TAREFA[tarefa]
    linhas = []
    for f in sorted((results_dir / "metrics").glob(f"{tarefa}_*.json")):
        if f.stem.endswith(("_palavras", "_comparacao")):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        if "test" not in r:
            continue
        cfg = r.get("config", {})
        linha = {"modelo": r["modelo"], "versao": cfg.get("versao", "-")}
        linha.update({c: r["test"].get(c) for c in COLUNAS})
        linha["f1_macro_val"] = r.get("val", {}).get("f1_macro")
        linhas.append(linha)
    if not linhas:
        raise SystemExit("Nenhum resultado encontrado. Treina primeiro os modelos.")
    return pd.DataFrame(linhas).sort_values(["versao", "f1_macro"], ascending=[False, False])


def grafico(df: pd.DataFrame, caminho: Path, titulo: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.sort_values("f1_macro")
    # com duas versões de texto (full/normalized), a segunda fica a cinzento
    principal = df["versao"].mode().iat[0]
    cores = ["#4C72B0" if v == principal else "#A0A0A0" for v in df["versao"]]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(df) + 1.2))
    barras = ax.barh(df["modelo"], df["f1_macro"], color=cores)
    ax.bar_label(barras, fmt="%.3f", padding=3)
    ax.set_xlim(max(0, df["f1_macro"].min() - 0.1), 1.0)
    ax.set_xlabel("F1 macro (teste)")
    ax.set_title(f"{titulo} — comparação de modelos")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", choices=list(COLUNAS_POR_TAREFA), default="fakenews")
    t = ap.parse_args().tarefa
    df = carregar_resultados(t)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False, float_format="%.4f"))

    pasta_m, pasta_f = RESULTS_DIR / "metrics", RESULTS_DIR / "figures"
    pasta_f.mkdir(parents=True, exist_ok=True)
    df.to_csv(pasta_m / f"{t}_comparacao.csv", index=False)
    df.to_latex(pasta_m / f"{t}_comparacao.tex", index=False, float_format="%.3f", escape=True,
                caption=TITULOS[t][1], label=f"tab:{t}_resultados")
    grafico(df, pasta_f / f"{t}_comparacao.png", TITULOS[t][0])
    print(f"\nGravado: {pasta_m / f'{t}_comparacao.csv'}, .tex e "
          f"{pasta_f / f'{t}_comparacao.png'}")


if __name__ == "__main__":
    main()
