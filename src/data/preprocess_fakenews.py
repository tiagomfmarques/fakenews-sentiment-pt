"""
Pré-processamento do Fake.Br Corpus: ficheiros .txt brutos -> CSV estruturados.

Lê as versões `full_texts` e/ou `size_normalized_texts` (as que existirem),
faz uma limpeza leve (adequada a modelos BERT), divide em treino/validação/teste
POR PARES (fake/N e true/N ficam sempre do mesmo lado) e grava em data/processed/.

Uso (a partir da raiz do projeto):
    python -m src.data.preprocess
    python -m src.data.preprocess --raw data/raw/Fake.br-Corpus --out data/processed
"""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

VERSOES = {"full": "full_texts", "normalized": "size_normalized_texts"}
LABELS = {"true": 0, "fake": 1}
SEED = 42


def ler_texto(caminho: Path) -> str:
    """Lê um .txt tolerando BOM e eventuais ficheiros em latin-1."""
    dados = caminho.read_bytes()
    try:
        return dados.decode("utf-8-sig")
    except UnicodeDecodeError:
        return dados.decode("latin-1")


def limpar_texto(texto: str) -> str:
    """Limpeza leve: mantém acentos, maiúsculas e pontuação (o BERT precisa deles)."""
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = texto.replace(" ", " ").replace("​", "")
    texto = re.sub(r"[ \t]+", " ", texto)          # espaços repetidos
    texto = re.sub(r" *\n *", "\n", texto)          # espaços à volta de quebras
    texto = re.sub(r"\n{2,}", "\n", texto)          # linhas em branco repetidas
    return texto.strip()


def carregar_versao(raiz: Path, versao: str) -> pd.DataFrame:
    pasta = raiz / VERSOES[versao]
    linhas = []
    for nome_label, label in LABELS.items():
        for f in sorted((pasta / nome_label).glob("*.txt"), key=lambda p: int(p.stem)):
            texto = limpar_texto(ler_texto(f))
            linhas.append({
                "pair_id": int(f.stem),
                "label": label,
                "label_name": nome_label,
                "text": texto,
                "n_words": len(texto.split()),
                "n_chars": len(texto),
            })
    df = pd.DataFrame(linhas)
    df.insert(0, "doc_id", df["label_name"] + "_" + df["pair_id"].astype(str))
    return df


def dividir_por_pares(pair_ids, val=0.1, test=0.1):
    """Devolve {pair_id: 'train'|'val'|'test'} — a mesma divisão serve para todas as versões."""
    pair_ids = sorted(set(pair_ids))
    treino, resto = train_test_split(pair_ids, test_size=val + test, random_state=SEED)
    val_ids, test_ids = train_test_split(resto, test_size=test / (val + test), random_state=SEED)
    split = {p: "train" for p in treino}
    split.update({p: "val" for p in val_ids})
    split.update({p: "test" for p in test_ids})
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/fakenews/Fake.br-Corpus")
    ap.add_argument("--out", default="data/processed/fakenews")
    args = ap.parse_args()

    raiz, saida = Path(args.raw), Path(args.out)
    saida.mkdir(parents=True, exist_ok=True)

    versoes = [v for v, pasta in VERSOES.items() if (raiz / pasta / "fake").exists()]
    if not versoes:
        raise SystemExit(f"Nenhuma versão encontrada em {raiz}")

    dados = {v: carregar_versao(raiz, v) for v in versoes}

    # As duas versões não têm exatamente os mesmos números de pares
    # (full não tem 697/1468; normalized não tem 586/1607), por isso a divisão
    # é feita sobre a união dos pair_id -> mesma divisão para todas as versões.
    todos_ids = set().union(*(d["pair_id"] for d in dados.values()))
    split = dividir_por_pares(todos_ids)

    for versao, df in dados.items():
        vazios = df["text"].str.len() == 0
        if vazios.any():
            print(f"[{versao}] aviso: {vazios.sum()} textos vazios removidos")
            df = df[~vazios].copy()

        df["split"] = df["pair_id"].map(split)

        df.to_csv(saida / f"fakebr_{versao}.csv", index=False)
        for s in ("train", "val", "test"):
            df[df["split"] == s].drop(columns="split").to_csv(
                saida / f"fakebr_{versao}_{s}.csv", index=False)

        print(f"\n[{versao}] {len(df)} textos")
        print(df.groupby(["split", "label_name"]).size().unstack())
        print("palavras (mediana):", df.groupby("label_name")["n_words"].median().to_dict())


if __name__ == "__main__":
    main()