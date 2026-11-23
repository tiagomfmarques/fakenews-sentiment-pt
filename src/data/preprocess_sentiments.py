"""
Pré-processamento do GoEmotions PT-BR (antoniomenezes/go_emotions_ptbr):
CSV "raw" (uma linha por anotador) -> CSV estruturados (uma linha por comentário).

Passos:
  1. junta goemotions_1_pt.csv, _2 e _3
  2. remove anotações marcadas como "very unclear"
  3. agrega por comentário: uma emoção conta se >= 2 anotadores a marcarem
     (mesmo critério dos autores do GoEmotions)
  4. mantém as 27 emoções + neutro SEPARADAS (28 classes, nomes em português)
  5. divide em treino/validação/teste (80/10/10)
  6. guarda duas versões com a MESMA divisão:
       sentiments_*.csv         -> multi-etiqueta (um texto pode ter várias emoções)
       sentiments_simples_*.csv -> só textos com uma emoção (para comparação)

Uso (a partir da raiz do projeto):
    python -m src.data.preprocess_emocoes
    python -m src.data.preprocess_emocoes --raw data/raw/goemotions_pt --out data/processed
"""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
MIN_ANOTADORES = 2

# Ordem original das colunas de emoção no GoEmotions raw
EMOCOES_GO = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
COLUNAS = (["text", "id", "author", "subreddit", "link_id", "parent_id",
            "created_utc", "rater_id", "example_very_unclear"]
           + EMOCOES_GO + ["texto"])

# Nome em português de cada uma das 28 classes (27 emoções + neutro), sem agrupar
NOMES_PT = {
    "admiration": "admiração", "amusement": "diversão", "anger": "raiva",
    "annoyance": "irritação", "approval": "aprovação", "caring": "cuidado",
    "confusion": "confusão", "curiosity": "curiosidade", "desire": "desejo",
    "disappointment": "desilusão", "disapproval": "desaprovação", "disgust": "nojo",
    "embarrassment": "vergonha", "excitement": "entusiasmo", "fear": "medo",
    "gratitude": "gratidão", "grief": "luto", "joy": "alegria", "love": "amor",
    "nervousness": "nervosismo", "optimism": "otimismo", "pride": "orgulho",
    "realization": "constatação", "relief": "alívio", "remorse": "remorso",
    "sadness": "tristeza", "surprise": "surpresa", "neutral": "neutro",
}
CLASSES = [NOMES_PT[e] for e in EMOCOES_GO]


def ler_csv(caminho: Path) -> pd.DataFrame:
    """Lê um CSV com ou sem cabeçalho e devolve-o com os nomes de COLUNAS."""
    df = pd.read_csv(caminho, header=None, dtype=str, keep_default_na=False)
    if df.iloc[0, 1] == "id":            # primeira linha é cabeçalho
        df = df.iloc[1:]
    if df.shape[1] != len(COLUNAS):
        raise SystemExit(f"{caminho.name}: {df.shape[1]} colunas, esperava {len(COLUNAS)}")
    df.columns = COLUNAS
    return df


def limpar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFC", str(texto))
    texto = texto.replace(" ", " ").replace("​", "")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/sentiments")
    ap.add_argument("--out", default="data/processed/sentiments")
    args = ap.parse_args()

    raiz, saida = Path(args.raw), Path(args.out)
    saida.mkdir(parents=True, exist_ok=True)

    ficheiros = sorted(raiz.glob("goemotions_*_pt.csv"))
    if not ficheiros:
        raise SystemExit(f"Nenhum goemotions_*_pt.csv encontrado em {raiz}")

    df = pd.concat([ler_csv(f) for f in ficheiros], ignore_index=True)
    print(f"Linhas raw (anotações): {len(df)}  |  ficheiros: {[f.name for f in ficheiros]}")

    # 1. remover anotações "very unclear"
    df = df[df["example_very_unclear"].str.lower() != "true"]
    df[EMOCOES_GO] = df[EMOCOES_GO].astype(int)

    # 2. agregar por comentário: nº de anotadores que marcaram cada emoção
    agg = df.groupby("id").agg(
        text_en=("text", "first"),
        text=("texto", "first"),
        n_anotadores=("rater_id", "nunique"),
        **{e: (e, "sum") for e in EMOCOES_GO},
    )

    # 3. cada emoção (sem agrupar) está presente se >= MIN_ANOTADORES a marcarem
    for en, pt in NOMES_PT.items():
        agg[pt] = (agg[en] >= MIN_ANOTADORES).astype(int)
    agg["n_emocoes"] = agg[CLASSES].sum(axis=1)

    agg["text"] = agg["text"].map(limpar_texto)
    agg = agg[agg["text"].str.len() > 0]

    print(f"Comentários únicos: {len(agg)}")
    print(f"Sem emoção consensual (descartados): {(agg['n_emocoes'] == 0).sum()}")
    agg = agg[agg["n_emocoes"] >= 1]           # sem emoção consensual -> descartado
    agg = agg[~agg["text"].duplicated()].reset_index()

    # 4. divisão 80/10/10, feita UMA vez e partilhada pelas duas versões.
    #    Estratificação multi-etiqueta aproximada: cada texto é estratificado pela
    #    sua emoção MAIS RARA, para as classes raras aparecerem em todas as divisões.
    freq = agg[CLASSES].sum()
    agg["estrato"] = agg[CLASSES].apply(
        lambda r: min((c for c in CLASSES if r[c]), key=lambda c: freq[c]), axis=1)
    treino, resto = train_test_split(agg.index, test_size=0.2,
                                     stratify=agg["estrato"], random_state=SEED)
    val, teste = train_test_split(resto, test_size=0.5,
                                  stratify=agg.loc[resto, "estrato"], random_state=SEED)
    agg["split"] = "train"
    agg.loc[val, "split"] = "val"
    agg.loc[teste, "split"] = "test"

    # 5a. MULTI-ETIQUETA (principal): todas as emoções de cada texto
    agg["labels"] = agg[CLASSES].apply(lambda r: ";".join(c for c in CLASSES if r[c]), axis=1)
    multi = agg[["id", "text", "text_en", "labels", "n_emocoes"] + CLASSES + ["split"]]
    guardar(multi, saida, "sentiments")

    # 5b. UMA ETIQUETA (opcional, para comparar): só textos com exatamente uma emoção
    simples = agg[agg["n_emocoes"] == 1].copy()
    simples["label_name"] = simples["labels"]
    simples["label"] = simples["label_name"].map({c: i for i, c in enumerate(CLASSES)})
    simples = simples[["id", "text", "text_en", "label", "label_name", "split"]]
    guardar(simples, saida, "sentiments_simples")

    print(f"Comentários com pelo menos uma emoção consensual: {len(multi)}")
    print(f"  com uma emoção:    {(multi['n_emocoes'] == 1).sum()}")
    print(f"  com várias emoções: {(multi['n_emocoes'] > 1).sum()}\n")
    contagem = multi.groupby("split")[CLASSES].sum().T[["train", "val", "test"]]
    contagem["total"] = contagem.sum(axis=1)
    print("Nº de textos com cada emoção (multi-etiqueta):")
    print(contagem.sort_values("total", ascending=False))
    print("\nOrdem das colunas de emoção:", CLASSES)


def guardar(df: pd.DataFrame, saida: Path, prefixo: str):
    df.to_csv(saida / f"{prefixo}.csv", index=False)
    for s in ("train", "val", "test"):
        df[df["split"] == s].drop(columns="split").to_csv(
            saida / f"{prefixo}_{s}.csv", index=False)

if __name__ == "__main__":
    main()