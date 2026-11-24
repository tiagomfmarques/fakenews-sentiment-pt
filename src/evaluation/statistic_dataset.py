from pathlib import Path
import pandas as pd

PASTA = Path("data/processed")
COLUNAS_INFO = {"id", "text", "text_en", "labels", "n_emocoes", "split"}


def ler(nome: str) -> pd.DataFrame:
    """Procura o ficheiro em data/processed/ e nas subpastas."""
    for c in [PASTA / nome, *PASTA.glob(f"*/{nome}")]:
        if c.exists():
            return pd.read_csv(c)
    raise FileNotFoundError(f"Não encontrei {nome} em {PASTA}")


# 1. Fake.Br
fake = ler("fakebr_full.csv")
print("FAKE.BR")
print(f"  Falsas:      {(fake['label'] == 1).sum()}")
print(f"  Verdadeiras: {(fake['label'] == 0).sum()}")
print(f"  Total:       {len(fake)}")

# 2. GoEmotions PT
emo = ler("sentiments.csv")
emocoes = [c for c in emo.columns if c not in COLUNAS_INFO]
contagem = emo[emocoes].sum().sort_values(ascending=False)

print("\nGOEMOTIONS PT")
for e, n in contagem.items():
    print(f"  {e:<14} {n:>6}")
print(f"\n  Total de textos: {len(emo)}")
