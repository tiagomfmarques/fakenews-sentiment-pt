import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.evaluation.metrics import (RESULTS_DIR, calcular_metricas, carregar_split,guardar_resultados)

SEED = 42
GRELHA_C = [0.1, 0.5, 1, 2, 5, 10, 20]


def criar_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents=None,       # mantém acentos (são informativos em PT)
            ngram_range=(1, 2),       # unigramas + bigramas
            min_df=2,
            max_df=0.95,
            max_features=100_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
    ])


def palavras_mais_indicativas(pipe: Pipeline, n: int = 25) -> dict:
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coefs = pipe.named_steps["clf"].coef_[0]          # >0 puxa para "fake"
    ordem = np.argsort(coefs)
    return {
        "fake": [(vocab[i], round(float(coefs[i]), 3)) for i in ordem[::-1][:n]],
        "true": [(vocab[i], round(float(coefs[i]), 3)) for i in ordem[:n]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versao", choices=["normalized", "full"], default="full")
    ap.add_argument("--data-dir", default="data/processed")
    args = ap.parse_args()

    nome = f"baseline_tfidf_lr_{args.versao}"
    treino = carregar_split(args.versao, "train", args.data_dir)
    val = carregar_split(args.versao, "val", args.data_dir)
    teste = carregar_split(args.versao, "test", args.data_dir)
    print(f"treino={len(treino)}  val={len(val)}  teste={len(teste)}")

    # 1. escolher C na validação
    melhor_C, melhor_f1 = None, -1
    for C in GRELHA_C:
        pipe = criar_pipeline(C).fit(treino["text"], treino["label"])
        f1 = calcular_metricas(val["label"], pipe.predict(val["text"]))["f1_macro"]
        print(f"  C={C:<5} f1_macro(val)={f1:.4f}")
        if f1 > melhor_f1:
            melhor_C, melhor_f1 = C, f1
    print(f"Melhor C = {melhor_C}")

    # 2. treino final com o melhor C (só no treino; o teste nunca é usado para escolher nada)
    pipe = criar_pipeline(melhor_C).fit(treino["text"], treino["label"])

    prob_val = pipe.predict_proba(val["text"])[:, 1]
    metricas_val = calcular_metricas(val["label"], (prob_val >= 0.5).astype(int), prob_val)

    prob_teste = pipe.predict_proba(teste["text"])[:, 1]
    pred_teste = (prob_teste >= 0.5).astype(int)
    metricas_teste = calcular_metricas(teste["label"], pred_teste, prob_teste)

    # 3. gravar modelo
    pasta_modelo = Path("models/fakenews") / nome
    pasta_modelo.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, pasta_modelo / "model.joblib")

    config = {"tipo": "tfidf+logreg", "versao": args.versao, "C": melhor_C,
              "ngram_range": [1, 2], "max_features": 100_000,
              "n_treino": len(treino), "pasta_modelo": str(pasta_modelo)}
    guardar_resultados(nome, config, metricas_val, metricas_teste, teste, pred_teste, prob_teste)

    # 4. palavras mais indicativas (útil para o relatório e para detetar atalhos)
    palavras = palavras_mais_indicativas(pipe)
    caminho = RESULTS_DIR / "metrics" / f"fakenews_{nome}_palavras.json"
    caminho.write_text(json.dumps(palavras, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Top palavras -> fake:", [w for w, _ in palavras["fake"][:10]])
    print("Top palavras -> true:", [w for w, _ in palavras["true"][:10]])


if __name__ == "__main__":
    main()