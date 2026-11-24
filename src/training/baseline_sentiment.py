import argparse
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from src.training.train_sentiment import (RESULTS_DIR, afinar_limiar, binarizar,carregar_split, classes_de, grafico_f1, metricas)

SEED = 42
GRELHA_C = [0.25, 1, 4]
NOME = "baseline"

def criar_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),       # unigramas + bigramas
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
        )),
        # uma regressão logística por emoção; class_weight compensa as emoções raras
        ("clf", OneVsRestClassifier(
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED),
            n_jobs=-1)),
    ])


def palavras_por_emocao(pipe: Pipeline, classes: list, n: int = 10) -> dict:
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    estimadores = pipe.named_steps["clf"].estimators_
    return {c: [str(w) for w in vocab[np.argsort(est.coef_[0])[::-1][:n]]]
            for c, est in zip(classes, estimadores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--models-dir", default="models/sentiment")
    args = ap.parse_args()

    train, val, test = (carregar_split(s, args.data_dir) for s in ("train", "val", "test"))
    classes = classes_de(train)
    Y_train, Y_val, Y_test = (d[classes].values for d in (train, val, test))
    print(f"treino={len(train)}  val={len(val)}  teste={len(test)}  emoções={len(classes)}")

    # 1. escolher C na validação (cada C com o seu melhor limiar)
    melhor = None
    for C in GRELHA_C:
        pipe = criar_pipeline(C).fit(train["text"], Y_train)
        p_val = pipe.predict_proba(val["text"])
        limiar = afinar_limiar(Y_val, p_val)
        f1 = f1_score(Y_val, binarizar(p_val, limiar), average="macro", zero_division=0)
        print(f"  C={C:<5} f1_macro(val)={f1:.4f}")
        if melhor is None or f1 > melhor[0]:
            melhor = (f1, C, limiar, pipe, p_val)
    _, C, limiar, pipe, p_val = melhor
    print(f"Melhor C = {C}  |  limiar = {limiar}")

    # 2. avaliar no teste (o teste nunca foi usado para escolher nada)
    p_test = pipe.predict_proba(test["text"])
    P_val, P_test = binarizar(p_val, limiar), binarizar(p_test, limiar)
    m_val, m_test = metricas(Y_val, P_val), metricas(Y_test, P_test)
    f1s = f1_score(Y_test, P_test, average=None, zero_division=0)
    por_classe = {c: {"f1": round(float(f), 4), "suporte": int(Y_test[:, i].sum())}
                  for i, (c, f) in enumerate(zip(classes, f1s))}

    # 3. gravar modelo
    pasta = Path(args.models_dir) / NOME
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, pasta / "model.joblib")
    (pasta / "limiar.json").write_text(json.dumps({"limiar": limiar, "classes": classes},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. gravar métricas, previsões, figura e palavras
    pasta_m, pasta_f = RESULTS_DIR / "metrics", RESULTS_DIR / "figures"
    pasta_m.mkdir(parents=True, exist_ok=True)
    pasta_f.mkdir(parents=True, exist_ok=True)
    config = {"modelo": NOME, "tipo": "tfidf+logreg (one-vs-rest)", "C": C, "limiar": limiar,
              "ngram_range": [1, 2], "max_features": 100_000, "class_weight": "balanced",
              "n_treino": len(train), "n_classes": len(classes), "pasta_modelo": str(pasta)}
    saida = {"modelo": NOME, "data": datetime.now().isoformat(timespec="seconds"),
             "config": config, "val": m_val, "test": m_test, "test_por_emocao": por_classe}
    (pasta_m / f"sentiment_{NOME}.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    def nomes(M):
        return [";".join(c for c, v in zip(classes, linha) if v) for linha in M]
    pd.DataFrame({"id": test["id"], "text": test["text"], "real": nomes(Y_test),
                  "previsto": nomes(P_test)}).to_csv(
        pasta_m / f"sentiment_{NOME}_preds.csv", index=False)
    grafico_f1(por_classe, NOME, pasta_f / f"sentiment_{NOME}_f1.png")
    (pasta_m / f"sentiment_{NOME}_palavras.json").write_text(
        json.dumps(palavras_por_emocao(pipe, classes), indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(f"\n[{NOME}] TESTE  f1_micro={m_test['f1_micro']:.4f}  f1_macro={m_test['f1_macro']:.4f}  "
          f"exact_match={m_test['exact_match']:.4f}")
    ordem = sorted(por_classe.items(), key=lambda kv: -kv[1]["f1"])
    print("Melhores:", [(c, v["f1"]) for c, v in ordem[:5]])
    print("Piores:  ", [(c, v["f1"]) for c, v in ordem[-5:]])
    print(f"Modelo em {pasta}  |  métricas em {pasta_m / f'sentiment_{NOME}.json'}")


if __name__ == "__main__":
    main()