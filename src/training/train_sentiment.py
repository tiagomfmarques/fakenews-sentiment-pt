import argparse
import inspect
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

RESULTS_DIR = Path("src/results")
MODELOS = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "albertina": "PORTULAN/albertina-100m-portuguese-ptbr-encoder",
    "xlmr": "FacebookAI/xlm-roberta-base",
}
COLUNAS_INFO = {"id", "text", "text_en", "labels", "n_emocoes", "split"}

def carregar_split(split: str, data_dir: str) -> pd.DataFrame:
    nome = f"sentiments_{split}.csv"
    for c in [Path(data_dir) / nome, *Path(data_dir).glob(f"*/{nome}")]:
        if c.exists():
            df = pd.read_csv(c)
            df["text"] = df["text"].fillna("").astype(str)
            return df
    raise SystemExit(f"Não encontrei {nome} em {data_dir}. "
                     "Corre primeiro: python -m src.data.preprocess_emocoes")

def classes_de(df: pd.DataFrame) -> list:
    """As 28 colunas de emoção (0/1), pela ordem do ficheiro."""
    return [c for c in df.columns if c not in COLUNAS_INFO]

def metricas(Y, P) -> dict:
    """Y, P = matrizes (n_textos, n_classes) de 0/1."""
    return {
        "f1_micro": round(float(f1_score(Y, P, average="micro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(Y, P, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(Y, P, average="weighted", zero_division=0)), 4),
        "precision_micro": round(float(precision_score(Y, P, average="micro", zero_division=0)), 4),
        "recall_micro": round(float(recall_score(Y, P, average="micro", zero_division=0)), 4),
        "exact_match": round(float((Y == P).all(axis=1).mean()), 4),   # todas as emoções certas
    }

def binarizar(probs, limiar):
    """Aplica o limiar; um texto sem nenhuma emoção acima fica com a mais provável."""
    P = (probs >= limiar).astype(int)
    vazios = P.sum(axis=1) == 0
    P[vazios, probs[vazios].argmax(axis=1)] = 1
    return P

def afinar_limiar(Y_val, probs_val) -> float:
    grelha = np.round(np.arange(0.05, 0.95, 0.05), 2)
    f1s = [f1_score(Y_val, binarizar(probs_val, t), average="macro", zero_division=0) for t in grelha]
    melhor = float(grelha[int(np.argmax(f1s))])
    print(f"Limiar escolhido na validação: {melhor}  (F1 macro val = {max(f1s):.4f})")
    return melhor

def treinar_baseline(train, val, test, classes, args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    import joblib

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                                  sublinear_tf=True)),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced"),
                                    n_jobs=-1)),
    ])
    pipe.fit(train["text"], train[classes].values)
    pasta = Path(args.models_dir) / "baseline"
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, pasta / "model.joblib")
    return pipe.predict_proba(val["text"]), pipe.predict_proba(test["text"]), pasta, {}


def treinar_transformer(train, val, test, classes, args):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                              TrainingArguments, set_seed)

    set_seed(args.seed)
    model_id = MODELOS.get(args.modelo, args.modelo)
    gpu = torch.cuda.is_available()
    disp = (torch.cuda.get_device_name(0) if gpu else
            "Apple MPS (GPU do Mac)" if torch.backends.mps.is_available() else "CPU (vai ser lento)")
    print(f"Modelo: {model_id}  |  Dispositivo: {disp}")

    tok = AutoTokenizer.from_pretrained(model_id)

    class DS(torch.utils.data.Dataset):
        def __init__(self, df):
            self.enc = tok(list(df["text"]), truncation=True, max_length=args.max_len)
            self.y = df[classes].values.astype("float32")   # BCE precisa de float

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.y[i]
            return item

    ds_train, ds_val, ds_test = DS(train), DS(val), DS(test)

    id2label = dict(enumerate(classes))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(classes), problem_type="multi_label_classification",
        id2label=id2label, label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True)
    model = model.float()    # checkpoints em bfloat16 (ex.: Albertina) -> float32

    usar_fp16 = gpu and not args.no_fp16 and model.config.model_type not in ("deberta", "deberta-v2")

    def metricas_trainer(ev):
        logits, labels = ev
        probs = 1 / (1 + np.exp(-logits))
        return metricas(labels.astype(int), binarizar(probs, 0.5))

    pasta_ckpt = Path(args.models_dir) / "checkpoints" / Path(args.modelo).name
    passos = int(np.ceil(len(ds_train) / (args.batch * args.grad_accum))) * args.epocas
    targs = dict(
        output_dir=str(pasta_ckpt), num_train_epochs=args.epocas, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch * 2,
        gradient_accumulation_steps=args.grad_accum, weight_decay=0.01,
        warmup_steps=int(0.1 * passos), eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, fp16=usar_fp16, logging_steps=100, report_to="none", seed=args.seed,
    )
    if "eval_strategy" not in inspect.signature(TrainingArguments).parameters:
        targs["evaluation_strategy"] = targs.pop("eval_strategy")
    kw = dict(model=model, args=TrainingArguments(**targs), train_dataset=ds_train,
              eval_dataset=ds_val, data_collator=DataCollatorWithPadding(tok),
              compute_metrics=metricas_trainer,
              callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    kw["processing_class" if "processing_class" in inspect.signature(Trainer).parameters
       else "tokenizer"] = tok
    trainer = Trainer(**kw)
    trainer.train()

    pasta = Path(args.models_dir) / Path(args.modelo).name
    trainer.save_model(str(pasta))
    tok.save_pretrained(str(pasta))

    def probs(ds):
        return 1 / (1 + np.exp(-trainer.predict(ds).predictions))

    p_val, p_test = probs(ds_val), probs(ds_test)
    shutil.rmtree(pasta_ckpt, ignore_errors=True)
    if pasta_ckpt.parent.exists() and not any(pasta_ckpt.parent.iterdir()):
        pasta_ckpt.parent.rmdir()
    extra = {"model_id": model_id, "fp16": usar_fp16,
             "historico": [h for h in trainer.state.log_history if "eval_f1_macro" in h]}
    return p_val, p_test, pasta, extra


def grafico_f1(por_classe: dict, nome: str, caminho: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    itens = sorted(por_classe.items(), key=lambda kv: kv[1]["f1"])
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh([k for k, _ in itens], [v["f1"] for _, v in itens], color="#4C72B0")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (teste)")
    ax.set_title(f"F1 por emoção — {nome}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

def criar_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="bertimbau",
                    help=f"baseline, um de {list(MODELOS)} ou um id/pasta do Hugging Face")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=128, help="os comentários são curtos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--models-dir", default="models/sentiment")
    return ap

def main():
    correr(criar_parser().parse_args())

def correr(args):
    """Treina e avalia um modelo de emoções (também chamado por train_baseline.py)."""
    train, val, test = (carregar_split(s, args.data_dir) for s in ("train", "val", "test"))
    classes = classes_de(train)
    print(f"treino={len(train)}  val={len(val)}  teste={len(test)}  emoções={len(classes)}")

    if args.modelo == "baseline":
        p_val, p_test, pasta, extra = treinar_baseline(train, val, test, classes, args)
    else:
        p_val, p_test, pasta, extra = treinar_transformer(train, val, test, classes, args)
    nome = Path(args.modelo).name

    Y_val, Y_test = val[classes].values, test[classes].values
    limiar = afinar_limiar(Y_val, p_val)
    P_val, P_test = binarizar(p_val, limiar), binarizar(p_test, limiar)
    (pasta / "limiar.json").write_text(json.dumps({"limiar": limiar, "classes": classes},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")

    m_val, m_test = metricas(Y_val, P_val), metricas(Y_test, P_test)
    f1s = f1_score(Y_test, P_test, average=None, zero_division=0)
    por_classe = {c: {"f1": round(float(f), 4), "suporte": int(Y_test[:, i].sum())}
                  for i, (c, f) in enumerate(zip(classes, f1s))}

    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ("modelo", "epocas", "lr", "batch", "grad_accum",
                                            "max_len", "seed")}
    config.update(limiar=limiar, pasta_modelo=str(pasta), n_classes=len(classes), **extra)
    if args.modelo == "baseline":
        config = {"modelo": "baseline", "tipo": "tfidf+logreg (one-vs-rest)", "limiar": limiar,
                  "pasta_modelo": str(pasta)}
    saida = {"modelo": nome, "data": datetime.now().isoformat(timespec="seconds"),
             "config": config, "val": m_val, "test": m_test, "test_por_emocao": por_classe}
    (RESULTS_DIR / "metrics" / f"sentiment_{nome}.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    def nomes(M):
        return [";".join(c for c, v in zip(classes, linha) if v) for linha in M]
    pd.DataFrame({"id": test["id"], "text": test["text"], "real": nomes(Y_test),
                  "previsto": nomes(P_test)}).to_csv(
        RESULTS_DIR / "metrics" / f"sentiment_{nome}_preds.csv", index=False)
    grafico_f1(por_classe, nome, RESULTS_DIR / "figures" / f"sentiment_{nome}_f1.png")

    print(f"\n[{nome}] TESTE  f1_micro={m_test['f1_micro']:.4f}  f1_macro={m_test['f1_macro']:.4f}  "
          f"exact_match={m_test['exact_match']:.4f}")
    melhores = sorted(por_classe.items(), key=lambda kv: -kv[1]["f1"])
    print("Melhores:", [(c, v["f1"]) for c, v in melhores[:5]])
    print("Piores:  ", [(c, v["f1"]) for c, v in melhores[-5:]])
    print(f"Modelo em {pasta}  |  métricas em src/results/metrics/sentiment_{nome}.json")


if __name__ == "__main__":
    main()