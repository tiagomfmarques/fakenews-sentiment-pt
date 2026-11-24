import argparse
import inspect
import shutil
from pathlib import Path
import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,DataCollatorWithPadding, EarlyStoppingCallback, Trainer,TrainingArguments, set_seed)
from src.evaluation.metrics import calcular_metricas, carregar_split, guardar_resultados

MODELOS = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "albertina": "PORTULAN/albertina-100m-portuguese-ptbr-encoder",
    "xlmr": "FacebookAI/xlm-roberta-base",
}
ID2LABEL = {0: "true", 1: "fake"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, textos, labels, tokenizer, max_len):
        self.enc = tokenizer(list(textos), truncation=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def metricas_trainer(eval_pred):
    logits, labels = eval_pred
    m = calcular_metricas(labels, np.argmax(logits, axis=-1))
    m.pop("confusion_matrix")          # o Trainer só regista números
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="bertimbau",
                    help=f"um de {list(MODELOS)} ou um id/pasta do Hugging Face")
    ap.add_argument("--versao", default="full", choices=["normalized", "full"])
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="acumula gradientes de N batches (batch efetivo = batch x N); útil com pouca memória")
    ap.add_argument("--no-fp16", action="store_true",
                    help="desliga fp16 na GPU NVIDIA (o Albertina/DeBERTa desliga-o sempre)")
    ap.add_argument("--models-dir", default="models/fakenews")
    args = ap.parse_args()

    set_seed(args.seed)
    model_id = MODELOS.get(args.modelo, args.modelo)
    nome = f"{Path(args.modelo).name}_{args.versao}"
    pasta_modelo = Path(args.models_dir) / nome
    pasta_ckpt = Path(args.models_dir) / "checkpoints" / nome
    gpu = torch.cuda.is_available()
    if gpu:
        dispositivo = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        dispositivo = "Apple MPS (GPU do Mac)"      # o Trainer usa-a automaticamente
    else:
        dispositivo = "CPU (vai ser lento)"
    print(f"Modelo: {model_id}  |  Dispositivo: {dispositivo}")

    train, val, test = (carregar_split(args.versao, s, args.data_dir)
                        for s in ("train", "val", "test"))
    for df in (train, val, test):
        df["text"] = df["text"].fillna("").astype(str)
    print(f"treino={len(train)}  val={len(val)}  teste={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ds_train = TextDataset(train["text"], train["label"], tokenizer, args.max_len)
    ds_val = TextDataset(val["text"], val["label"], tokenizer, args.max_len)
    ds_test = TextDataset(test["text"], test["label"], tokenizer, args.max_len)

    # quantos textos são cortados pelo limite de tokens (vai para o relatório)
    n_tok = [len(tokenizer(t)["input_ids"]) for t in train["text"]]
    cortados = float(np.mean(np.array(n_tok) > args.max_len))
    print(f"Tokens por texto (treino): mediana={int(np.median(n_tok))}, "
          f"p95={int(np.percentile(n_tok, 95))}, cortados a {args.max_len}: {cortados:.1%}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
    # Alguns checkpoints (ex.: Albertina) vêm guardados em bfloat16 e as versões
    # recentes do transformers carregam-nos nesse tipo -> erro "Float and BFloat16".
    # Treinamos sempre a partir de pesos em float32 (o fp16, quando ligado, é aplicado por cima).
    model = model.float()

    # fp16 só com CUDA. O DeBERTa (Albertina) dá overflow em fp16
    # ("value cannot be converted to type c10::Half"), por isso treina sempre em fp32.
    usar_fp16 = gpu and not args.no_fp16 and model.config.model_type not in ("deberta", "deberta-v2")
    if gpu and not usar_fp16:
        print("fp16 desligado: treino em fp32")

    targs = dict(
        output_dir=str(pasta_ckpt),
        num_train_epochs=args.epocas,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        weight_decay=0.01,
        warmup_steps=int(0.1 * args.epocas * np.ceil(len(ds_train) / (args.batch * args.grad_accum))),  # 10% aquecimento
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=usar_fp16,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    # compatibilidade com versões antigas do transformers
    if "eval_strategy" not in inspect.signature(TrainingArguments).parameters:
        targs["evaluation_strategy"] = targs.pop("eval_strategy")

    trainer_kwargs = dict(
        model=model,
        args=TrainingArguments(**targs),
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metricas_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    if "processing_class" in inspect.signature(Trainer).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # guardar modelo final (é isto que a app carrega offline)
    trainer.save_model(str(pasta_modelo))
    tokenizer.save_pretrained(str(pasta_modelo))
    print(f"Modelo guardado em {pasta_modelo}")

    # avaliação com o melhor modelo (escolhido na validação) em val e TESTE
    def prever(ds):
        logits = trainer.predict(ds).predictions
        prob_fake = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        return (prob_fake >= 0.5).astype(int), prob_fake

    pred_val, prob_val = prever(ds_val)
    pred_test, prob_test = prever(ds_test)

    config = {
        "tipo": "transformer", "model_id": model_id, "versao": args.versao,
        **{k: getattr(args, k) for k in ("epocas", "lr", "batch", "grad_accum", "max_len", "seed")},
        "fp16": usar_fp16,
        "textos_cortados_treino": round(cortados, 4),
        "melhor_checkpoint": trainer.state.best_model_checkpoint,
        "pasta_modelo": str(pasta_modelo),
        "historico": [h for h in trainer.state.log_history if "eval_f1_macro" in h],
    }
    guardar_resultados(nome, config,
                       calcular_metricas(val["label"], pred_val, prob_val),
                       calcular_metricas(test["label"], pred_test, prob_test),
                       test, pred_test, prob_test)

    # apagar checkpoints intermédios (ocupam muito espaço)
    shutil.rmtree(pasta_ckpt, ignore_errors=True)
    if pasta_ckpt.parent.exists() and not any(pasta_ckpt.parent.iterdir()):
        pasta_ckpt.parent.rmdir()

if __name__ == "__main__":
    main()