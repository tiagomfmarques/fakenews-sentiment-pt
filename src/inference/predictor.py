import os

# garantir que NADA vai à Internet (regra do enunciado): só ficheiros locais
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import re
from pathlib import Path

import numpy as np

CLASSES_FAKE = {0: "verdadeira", 1: "falsa"}


def _dispositivo():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dividir_frases(texto: str, min_palavras: int = 3) -> list:
    """Divide um texto em frases (regex simples, sem dependências extra)."""
    partes = []
    for linha in texto.splitlines():              # cada linha (ex.: o título) é tratada à parte
        linha = re.sub(r"\s+", " ", linha).strip()
        if linha:
            partes += re.split(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÀÇ\"“«(0-9])", linha)
    return [p.strip() for p in partes if len(p.split()) >= min_palavras]


class _Base:
    """Lógica comum: decide se a pasta é uma baseline (joblib) ou um Transformer."""

    def __init__(self, pasta: str, max_len: int):
        self.pasta = Path(pasta)
        if not self.pasta.exists():
            raise FileNotFoundError(f"Pasta do modelo não existe: {self.pasta}")
        self.max_len = max_len
        self.baseline = (self.pasta / "model.joblib").exists()
        if self.baseline:
            import joblib
            self.pipe = joblib.load(self.pasta / "model.joblib")
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.device = _dispositivo()
            self.tok = AutoTokenizer.from_pretrained(self.pasta, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.pasta, local_files_only=True).float().to(self.device).eval()

    def _logits(self, textos: list, batch: int = 16) -> np.ndarray:
        saidas = []
        with self.torch.no_grad():
            for i in range(0, len(textos), batch):
                enc = self.tok(textos[i:i + batch], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.device)
                saidas.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(saidas)

    @property
    def nome(self) -> str:
        return self.pasta.name


class FakeNewsPredictor(_Base):
    """Notícia inteira -> verdadeira / falsa + probabilidade."""

    def __init__(self, pasta: str, max_len: int = 512):
        super().__init__(pasta, max_len)

    def prob_fake(self, textos: list) -> np.ndarray:
        if self.baseline:
            return self.pipe.predict_proba(textos)[:, 1]
        logits = self._logits(textos, batch=4)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, 1]

    def prever(self, texto: str, limiar: float = 0.5) -> dict:
        p = float(self.prob_fake([texto])[0])
        classe = int(p >= limiar)
        return {"classe": CLASSES_FAKE[classe], "prob_fake": round(p, 4),
                "confianca": round(p if classe else 1 - p, 4), "modelo": self.nome}

    def palavras_decisivas(self, texto: str, n: int = 10) -> list:
        """Só para a baseline: termos do texto que mais puxaram para 'falsa' (+) ou 'verdadeira' (-)."""
        if not self.baseline:
            return []
        tfidf, clf = self.pipe.named_steps["tfidf"], self.pipe.named_steps["clf"]
        x = tfidf.transform([texto])
        contrib = x.multiply(clf.coef_[0]).tocsr()
        vocab = tfidf.get_feature_names_out()
        pares = sorted(zip(contrib.indices, contrib.data), key=lambda t: -abs(t[1]))[:n]
        return [(str(vocab[i]), round(float(v), 4)) for i, v in pares]


class EmotionPredictor(_Base):
    """Frases -> emoções (28 classes, multi-etiqueta) usando o limiar guardado no treino."""

    def __init__(self, pasta: str, max_len: int = 128):
        super().__init__(pasta, max_len)
        info = json.loads((self.pasta / "limiar.json").read_text(encoding="utf-8"))
        self.limiar = info["limiar"]
        self.classes = info["classes"]

    def probs(self, textos: list) -> np.ndarray:
        if self.baseline:
            return self.pipe.predict_proba(textos)
        return 1 / (1 + np.exp(-self._logits(textos)))

    def prever(self, textos: list, top: int = 3) -> list:
        """Para cada texto: as emoções acima do limiar (ou a mais provável) + o top-N."""
        P = self.probs(textos)
        res = []
        for texto, p in zip(textos, P):
            idx = np.where(p >= self.limiar)[0]
            if len(idx) == 0:
                idx = [int(p.argmax())]
            res.append({
                "texto": texto,
                "emocoes": [self.classes[i] for i in sorted(idx, key=lambda i: -p[i])],
                "top": [(self.classes[i], round(float(p[i]), 3)) for i in np.argsort(p)[::-1][:top]],
            })
        return res

    def analisar_noticia(self, texto: str, limiar_dominante: float = 0.15) -> dict:
        """Classifica frase a frase e agrega num perfil emocional da notícia."""
        frases = dividir_frases(texto) or [texto]
        P = self.probs(frases)
        perfil = P.mean(axis=0)
        ordem = np.argsort(perfil)[::-1]
        sem_neutro = [i for i in ordem if self.classes[i] != "neutro"]
        dom = sem_neutro[0] if sem_neutro and perfil[sem_neutro[0]] >= limiar_dominante else None
        return {
            "n_frases": len(frases),
            "emocao_dominante": self.classes[dom] if dom is not None else "neutro",
            "perfil": {self.classes[i]: round(float(perfil[i]), 3) for i in ordem[:8]},
            "por_frase": self.prever(frases, top=2),
            "modelo": self.nome,
        }