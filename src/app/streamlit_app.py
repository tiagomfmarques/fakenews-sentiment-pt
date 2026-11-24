"""
Análise de Notícias — Hugging Face Space (Streamlit).
Deteção de notícias falsas (BERTimbau) + análise de sentimentos (XLM-RoBERTa) em português.
Não depende de outros scripts do projeto: só precisa do exemplos.json na mesma pasta.
Os modelos são descarregados do Hugging Face Hub no arranque e ficam em cache.
Modelos:
    https://huggingface.co/tiagomfmarques/bertimbau-fakenews-pt
    https://huggingface.co/tiagomfmarques/xlmr-emocoes-pt
"""

import html
import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Configuração
REPO_FAKE = "tiagomfmarques/bertimbau-fakenews-pt"
REPO_EMO = "tiagomfmarques/xlmr-emocoes-pt"

LINK_HF_FAKE = f"https://huggingface.co/{REPO_FAKE}"
LINK_HF_EMO = f"https://huggingface.co/{REPO_EMO}"
LINK_GITHUB = "https://github.com/tiagomfmarques/fakenews-sentiment-pt"

MAX_LEN_FAKE = 512      # igual ao treino
MAX_LEN_EMO = 128       # igual ao treino
LIMIAR_EMO_DEFEITO = 0.3  # usado só se o repositório não tiver limiar.json
LIMIAR_DOMINANTE = 0.15   # presença mínima para uma emoção ser "dominante"


# Exemplos (lidos de exemplos.json, na mesma pasta que este ficheiro)
FICHEIRO_EXEMPLOS = Path(__file__).resolve().parent / "exemplos.json"
# nome mostrado na barra lateral -> chave no JSON
OPCOES_EXEMPLO = {"Notícia Verdadeira": "verdadeira", "Notícia Falsa": "falsa"}


@st.cache_data
def carregar_exemplos() -> dict:
    try:
        return json.loads(FICHEIRO_EXEMPLOS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


# Cores (texto escuro sobre fundos claros; branco sobre fundos fortes)
TEXTO = "#1f2933"
TEXTO_SUAVE = "#52606d"
CINZA_FUNDO = "#e4e7eb"
FAMILIA = {
    "positiva": ["diversão", "entusiasmo", "alegria", "amor", "desejo", "otimismo", "cuidado",
                 "orgulho", "admiração", "gratidão", "alívio", "aprovação"],
    "negativa": ["medo", "nervosismo", "remorso", "vergonha", "desilusão", "tristeza", "luto",
                 "nojo", "raiva", "irritação", "desaprovação"],
    "ambígua": ["constatação", "surpresa", "curiosidade", "confusão"],
    "neutra": ["neutro"],
}
FAMILIA_DE = {e: f for f, es in FAMILIA.items() for e in es}

NOME_PT = {
    "admiração": "admiração", "diversão": "divertimento", "raiva": "raiva",
    "irritação": "irritação", "aprovação": "aprovação", "cuidado": "afeto",
    "confusão": "confusão", "curiosidade": "curiosidade", "desejo": "desejo",
    "desilusão": "desilusão", "desaprovação": "desaprovação", "nojo": "repulsa",
    "vergonha": "embaraço", "entusiasmo": "entusiasmo", "medo": "medo",
    "gratidão": "gratidão", "luto": "luto", "alegria": "alegria", "amor": "amor",
    "nervosismo": "nervosismo", "otimismo": "otimismo", "orgulho": "orgulho",
    "constatação": "perceção", "alívio": "alívio", "remorso": "remorso",
    "tristeza": "tristeza", "surpresa": "surpresa", "neutro": "neutro",
}


def nome(emocao: str) -> str:
    return NOME_PT.get(emocao, emocao)


CORES = {  # (cor forte, fundo claro)
    "positiva": ("#1b7a3a", "#e3f4e8"),
    "negativa": ("#b42318", "#fdeceb"),
    "ambígua": ("#9a5b00", "#fff2dc"),
    "neutra": ("#4b5563", "#eef0f3"),
}
VERDADEIRA, FALSA = CORES["positiva"], CORES["negativa"]
ESPACO = "margin:10px 0 22px;"  # espaço entre caixas e secções


def cor(emocao: str):
    return CORES[FAMILIA_DE.get(emocao, "neutra")]


# Modelos (descarregados do Hub uma vez e guardados em cache)
def dividir_frases(texto: str, min_palavras: int = 3) -> list:
    """Divide o texto em frases; cada linha (ex.: o título) é tratada à parte."""
    partes = []
    for linha in texto.splitlines():
        linha = re.sub(r"\s+", " ", linha).strip()
        if linha:
            partes += re.split(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÀÇ\"“«(0-9])", linha)
    return [p.strip() for p in partes if len(p.split()) >= min_palavras]


@st.cache_resource(show_spinner="A carregar os modelos...")
def carregar_modelos():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok_f = AutoTokenizer.from_pretrained(REPO_FAKE)
    mod_f = AutoModelForSequenceClassification.from_pretrained(REPO_FAKE).float().eval()
    tok_e = AutoTokenizer.from_pretrained(REPO_EMO)
    mod_e = AutoModelForSequenceClassification.from_pretrained(REPO_EMO).float().eval()
    try:
        info = json.loads(open(hf_hub_download(REPO_EMO, "limiar.json"), encoding="utf-8").read())
        limiar, classes = info["limiar"], info["classes"]
    except Exception:
        limiar = LIMIAR_EMO_DEFEITO
        classes = [mod_e.config.id2label[i] for i in range(mod_e.config.num_labels)]
    return tok_f, mod_f, tok_e, mod_e, limiar, classes


def prever_fake(tok, modelo, texto: str) -> dict:
    enc = tok(texto, truncation=True, max_length=MAX_LEN_FAKE, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(modelo(**enc).logits.float(), dim=-1)[0].numpy()
    # índice da classe "falsa" a partir do config (no treino: 0 = true, 1 = fake)
    id_fake = next((int(i) for i, l in modelo.config.id2label.items()
                    if str(l).lower() in ("fake", "falsa")), 1)
    p = float(probs[id_fake])
    falsa = p >= 0.5
    return {"classe": "falsa" if falsa else "verdadeira", "prob_fake": round(p, 4),
            "confianca": round(p if falsa else 1 - p, 4)}


def analisar_emocoes(tok, modelo, limiar: float, classes: list, texto: str) -> dict:
    frases = dividir_frases(texto) or [texto]
    saidas = []
    with torch.no_grad():
        for i in range(0, len(frases), 16):
            enc = tok(frases[i:i + 16], truncation=True, max_length=MAX_LEN_EMO,
                      padding=True, return_tensors="pt")
            saidas.append(torch.sigmoid(modelo(**enc).logits.float()).numpy())
    P = np.concatenate(saidas)

    por_frase = []
    for frase, p in zip(frases, P):
        idx = np.where(p >= limiar)[0]
        if len(idx) == 0:
            idx = [int(p.argmax())]
        por_frase.append({"texto": frase,
                          "emocoes": [classes[i] for i in sorted(idx, key=lambda i: -p[i])]})

    perfil = P.mean(axis=0)
    ordem = np.argsort(perfil)[::-1]
    sem_neutro = [i for i in ordem if classes[i] != "neutro"]
    dom = sem_neutro[0] if sem_neutro and perfil[sem_neutro[0]] >= LIMIAR_DOMINANTE else None
    return {"n_frases": len(frases),
            "emocao_dominante": classes[dom] if dom is not None else "neutro",
            "perfil": {classes[i]: round(float(perfil[i]), 3) for i in ordem[:8]},
            "por_frase": por_frase}


# Blocos visuais (HTML com cores explícitas, legível em tema claro e escuro)
def bloco_veredicto(r: dict) -> str:
    falsa = r["classe"] == "falsa"
    forte, claro = FALSA if falsa else VERDADEIRA
    rotulo = "FALSA" if falsa else "VERDADEIRA"
    conf = r["confianca"] * 100
    return f"""
<div style="background:#ffffff;border:1px solid {CINZA_FUNDO};border-radius:12px;padding:20px;{ESPACO}">
  <div style="font-size:0.8rem;color:{TEXTO_SUAVE};letter-spacing:.06em;margin-bottom:10px;">CLASSIFICAÇÃO DA NOTÍCIA</div>
  <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
    <div style="background:{forte};color:#ffffff;font-weight:700;font-size:1.5rem;letter-spacing:.05em;
                padding:14px 28px;border-radius:8px;min-width:190px;text-align:center;">{rotulo}</div>
    <div style="flex:1;min-width:220px;">
      <div style="display:flex;justify-content:space-between;font-size:0.9rem;color:{TEXTO};margin-bottom:6px;">
        <span>Confiança</span><span style="font-weight:700;">{conf:.1f}%</span>
      </div>
      <div style="background:{CINZA_FUNDO};border-radius:6px;height:18px;">
        <div style="width:{conf:.1f}%;background:{forte};height:18px;border-radius:6px;"></div>
      </div>
      <div style="font-size:0.8rem;color:{TEXTO_SUAVE};margin-top:6px;">
        Probabilidade de ser falsa: {r['prob_fake']:.1%}
      </div>
    </div>
  </div>
</div>"""


def bloco_dominante(emocao: str, valor: float, n_frases: int, n_emotivas: int,
                    neutro: float) -> str:
    forte, claro = cor(emocao)
    nota = ""
    if emocao != "neutro" and neutro > valor:
        nota = (f'<div style="font-size:0.85rem;color:{TEXTO_SUAVE};margin-top:8px;">'
                f'O texto é sobretudo neutro ({neutro:.0%}); esta é a emoção mais presente '
                f'entre as restantes.</div>')
    return f"""
<div style="background:{claro};border-left:6px solid {forte};border-radius:0;padding:16px 20px;{ESPACO}">
  <div style="font-size:0.8rem;color:{TEXTO_SUAVE};letter-spacing:.06em;">EMOÇÃO DOMINANTE</div>
  <div style="font-size:1.7rem;font-weight:700;color:{forte};text-transform:capitalize;margin:2px 0 10px;">{nome(emocao)}</div>
  <div style="display:flex;gap:32px;flex-wrap:wrap;font-size:0.9rem;color:{TEXTO};">
    <div>Presença ao longo do texto: <b>{valor:.0%}</b></div>
    <div>Frases com emoção identificada: <b>{n_emotivas} de {n_frases}</b></div>
  </div>{nota}
</div>"""


def bloco_informacao() -> str:
    familias = []
    for familia, emocoes in FAMILIA.items():
        forte, claro = CORES[familia]
        lista = ", ".join(nome(e) for e in emocoes)
        familias.append(
            f'<div style="background:{claro};border-left:4px solid {forte};padding:6px 10px;'
            f'margin:6px 0;border-radius:0;color:{TEXTO};font-size:0.88rem;">'
            f'<b style="color:{forte};">{familia.capitalize()}</b>: {lista}</div>')
    caixa = f"background:#ffffff;border:1px solid {CINZA_FUNDO};border-radius:10px;padding:12px 16px;"
    return f"""
<div style="{caixa}color:{TEXTO};font-size:0.92rem;">
  <b>Notícias falsas</b><br>
  O modelo classifica a notícia como <b>Falsa</b> ou <b>Verdadeira</b>.
  A confiança mostra o grau de certeza do modelo, não é uma verificação de factos.
</div>
<div style="height:10px;background:{CINZA_FUNDO};margin:12px 0;border-radius:4px;"></div>
<div style="{caixa}color:{TEXTO};font-size:0.92rem;">
  <b>Sentimentos</b><br>
  Cada frase é analisada à parte. A emoção dominante é a mais presente no texto, sem contar com o neutro.
  {"".join(familias)}
</div>"""


def bloco_perfil(perfil: dict, n: int = 6) -> str:
    itens = list(perfil.items())[:n]
    maximo = max(v for _, v in itens) or 1
    linhas = []
    for emocao, v in itens:
        forte, _ = cor(emocao)
        linhas.append(f"""
<div style="display:flex;align-items:center;gap:12px;margin:7px 0;">
  <div style="width:130px;font-size:0.95rem;color:{TEXTO};text-transform:capitalize;">{nome(emocao)}</div>
  <div style="flex:1;background:{CINZA_FUNDO};border-radius:6px;height:14px;">
    <div style="width:{100 * v / maximo:.0f}%;background:{forte};height:14px;border-radius:6px;"></div>
  </div>
  <div style="width:48px;text-align:right;font-size:0.9rem;color:{TEXTO};">{v:.0%}</div>
</div>""")
    return (f'<div style="background:#ffffff;border:1px solid {CINZA_FUNDO};border-radius:12px;'
            f'padding:14px 20px;{ESPACO}">{"".join(linhas)}</div>')


def bloco_mapa(por_frase: list) -> str:
    partes = []
    for f in por_frase:
        forte, claro = cor(f["emocoes"][0])
        etiqueta = ", ".join(nome(e) for e in f["emocoes"][:2])
        partes.append(
            f'<span style="background:{claro};color:{TEXTO};border-bottom:3px solid {forte};'
            f'padding:2px 4px;border-radius:4px;line-height:2.2;">{html.escape(f["texto"])}'
            f'<sup style="color:{forte};font-size:0.72rem;font-weight:600;margin-left:4px;'
            f'white-space:nowrap;">{etiqueta}</sup></span> ')
    legenda = " &nbsp; ".join(
        f'<span style="background:{claro};color:{TEXTO};border-bottom:3px solid {forte};'
        f'padding:1px 8px;border-radius:4px;font-size:0.85rem;">{familia.capitalize()}</span>'
        for familia, (forte, claro) in CORES.items())
    return (f'<div style="background:#ffffff;border:1px solid {CINZA_FUNDO};border-radius:12px;'
            f'padding:14px 20px;{ESPACO}"><div style="margin-bottom:10px;">{legenda}</div>'
            f'<div style="font-size:1rem;">{"".join(partes)}</div></div>')


# Página
st.set_page_config(page_title="Análise de Notícias", layout="centered",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarResizeHandle"] { display: none !important; }
section[data-testid="stSidebar"] { width: 300px !important; min-width: 300px !important;
                                   max-width: 300px !important; }
</style>""", unsafe_allow_html=True)

st.markdown(
    f'<h1 style="text-align:center;margin-bottom:0;">Análise de Notícias</h1>'
    f'<p style="text-align:center;color:{TEXTO_SUAVE};font-size:0.95rem;margin-top:4px;">'
    f'Análise de Notícias Falsas e Análise de Sentimentos em Português</p>',
    unsafe_allow_html=True)

if "texto" not in st.session_state:
    st.session_state.texto = ""


exemplos = carregar_exemplos()


def usar_exemplo():
    escolha = st.session_state.get("exemplo")
    if escolha and OPCOES_EXEMPLO[escolha] in exemplos:
        st.session_state.texto = exemplos[OPCOES_EXEMPLO[escolha]]["texto"]


with st.sidebar:
    st.markdown("### Exemplos")
    st.selectbox("Exemplos", [o for o, k in OPCOES_EXEMPLO.items() if k in exemplos], index=None, key="exemplo",
                 placeholder="Escolher uma notícia", on_change=usar_exemplo,
                 label_visibility="collapsed")
    st.markdown("### Modelos")
    st.markdown(f"**Notícias falsas:** [BERTimbau]({LINK_HF_FAKE})")
    st.markdown(f"**Sentimentos:** [XLM-RoBERTa]({LINK_HF_EMO})")
    st.markdown("### Projeto")
    st.markdown(f"[Repositório no GitHub]({LINK_GITHUB})")
    st.caption("Demonstração online. A versão do projeto corre localmente, sem Internet "
               "e sem chamadas a APIs.")


with st.expander("Informação sobre a análise"):
    st.markdown(bloco_informacao(), unsafe_allow_html=True)

st.markdown("**Introduzir Notícia:**")
texto = st.text_area("Introduzir Notícia", key="texto", height=220, label_visibility="collapsed",
                     placeholder="Introduz aqui o título e o texto da notícia")

st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
analisar = st.button("Análise de Notícias", type="primary", use_container_width=True)

if analisar:
    if len(texto.split()) < 5:
        st.warning("Introduz pelo menos uma frase completa para analisar.")
        st.stop()

    with st.spinner("Analisar..."):
        tok_f, mod_f, tok_e, mod_e, limiar, classes = carregar_modelos()
        r_fake = prever_fake(tok_f, mod_f, texto)
        r_emo = analisar_emocoes(tok_e, mod_e, limiar, classes, texto)

    emotivas = sum(1 for f in r_emo["por_frase"] if f["emocoes"][0] != "neutro")
    dominante = r_emo["emocao_dominante"]

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    st.markdown("#### Notícia falsa ou verdadeira")
    st.markdown(bloco_veredicto(r_fake), unsafe_allow_html=True)

    st.markdown("#### Análise de sentimentos")
    st.markdown(bloco_dominante(dominante, r_emo["perfil"].get(dominante, 0.0),
                                r_emo["n_frases"], emotivas, r_emo["perfil"].get("neutro", 0.0)),
                unsafe_allow_html=True)
    st.markdown("**Perfil Emocional**")
    st.markdown(bloco_perfil(r_emo["perfil"]), unsafe_allow_html=True)

    st.markdown("#### Mapa Emocional")
    st.markdown(bloco_mapa(r_emo["por_frase"]), unsafe_allow_html=True)

    with st.expander("Detalhes da análise"):
        st.write(f"**Modelo de notícias falsas:** `{REPO_FAKE}`  |  "
                 f"**Modelo de sentimentos:** `{REPO_EMO}`  |  limiar das emoções: {limiar}")
        st.json({"noticias_falsas": r_fake,
                 "sentimentos": {k: v for k, v in r_emo.items() if k != "por_frase"}})

    st.caption("Resultado de modelos estatísticos treinados com dados em português; "
               "não substitui a verificação de factos.")