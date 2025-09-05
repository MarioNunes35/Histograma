# Streamlit – Histogramas (com KDE, ECDF, agrupamento e controles finos)
# Author: ChatGPT (GPT-5 Thinking)
# Descrição:
#   App para criar histogramas de forma flexível:
#   • Bins: Auto (Freedman–Diaconis), Sturges, Scott, Doane, Rice, √N, largura fixa ou nº de bins.
#   • Normalização: contagem, frequência, densidade (PDF), probabilidade (%).
#   • Cumulativo opcional; orientação vertical/horizontal; eixos log.
#   • Agrupamento por categoria ou por arquivo (múltiplos CSVs) – modo sobreposto/empilhado/agrupado.
#   • Suavização opcional via KDE (gaussian_kde) por grupo; ECDF opcional.
#   • Tratamento de outliers: recorte por quantis ou winsorização; faixa manual.
#   • Exporta dados processados, tabela de bins/contagens e figura HTML/PNG.

import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde, skew
import plotly.io as pio

st.set_page_config(page_title="Histogramas • KDE • ECDF", page_icon="📊", layout="wide")

st.title("📊 Histogramas avançados (KDE • ECDF • Agrupamento)")
st.caption("Carregue 1+ CSVs, escolha colunas numéricas e gere histogramas com controles de bins, normalização, KDE e mais.")

# ------------------------------ Leitura CSV ------------------------------ #
@st.cache_data
def robust_read_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    bio = BytesIO(file_bytes)
    for kw in [dict(), dict(sep=';'), dict(decimal=','), dict(sep=';', decimal=',')]:
        try:
            bio.seek(0)
            return pd.read_csv(bio, **kw)
        except Exception:
            continue
    bio.seek(0)
    return pd.read_csv(bio, engine='python')

# ------------------------------ Sidebar --------------------------------- #
st.sidebar.header("Entrada de dados")
files = st.sidebar.file_uploader("Arquivos CSV (1 ou mais)", type=["csv", "txt"], accept_multiple_files=True)
if not files:
    st.info("Envie ao menos 1 CSV.")
    st.stop()

# Carregar e combinar
frames = []
for f in files:
    df = robust_read_csv(f.getvalue())
    df["__file__"] = f.name
    frames.append(df)
raw = pd.concat(frames, axis=0, ignore_index=True)

cols_all = list(raw.columns)
num_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in cols_all if c not in num_cols and c != "__file__"]

st.sidebar.subheader("Seleção de dados")
col_value = st.sidebar.selectbox("Coluna numérica (valor)", num_cols if num_cols else cols_all)

# Agrupamento: por arquivo ou coluna categórica
group_mode = st.sidebar.radio("Agrupar por", ["Nenhum", "Coluna categórica", "Arquivo (cada CSV = grupo)"])
if group_mode == "Coluna categórica":
    if not cat_cols:
        st.warning("Não encontrei colunas categóricas; selecione outro modo.")
        group_mode = "Nenhum"
    else:
        col_group = st.sidebar.selectbox("Coluna de grupo", cat_cols)
else:
    col_group = None

# Filtros / tratamento de outliers
st.sidebar.subheader("Faixa e outliers")
use_range = st.sidebar.checkbox("Definir faixa manual")
if use_range:
    x_min = float(raw[col_value].quantile(0.01))
    x_max = float(raw[col_value].quantile(0.99))
    vmin, vmax = st.sidebar.slider("Faixa X", float(raw[col_value].min()), float(raw[col_value].max()), (x_min, x_max))
else:
    vmin, vmax = (None, None)

out_mode = st.sidebar.selectbox("Tratamento de outliers", ["Nenhum", "Recortar por quantis", "Winsorizar"], index=0)
q_low = st.sidebar.slider("Quantil inferior", 0.0, 0.2, 0.01)
q_high = st.sidebar.slider("Quantil superior", 0.8, 1.0, 0.99)

# Binning
st.sidebar.header("Bins")
bin_strategy = st.sidebar.selectbox("Estratégia", [
    "Auto (Freedman–Diaconis)", "Sturges", "Scott", "Doane", "Rice", "√N", "Largura fixa", "Número de bins"
])
bin_width = st.sidebar.number_input("Largura (se aplicável)", value=1.0, min_value=1e-9, step=0.1, format="%.6f")
nbins = st.sidebar.number_input("# bins (se aplicável)", value=30, min_value=1, step=1)

# Normalização e plot
st.sidebar.header("Plotagem & estilo")
histnorm = st.sidebar.selectbox("Normalização", ["count", "probability", "percent", "density"], index=0)
barmode = st.sidebar.selectbox("Modo quando houver grupos", ["overlay", "stack", "group"], index=0)
orientation = st.sidebar.selectbox("Orientação", ["vertical", "horizontal"], index=0)
cumulative = st.sidebar.checkbox("Cumulativo", value=False)
logx = st.sidebar.checkbox("Eixo X log", value=False)
logy = st.sidebar.checkbox("Eixo Y log", value=False)
palette = st.sidebar.selectbox("Paleta", ["Plotly", "Viridis", "Cividis", "Plasma", "Turbo"], index=0)
opacity = st.sidebar.slider("Opacidade das barras", 0.1, 1.0, 0.8)

# KDE e ECDF
st.sidebar.header("Curvas auxiliares")
show_kde = st.sidebar.checkbox("Mostrar KDE (densidade)", value=False)
bandwidth = st.sidebar.number_input("KDE: bandwidth (auto≈0)", value=0.0, min_value=0.0, step=0.1)
show_ecdf = st.sidebar.checkbox("Mostrar ECDF (curva acumulada empírica)", value=False)

# --------------------------- Pré-processamento --------------------------- #
# Selecionar dados e aplicar filtros/outliers
work = raw[[col_value]].copy()
if col_group is not None:
    work[col_group] = raw[col_group]
if group_mode == "Arquivo (cada CSV = grupo)":
    work["__grp__"] = raw["__file__"]
elif group_mode == "Coluna categórica":
    work["__grp__"] = raw[col_group].astype(str)
else:
    work["__grp__"] = "_all_"

# Range manual
if vmin is not None and vmax is not None:
    work = work[(work[col_value] >= vmin) & (work[col_value] <= vmax)]

# Outliers
if out_mode != "Nenhum":
    g = work.groupby("__grp__")
    rows = []
    for grp, df in g:
        x = df[col_value].to_numpy().astype(float)
        lo = np.quantile(x, q_low)
        hi = np.quantile(x, q_high)
        if out_mode == "Recortar por quantis":
            m = (x >= lo) & (x <= hi)
            x2 = x[m]
        else:  # Winsorizar
            x2 = x.copy()
            x2[x2 < lo] = lo
            x2[x2 > hi] = hi
        tmp = pd.DataFrame({col_value: x2})
        tmp["__grp__"] = grp
        rows.append(tmp)
    work = pd.concat(rows, ignore_index=True)

# ------------------------------- Bins ------------------------------------ #

def freedman_diaconis_bins(x: np.ndarray) -> Tuple[float, int, float, float]:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return 1.0, 1, float(np.nanmin(x, initial=0.0)), float(np.nanmax(x, initial=1.0))
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        iqr = np.std(x) or 1.0
    h = 2.0 * iqr * (n ** (-1/3))
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if h <= 0:
        h = (xmax - xmin) / max(10, int(np.sqrt(n)))
    k = int(np.ceil((xmax - xmin) / h)) if h > 0 else int(np.sqrt(n))
    return float(h), int(max(k, 1)), xmin, xmax


def doane_bins(x: np.ndarray) -> int:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return max(1, int(np.sqrt(n)))
    g1 = skew(x)
    sigma_g1 = np.sqrt(6*(n-2)/((n+1)*(n+3)))
    return int(np.ceil(1 + np.log2(n) + np.log2(1 + abs(g1)/sigma_g1)))


def compute_bins(x: np.ndarray, strategy: str, nbins_user: int, width_user: float, vmin: Optional[float], vmax: Optional[float]):
    x = x[~np.isnan(x)]
    xmin = float(np.min(x)) if vmin is None else float(vmin)
    xmax = float(np.max(x)) if vmax is None else float(vmax)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if strategy == "Auto (Freedman–Diaconis)":
        h, k, _, _ = freedman_diaconis_bins(x)
        size = h
        start = xmin
        end = xmax
    elif strategy == "Sturges":
        k = int(np.ceil(np.log2(len(x)) + 1))
        size = (xmax - xmin) / max(k, 1)
        start, end = xmin, xmax
    elif strategy == "Scott":
        sigma = np.std(x)
        h = 3.5 * sigma * (len(x) ** (-1/3)) if len(x) > 0 else 1.0
        size = h if h > 0 else (xmax - xmin) / max(int(np.sqrt(len(x))), 1)
        start, end = xmin, xmax
    elif strategy == "Doane":
        k = doane_bins(x)
        size = (xmax - xmin) / max(k, 1)
        start, end = xmin, xmax
    elif strategy == "Rice":
        k = int(np.ceil(2 * (len(x) ** (1/3))))
        size = (xmax - xmin) / max(k, 1)
        start, end = xmin, xmax
    elif strategy == "√N":
        k = int(np.ceil(np.sqrt(len(x))))
        size = (xmax - xmin) / max(k, 1)
        start, end = xmin, xmax
    elif strategy == "Largura fixa":
        size = float(width_user)
        start = xmin
        end = xmax
    else:  # Número de bins
        k = int(max(1, nbins_user))
        size = (xmax - xmin) / k
        start, end = xmin, xmax
    # Garantir tamanho positivo
    size = max(size, (xmax - xmin) / 1000.0)
    return dict(start=start, end=end, size=size)

# ------------------------------- Plot ------------------------------------ #
# Paletas
if palette == "Plotly":
    colorway = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
elif palette == "Viridis":
    colorway = ["#440154","#482878","#3E4989","#31688E","#26828E","#1F9E89","#35B779","#6DCD59","#B4DE2C","#FDE725"]
elif palette == "Cividis":
    colorway = ["#00224E","#25366F","#3F4A89","#5A5D9C","#7370A3","#8B84A2","#A29A98","#B9B08D","#D0C781","#E8DF74"]
elif palette == "Plasma":
    colorway = ["#0d0887","#6a00a8","#b12a90","#e16462","#fca636","#fcffa4"]
else:
    colorway = ["#30123B","#4145AB","#2CA6D8","#2AD4A5","#7CE080","#F9F871","#F6C64F","#F08E3E","#E84F3D","#D61E3C"]

fig = go.Figure()
fig.update_layout(template='plotly_dark', colorway=colorway)

# Gerar hist por grupos
groups = sorted(work["__grp__"].unique())

xbins = compute_bins(work[col_value].to_numpy().astype(float), bin_strategy, int(nbins), float(bin_width), vmin, vmax)

for i, gname in enumerate(groups):
    data = work.loc[work["__grp__"] == gname, col_value].astype(float).to_numpy()
    # Histograma
    fig.add_trace(go.Histogram(
        x=None if orientation == "horizontal" else data,
        y=data if orientation == "horizontal" else None,
        xbins=xbins if orientation == "vertical" else None,
        ybins=xbins if orientation == "horizontal" else None,
        histnorm=histnorm,
        name=str(gname) if gname != "_all_" else "dados",
        opacity=opacity,
        cumulative_enabled=cumulative,
        nbinsx=None if orientation == "vertical" else None,
        nbinsy=None if orientation == "horizontal" else None,
    ))

# Layout geral
if orientation == "vertical":
    fig.update_xaxes(title_text=col_value, type='log' if logx else 'linear')
    fig.update_yaxes(title_text={"count":"Contagem","probability":"Prob","percent":"%","density":"Densidade"}[histnorm], type='log' if logy else 'linear')
else:
    fig.update_yaxes(title_text=col_value, type='log' if logy else 'linear')
    fig.update_xaxes(title_text={"count":"Contagem","probability":"Prob","percent":"%","density":"Densidade"}[histnorm], type='log' if logx else 'linear')

fig.update_layout(barmode=barmode, height=560, bargap=0.05, bargroupgap=0.02, legend_title="Grupo")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------- KDE & ECDF --------------------------------- #
aux_tabs = st.tabs(["KDE" if show_kde else "KDE (desativado)", "ECDF" if show_ecdf else "ECDF (desativado)"])

# KDE
with aux_tabs[0]:
    if show_kde:
        xbins_line = xbins
        grid = np.linspace(xbins_line['start'], xbins_line['end'], 500)
        fig_kde = go.Figure()
        fig_kde.update_layout(template='plotly_dark', colorway=colorway, height=420)
        for gname in groups:
            x = work.loc[work["__grp__"] == gname, col_value].astype(float).to_numpy()
            x = x[~np.isnan(x)]
            if len(x) < 2:
                continue
            kde = gaussian_kde(x) if bandwidth <= 0 else gaussian_kde(x, bw_method=bandwidth)
            y = kde(grid)
            fig_kde.add_trace(go.Scatter(x=grid, y=y, mode='lines', name=str(gname)))
        fig_kde.update_xaxes(title_text=col_value, type='log' if logx else 'linear')
        fig_kde.update_yaxes(title_text='Densidade (KDE)')
        st.plotly_chart(fig_kde, use_container_width=True)
    else:
        st.info("Ative a opção 'Mostrar KDE' na barra lateral para ver esta aba.")

# ECDF
with aux_tabs[1]:
    if show_ecdf:
        fig_ecdf = go.Figure()
        fig_ecdf.update_layout(template='plotly_dark', colorway=colorway, height=420)
        for gname in groups:
            x = work.loc[work["__grp__"] == gname, col_value].astype(float).to_numpy()
            x = np.sort(x[np.isfinite(x)])
            if len(x) == 0:
                continue
            y = np.arange(1, len(x)+1) / len(x)
            if orientation == 'vertical':
                fig_ecdf.add_trace(go.Scatter(x=x, y=y, mode='lines', name=str(gname)))
                fig_ecdf.update_xaxes(title_text=col_value, type='log' if logx else 'linear')
                fig_ecdf.update_yaxes(title_text='F(x)')
            else:
                fig_ecdf.add_trace(go.Scatter(x=y, y=x, mode='lines', name=str(gname)))
                fig_ecdf.update_xaxes(title_text='F(x)')
                fig_ecdf.update_yaxes(title_text=col_value, type='log' if logy else 'linear')
        st.plotly_chart(fig_ecdf, use_container_width=True)
    else:
        st.info("Ative a opção 'Mostrar ECDF' na barra lateral para ver esta aba.")

# --------------------------- Estatísticas rápidas ------------------------ #
st.subheader("Estatísticas por grupo")
def stats_one(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "média": np.nan, "mediana": np.nan, "desvio": np.nan, "mín": np.nan, "máx": np.nan, "cv%": np.nan}
    return {
        "n": int(len(x)),
        "média": float(np.mean(x)),
        "mediana": float(np.median(x)),
        "desvio": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "mín": float(np.min(x)),
        "máx": float(np.max(x)),
        "cv%": float(100*np.std(x, ddof=1)/np.mean(x)) if np.mean(x) != 0 and len(x) > 1 else np.nan,
    }

rows = []
for gname in groups:
    x = work.loc[work["__grp__"] == gname, col_value].astype(float).to_numpy()
    s = stats_one(x)
    s.update({"Grupo": gname})
    rows.append(s)
if rows:
    st.dataframe(pd.DataFrame(rows)[["Grupo", "n", "média", "mediana", "desvio", "cv%", "mín", "máx"]], use_container_width=True)

# ------------------------------- Export ---------------------------------- #
st.subheader("Exportar")
# Dados processados (após filtros/outliers)
out_buf = io.StringIO(); work.to_csv(out_buf, index=False)
st.download_button("⬇️ CSV – dados processados", out_buf.getvalue(), file_name="histogram_processed.csv", mime="text/csv")

# Figura principal
html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
st.download_button("⬇️ HTML interativo – histograma", data=html, file_name="histogram.html")

png_ok = st.checkbox("Exportar PNG (requer 'kaleido')", value=False)
if png_ok:
    try:
        import kaleido  # noqa: F401
        png_bytes = pio.to_image(fig, format='png', scale=2)
        st.download_button("⬇️ PNG – histograma", data=png_bytes, file_name="histogram.png")
    except Exception as e:
        st.warning(f"PNG indisponível: instale 'kaleido'. Erro: {e}")

# ------------------------------- Notas ----------------------------------- #
with st.expander("Notas & Boas Práticas"):
    st.markdown(
        """
        - **Bins**: *Freedman–Diaconis* se adapta bem a distribuições com caudas; *Doane* ajusta melhor para assimetria (skew). 
        - **Normalização**: `count` (contagens), `probability` (probabilidade), `percent` (%), `density` (área=1).
        - **KDE**: estimativa suave da densidade; o parâmetro *bandwidth* controla o alisamento (0=auto do SciPy).
        - **ECDF**: útil para comparar distribuições acumuladas e quantis.
        - **Outliers**: prefira winsorizar para manter tamanho amostral; use recorte por quantis quando quiser descartar valores.
        """
    )
