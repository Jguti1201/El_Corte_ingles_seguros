"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CENTRO DE INNOVACIÓN EN INTELIGENCIA ARTIFICIAL                           ║
║  Seguros El Corte Inglés · Alianza Mutua Madrileña                         ║
║  Aplicación interna de análisis y predicción de siniestros                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Para ejecutar:
    pip install streamlit openai pandas numpy scikit-learn matplotlib seaborn plotly imbalanced-learn xgboost
    streamlit run eci_mutua_ia_app.py

Coloca en la misma carpeta:
    - insurance_claims.csv
    - insurance_fraud_data.csv
    - logo_el_corte_ingles.png  (opcional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Centro IA · Seguros El Corte Inglés",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETA CORPORATIVA EL CORTE INGLÉS
# ─────────────────────────────────────────────────────────────────────────────
ECI_GREEN_DARK  = "#1a5c38"   # Verde corporativo oscuro
ECI_GREEN_MID   = "#2e7d4f"   # Verde medio
ECI_GREEN_LIGHT = "#4caf7d"   # Verde claro
ECI_GREEN_PALE  = "#e8f5ee"   # Fondo verde muy suave
ECI_GOLD        = "#c8a84b"   # Dorado acento premium
ECI_WHITE       = "#ffffff"
ECI_DARK        = "#1a1a1a"
ECI_GRAY        = "#6b7280"
ECI_LIGHT_GRAY  = "#f4f5f4"

ECI_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

/* ── Reset global ── */
html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif;
    color: {ECI_DARK};
}}

/* ── Fondo principal ── */
.main .block-container {{
    background: {ECI_WHITE};
    padding: 2rem 3rem;
    max-width: 1380px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {ECI_GREEN_DARK};
}}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {{
    color: {ECI_WHITE} !important;
}}
section[data-testid="stSidebar"] .stRadio > label {{
    color: {ECI_WHITE} !important;
    font-weight: 500;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
    color: {ECI_WHITE} !important;
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 6px 10px;
    margin: 2px 0;
    transition: background 0.2s;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
    background: rgba(255,255,255,0.18);
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.2) !important;
}}

/* ── Header corporativo ── */
.eci-header {{
    background: linear-gradient(135deg, {ECI_GREEN_DARK} 0%, {ECI_GREEN_MID} 70%, {ECI_GREEN_LIGHT} 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 6px 28px rgba(26,92,56,0.22);
    position: relative;
    overflow: hidden;
}}
.eci-header::after {{
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}}
.eci-header-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: {ECI_WHITE};
    margin: 0;
    line-height: 1.2;
}}
.eci-header-sub {{
    color: rgba(255,255,255,0.78);
    font-size: 0.9rem;
    margin: 0.3rem 0 0 0;
    font-weight: 300;
    letter-spacing: 0.5px;
}}
.eci-badge {{
    background: {ECI_GOLD};
    color: {ECI_WHITE};
    font-family: 'Source Sans 3';
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 0.6rem;
}}

/* ── Section headers ── */
.sec-header {{
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: {ECI_GREEN_DARK};
    border-left: 4px solid {ECI_GREEN_DARK};
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}}
.sec-sub {{
    font-size: 0.9rem;
    color: {ECI_GRAY};
    margin-bottom: 1.2rem;
    line-height: 1.6;
}}

/* ── KPI Cards ── */
.kpi-card {{
    background: {ECI_WHITE};
    border: 1px solid #e2e8e4;
    border-top: 4px solid {ECI_GREEN_DARK};
    border-radius: 10px;
    padding: 1.4rem;
    box-shadow: 0 2px 10px rgba(26,92,56,0.07);
    transition: transform 0.18s, box-shadow 0.18s;
    margin-bottom: 1rem;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(26,92,56,0.13);
}}
.kpi-card.gold {{ border-top-color: {ECI_GOLD}; }}
.kpi-card.light {{ border-top-color: {ECI_GREEN_LIGHT}; }}
.kpi-value {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: {ECI_GREEN_DARK};
    line-height: 1;
}}
.kpi-label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {ECI_GRAY};
    font-weight: 600;
    margin-bottom: 0.4rem;
}}
.kpi-delta {{
    font-size: 0.85rem;
    color: {ECI_GREEN_LIGHT};
    font-weight: 600;
    margin-top: 0.3rem;
}}

/* ── Insight boxes ── */
.insight-box {{
    background: {ECI_GREEN_PALE};
    border: 1px solid #c3dfd0;
    border-left: 5px solid {ECI_GREEN_DARK};
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}}
.insight-box p {{ margin: 0; font-size: 0.92rem; line-height: 1.65; }}
.insight-box strong {{ color: {ECI_GREEN_DARK}; }}

/* ── AI Explanation box ── */
.ai-box {{
    background: linear-gradient(135deg, #f0f9f4, #fafdf7);
    border: 1px solid {ECI_GREEN_LIGHT};
    border-radius: 12px;
    padding: 1.8rem;
    margin: 1rem 0;
    position: relative;
}}
.ai-box::before {{
    content: '✦ IA Generativa · Explicación ejecutiva';
    font-size: 0.68rem;
    font-weight: 700;
    color: {ECI_WHITE};
    letter-spacing: 2px;
    text-transform: uppercase;
    background: {ECI_GREEN_DARK};
    padding: 3px 14px;
    border-radius: 20px;
    position: absolute;
    top: -11px;
    left: 18px;
}}
.ai-box p {{
    color: {ECI_DARK};
    line-height: 1.8;
    font-size: 0.94rem;
    margin: 0;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {ECI_GREEN_DARK} !important;
    color: {ECI_WHITE} !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Source Sans 3' !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.8rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 3px 10px rgba(26,92,56,0.3) !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    background: {ECI_GREEN_MID} !important;
    box-shadow: 0 6px 16px rgba(26,92,56,0.4) !important;
    transform: translateY(-1px) !important;
}}

/* ── Plan 30-60-90 cards ── */
.plan-card {{
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}}
.plan-card.p30 {{ background: linear-gradient(135deg, {ECI_GREEN_PALE}, #d4edde); border-left: 6px solid {ECI_GREEN_LIGHT}; }}
.plan-card.p60 {{ background: linear-gradient(135deg, #eef6e9, {ECI_GREEN_PALE}); border-left: 6px solid {ECI_GREEN_MID}; }}
.plan-card.p90 {{ background: linear-gradient(135deg, #e6f0ea, #dceee4); border-left: 6px solid {ECI_GREEN_DARK}; }}
.plan-card h3 {{
    font-family: 'Playfair Display', serif;
    color: {ECI_GREEN_DARK};
    margin: 0 0 1rem 0;
    font-size: 1.2rem;
}}
.plan-tag {{
    display: inline-block;
    background: {ECI_GREEN_DARK};
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 2px 12px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
}}

/* ── Caso de uso cards ── */
.caso-card {{
    background: {ECI_WHITE};
    border: 1px solid #dde8e3;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(26,92,56,0.06);
    transition: box-shadow 0.2s;
}}
.caso-card:hover {{ box-shadow: 0 6px 20px rgba(26,92,56,0.12); }}
.caso-card h4 {{
    font-family: 'Playfair Display', serif;
    color: {ECI_GREEN_DARK};
    margin: 0 0 0.8rem 0;
    font-size: 1.05rem;
}}
.caso-tag {{
    display: inline-block;
    background: {ECI_GREEN_PALE};
    color: {ECI_GREEN_DARK};
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 4px;
    margin-bottom: 4px;
    border: 1px solid #c3dfd0;
}}
.caso-gold {{ background: #fdf6e3; border-color: {ECI_GOLD}; color: #8a6914; }}
.complexity {{
    font-size: 0.8rem;
    color: {ECI_GRAY};
    margin-top: 0.5rem;
}}

/* ── Tables ── */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}

/* ── Footer ── */
.eci-footer {{
    background: {ECI_GREEN_DARK};
    color: rgba(255,255,255,0.65);
    text-align: center;
    padding: 1.3rem;
    border-radius: 10px;
    margin-top: 3rem;
    font-size: 0.82rem;
    letter-spacing: 0.3px;
}}
.eci-footer strong {{ color: {ECI_GOLD}; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: {ECI_GREEN_DARK} !important; }}

/* ── Divider ── */
hr {{ border-color: #dde8e3 !important; }}
</style>
"""

st.markdown(ECI_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: COLORES PARA GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

PLOT_PALETTE = [ECI_GREEN_DARK, ECI_GREEN_MID, ECI_GREEN_LIGHT, ECI_GOLD,
                "#6bb89a", "#a8d5bc", "#8aab97", "#3d7a5c"]

def eci_plotly_theme():
    return dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Source Sans 3", color=ECI_DARK),
        title_font=dict(family="Source Sans Pro", size=15, color=ECI_GREEN_DARK),
        colorway=PLOT_PALETTE
    )


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI
# ─────────────────────────────────────────────────────────────────────────────

def get_openai_explanation(prompt_content: str, api_key: str) -> str:
    """Llama a GPT-4o-mini y devuelve explicación ejecutiva detallada."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres el Director de Inteligencia Artificial del Centro de Innovación IA "
                        "de Seguros El Corte Inglés, en alianza estratégica con Mutua Madrileña. "
                        "Tu misión es explicar los resultados de modelos de Inteligencia Artificial "
                        "a personas sin conocimientos técnicos: directivos, comités de dirección, "
                        "responsables de negocio y equipos comerciales.\n\n"

                        "CÓMO DEBES COMUNICAR:\n"
                        "- Usa siempre analogías del mundo real para explicar conceptos técnicos. "
                        "Por ejemplo: 'El modelo funciona como un perito con 10 años de experiencia "
                        "que ha revisado miles de expedientes y ha aprendido a detectar señales de alerta.'\n"
                        "- Traduce SIEMPRE cada métrica a consecuencias concretas. Nunca digas solo "
                        "'el Recall es 0.72'. Di: 'De cada 100 fraudes reales, el modelo detecta 72 "
                        "antes de que se paguen. Los 28 restantes pasarían inadvertidos.'\n"
                        "- Usa euros y porcentajes de impacto económico siempre que sea posible.\n"
                        "- Explica el razonamiento del modelo: qué variables usa, por qué tienen "
                        "sentido desde el punto de vista del negocio asegurador.\n"
                        "- Menciona tanto lo que el modelo hace bien como sus limitaciones honestas.\n"
                        "- Incluye siempre una recomendación clara de próximo paso para el negocio.\n\n"

                        "ESTRUCTURA DE TU RESPUESTA (usa siempre este formato):\n"
                        "1. 🎯 QUÉ HEMOS CONSTRUIDO — Explica en 2-3 frases qué hace el modelo, "
                        "como si se lo explicaras a alguien que nunca ha oído hablar de Machine Learning.\n"
                        "2. 📊 QUÉ NOS DICEN LOS RESULTADOS — Traduce cada métrica clave a lenguaje "
                        "de negocio con ejemplos concretos y cifras reales (si las tienes).\n"
                        "3. 💡 POR QUÉ FUNCIONA — Explica qué señales o patrones ha aprendido el modelo "
                        "y por qué tienen sentido en el contexto asegurador.\n"
                        "4. ⚠️ LIMITACIONES HONESTAS — Qué casos no cubre bien el modelo, cuándo "
                        "puede equivocarse y qué vigilar. La transparencia genera confianza.\n"
                        "5. 🚀 PRÓXIMO PASO RECOMENDADO — Una acción concreta y accionable que el "
                        "equipo debería tomar a partir de estos resultados.\n\n"

                        "TONO: Ejecutivo pero cercano. Riguroso pero accesible. "
                        "Nunca uses jerga estadística sin explicarla inmediatamente. "
                        "Responde siempre en español. Extensión: entre 350 y 500 palabras."
                    )
                },
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.25,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ La conexión con el modelo de IA no está disponible: {str(e)}\n\nVerifica la API key en los ajustes."


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPROCESADO DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_claims():
    """Carga y limpia insurance_claims.csv."""
    paths = ["insurance_claims.csv", "/mnt/user-data/uploads/insurance_claims.csv"]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    else:
        st.error("No se encontró insurance_claims.csv"); st.stop()

    df = df.drop(columns=["_c39", "policy_number", "insured_zip",
                           "incident_location", "incident_date",
                           "policy_bind_date", "auto_model"], errors="ignore")
    df["fraud_reported"] = (df["fraud_reported"] == "Y").astype(int)
    df = df.dropna()
    return df


@st.cache_data
def load_fraud():
    """Carga y limpia insurance_fraud_data.csv."""
    paths = ["insurance_fraud_data.csv", "/mnt/user-data/uploads/insurance_fraud_data.csv"]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    else:
        st.error("No se encontró insurance_fraud_data.csv"); st.stop()

    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.dropna(subset=["fraud_reported"])
    df["fraud_reported"] = (df["fraud_reported"] == "Y").astype(int)
    df["age_of_vehicle"] = pd.to_numeric(df["age_of_vehicle"], errors="coerce")
    df["age_of_vehicle"] = df["age_of_vehicle"].fillna(df["age_of_vehicle"].median())
    df = df.drop(columns=["claim_number", "claim_date"], errors="ignore")
    return df


@st.cache_data
def prepare_claims_model(df_raw):
    """Prepara features y entrena modelo sobre claims."""
    df = df_raw.copy()
    # Ahora (compatible con todas las versiones)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    X = df.drop("fraud_reported", axis=1)
    y = df["fraud_reported"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return rf, metrics, feat_imp, fpr, tpr, cm, X_test, y_test, y_pred, y_proba


@st.cache_data
def prepare_fraud_model(df_raw):
    """Prepara features y entrena modelo antifraude."""
    df = df_raw.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    X = df.drop("fraud_reported", axis=1)
    y = df["fraud_reported"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Intentar SMOTE; si no está instalado, usar class_weight
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        resampled = True
    except ImportError:
        X_res, y_res = X_train, y_train
        resampled = False

    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_res, y_res)
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # Ajuste de threshold a 0.35 para maximizar recall en fraude
    threshold = 0.35
    y_pred_adj = (y_proba >= threshold).astype(int)

    metrics_std = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    metrics_adj = {
        "accuracy":  accuracy_score(y_test, y_pred_adj),
        "precision": precision_score(y_test, y_pred_adj, zero_division=0),
        "recall":    recall_score(y_test, y_pred_adj, zero_division=0),
        "f1":        f1_score(y_test, y_pred_adj, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm_adj = confusion_matrix(y_test, y_pred_adj)

    return rf, metrics_std, metrics_adj, feat_imp, fpr, tpr, cm_adj, X_test, y_test, y_pred_adj, y_proba, resampled


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo
    logo_paths = ["elcorteingles.png", "logo_eci.png"]
    logo_found = False
    for lp in logo_paths:
        if os.path.exists(lp):
            st.image(lp, width=200)
            logo_found = True
            break
    if not logo_found:
        st.markdown(f"""
        <div style='text-align:center; padding:1.5rem 0 0.5rem;'>
            <div style='background:white; display:inline-block; padding:10px 18px; border-radius:8px;'>
                <span style='font-family:serif; font-size:1.1rem; font-weight:700; color:{ECI_GREEN_DARK};'>
                    EL CORTE INGLÉS
                </span><br>
                <span style='font-size:0.65rem; color:{ECI_GOLD}; letter-spacing:2px; font-weight:600;'>
                    SEGUROS · IA
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <p style='text-align:center; color:rgba(255,255,255,0.5); font-size:0.7rem; letter-spacing:1px;
              text-transform:uppercase; margin: 0 0 0.8rem 0;'>
        Centro de Innovación IA
    </p>
    <hr style='border-color:rgba(255,255,255,0.15); margin:0 0 1rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navegación", [
        "🟢  Caso 1 · Siniestros",
        "🔵  Caso 2 · Antifraude",
        "🟣  Plan 30-60-90 días",
        "🟠  8 Propuestas Estratégicas"
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    api_key = st.text_input("🔑 OpenAI API Key", type="password",
                             placeholder="sk-...",
                             help="Para activar las explicaciones de IA generativa")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except:
            api_key = ""

    st.markdown(f"""
    <div style='text-align:center; color:rgba(255,255,255,0.4); font-size:0.72rem;
                margin-top:2rem; line-height:1.7;'>
        <strong style='color:rgba(255,255,255,0.7);'>Seguros El Corte Inglés</strong><br>
        Alianza Mutua Madrileña<br>
        Departamento IA & Innovación<br>
        <span style='color:{ECI_GOLD};'>v1.0 · 2025</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER CORPORATIVO (aparece en todas las páginas)
# ─────────────────────────────────────────────────────────────────────────────

page_titles = {
    "🟢  Caso 1 · Siniestros":        ("Predicción de Siniestros de Alto Coste", "Modelo de clasificación de riesgo · Dataset Insurance Claims"),
    "🔵  Caso 2 · Antifraude":         ("Sistema de Detección de Fraude", "Motor predictivo antifraude · Dataset Insurance Fraud"),
    "🟣  Plan 30-60-90 días":          ("Plan de Implantación del Departamento IA", "Hoja de ruta estratégica 90 días · Alianza Mutua + ECI"),
    "🟠  8 Propuestas Estratégicas":   ("8 Casos de Uso Estratégicos de IA", "Innovación, automatización y IA generativa · Seguros ECI"),
}
h1, h2 = page_titles.get(page, ("Centro IA", ""))
st.markdown(f"""
<div class='eci-header'>
    <div>
        <p class='eci-header-title'>{h1}</p>
        <p class='eci-header-sub'>{h2}</p>
        <span class='eci-badge'>Centro de Innovación IA · Seguros El Corte Inglés</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — CASO 1: INSURANCE CLAIMS
# ═════════════════════════════════════════════════════════════════════════════

if page == "🟢  Caso 1 · Siniestros":

    df = load_claims()

    # ── 1. Definición del problema ─────────────────────────────────────────
    st.markdown("<div class='sec-header'>1 · Definición del Problema de Negocio</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <p><strong>Hipótesis central:</strong> Es posible predecir, en el momento de apertura del expediente,
    si un siniestro tiene alta probabilidad de ser <em>fraudulento</em>, permitiendo a Seguros ECI 
    priorizar la investigación manual y reducir el pago indebido de siniestros irregulares.</p>
    <p style='margin-top:0.8rem;'><strong>Impacto en negocio:</strong> El fraude en seguros representa entre el 8% y el 12% de las primas 
    emitidas en el mercado español. Un sistema predictivo con alta precisión permite redirigir el 
    esfuerzo investigador a los expedientes con mayor riesgo, optimizando recursos y reduciendo 
    el ratio de siniestralidad.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    fraud_rate = df["fraud_reported"].mean()
    avg_claim  = df["total_claim_amount"].mean()
    with col1:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Total Expedientes</div>
        <div class='kpi-value'>{len(df):,}</div><div class='kpi-delta'>Dataset completo</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='kpi-card gold'><div class='kpi-label'>Tasa de Fraude</div>
        <div class='kpi-value'>{fraud_rate:.1%}</div><div class='kpi-delta'>Casos reportados como fraude</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='kpi-card light'><div class='kpi-label'>Siniestro Medio</div>
        <div class='kpi-value'>${avg_claim:,.0f}</div><div class='kpi-delta'>Total claim amount</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Variables Disponibles</div>
        <div class='kpi-value'>{df.shape[1]-1}</div><div class='kpi-delta'>Features tras limpieza</div></div>""", unsafe_allow_html=True)

    # ── 2. EDA ────────────────────────────────────────────────────────────
    st.markdown("<div class='sec-header'>2 · Análisis Exploratorio de Datos</div>", unsafe_allow_html=True)

    tab_eda1, tab_eda2, tab_eda3, tab_eda4 = st.tabs(
        ["📊 Distribuciones", "🔍 Fraude vs No Fraude", "🌡️ Correlaciones", "🗂️ Datos brutos"])

    with tab_eda1:
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.histogram(df, x="total_claim_amount", nbins=40,
                               color_discrete_sequence=[ECI_GREEN_DARK],
                               title="Distribución del Importe Total del Siniestro",
                               labels={"total_claim_amount": "Importe ($)", "count": "Frecuencia"})
            fig.update_layout(**eci_plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            fig2 = px.histogram(df, x="age", nbins=30,
                                color_discrete_sequence=[ECI_GREEN_MID],
                                title="Distribución de Edad del Asegurado",
                                labels={"age": "Edad", "count": "Frecuencia"})
            fig2.update_layout(**eci_plotly_theme())
            st.plotly_chart(fig2, use_container_width=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            sev_counts = df["incident_severity"].value_counts().reset_index()
            sev_counts.columns = ["Severidad", "Expedientes"]
            fig3 = px.bar(sev_counts, x="Severidad", y="Expedientes",
                          color="Expedientes", color_continuous_scale=[[0, ECI_GREEN_PALE],[1, ECI_GREEN_DARK]],
                          title="Expedientes por Severidad del Incidente")
            fig3.update_layout(**eci_plotly_theme(), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)
        with col_r2:
            type_counts = df["incident_type"].value_counts().reset_index()
            type_counts.columns = ["Tipo", "Expedientes"]
            fig4 = px.pie(type_counts, values="Expedientes", names="Tipo",
                          color_discrete_sequence=PLOT_PALETTE,
                          title="Tipo de Incidente")
            fig4.update_layout(**eci_plotly_theme())
            st.plotly_chart(fig4, use_container_width=True)

    with tab_eda2:
        col_l, col_r = st.columns(2)
        with col_l:
            fig5 = px.box(df, x="fraud_reported", y="total_claim_amount",
                          color="fraud_reported",
                          color_discrete_map={0: ECI_GREEN_LIGHT, 1: ECI_GOLD},
                          labels={"fraud_reported": "Fraude (1=Sí)", "total_claim_amount": "Importe ($)"},
                          title="Importe del Siniestro: Fraude vs No Fraude")
            fig5.update_layout(**eci_plotly_theme(), showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
        with col_r:
            fraud_sev = df.groupby(["incident_severity","fraud_reported"]).size().reset_index(name="n")
            fig6 = px.bar(fraud_sev, x="incident_severity", y="n", color="fraud_reported",
                          barmode="group",
                          color_discrete_map={0: ECI_GREEN_LIGHT, 1: ECI_GOLD},
                          labels={"incident_severity": "Severidad", "n": "Expedientes",
                                  "fraud_reported": "Fraude"},
                          title="Severidad vs Fraude")
            fig6.update_layout(**eci_plotly_theme())
            st.plotly_chart(fig6, use_container_width=True)

        fraud_hour = df.groupby("incident_hour_of_the_day")["fraud_reported"].mean().reset_index()
        fraud_hour.columns = ["Hora", "Tasa Fraude"]
        fig7 = px.line(fraud_hour, x="Hora", y="Tasa Fraude",
                       line_shape="spline",
                       color_discrete_sequence=[ECI_GREEN_DARK],
                       title="Tasa de Fraude por Hora del Incidente",
                       markers=True)
        fig7.update_layout(**eci_plotly_theme())
        fig7.add_hline(y=fraud_rate, line_dash="dot", line_color=ECI_GOLD,
                       annotation_text=f"Media global: {fraud_rate:.1%}")
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown("""<div class='insight-box'><p>
        <strong>Insights clave:</strong> Los siniestros declarados a horas intempestivas (00–04h y 20–24h)
        tienen tasas de fraude significativamente superiores a la media. Las colisiones de un solo vehículo
        y los daños mayores concentran más casos fraudulentos. Estos patrones son coherentes con la
        literatura aseguradora sobre indicadores de alerta temprana.</p></div>""", unsafe_allow_html=True)

    with tab_eda3:
        num_df = df.select_dtypes(include="number").drop(columns=["fraud_reported"], errors="ignore")
        corr   = num_df.corr()
        fig_corr, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(145, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
                    linewidths=0.4, ax=ax, annot_kws={"size": 8})
        ax.set_title("Mapa de Correlaciones — Variables Numéricas", fontsize=13,
                     color=ECI_GREEN_DARK, fontfamily="serif", pad=14)
        fig_corr.patch.set_facecolor("white")
        st.pyplot(fig_corr, use_container_width=True)

    with tab_eda4:
        st.dataframe(df.head(50), use_container_width=True, height=380)
        st.caption(f"Mostrando 50 de {len(df):,} registros")

    # ── 3. Feature Engineering ────────────────────────────────────────────
    st.markdown("<div class='sec-header'>3 · Feature Engineering</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sec-sub'>
    Las variables categóricas se codifican mediante Label Encoding. 
    Se eliminan identificadores y fechas que no aportan señal predictiva.
    La variable objetivo es <code>fraud_reported</code> (binaria: 1=fraude, 0=legítimo).
    El modelo se entrena con pesos de clase balanceados para compensar el desequilibrio (75/25).
    </div>""", unsafe_allow_html=True)

    # ── 4 & 5. Modelo y Evaluación ────────────────────────────────────────
    st.markdown("<div class='sec-header'>4 · Modelo Predictivo y Evaluación</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sec-sub'>
    <strong>Algoritmo elegido: Random Forest Classifier.</strong> Justificación: robusto ante variables mixtas 
    (numéricas y categóricas codificadas), maneja bien el desequilibrio de clases con <em>class_weight='balanced'</em>,
    ofrece importancia de variables interpretable y no requiere normalización. 
    Validación: holdout estratificado 80/20.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Entrenando modelo de clasificación..."):
        rf_c, metrics_c, feat_imp_c, fpr_c, tpr_c, cm_c, Xte_c, yte_c, yp_c, yprob_c = prepare_claims_model(df)

    m = metrics_c
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, label, val, suffix in [
        (col1, "Accuracy",  m["accuracy"],  ""),
        (col2, "Precision", m["precision"], ""),
        (col3, "Recall",    m["recall"],    ""),
        (col4, "F1 Score",  m["f1"],        ""),
        (col5, "ROC-AUC",   m["roc_auc"],   ""),
    ]:
        with col:
            color = "gold" if label == "ROC-AUC" else ""
            st.markdown(f"""<div class='kpi-card {color}'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{val:.3f}</div>
            </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Matriz de confusión
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        cmap_cm = sns.light_palette(ECI_GREEN_DARK, as_cmap=True)
        sns.heatmap(cm_c, annot=True, fmt="d", cmap=cmap_cm, ax=ax,
                    xticklabels=["Legítimo", "Fraude"],
                    yticklabels=["Legítimo", "Fraude"],
                    linewidths=1, linecolor="white")
        ax.set_title("Matriz de Confusión", color=ECI_GREEN_DARK, fontfamily="serif", fontsize=12)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        fig_cm.patch.set_facecolor("white")
        st.pyplot(fig_cm, use_container_width=True)

    with col_r:
        # ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_c, y=tpr_c, mode="lines",
                                      name=f"ROC (AUC={m['roc_auc']:.3f})",
                                      line=dict(color=ECI_GREEN_DARK, width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                      name="Aleatorio", line=dict(dash="dot", color=ECI_GOLD, width=1.5)))
        fig_roc.update_layout(title="Curva ROC", xaxis_title="Tasa Falsos Positivos",
                               yaxis_title="Tasa Verdaderos Positivos",
                               **eci_plotly_theme(), height=350, legend=dict(x=0.55, y=0.1))
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature importance
    top_feat = feat_imp_c.head(12).reset_index()
    top_feat.columns = ["Feature", "Importancia"]
    fig_fi = px.bar(top_feat, x="Importancia", y="Feature", orientation="h",
                    color="Importancia",
                    color_continuous_scale=[[0, ECI_GREEN_PALE],[1, ECI_GREEN_DARK]],
                    title="Importancia de Variables — Top 12")
    fig_fi.update_layout(**eci_plotly_theme(), yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(f"""<div class='insight-box'><p>
    <strong>Interpretación ejecutiva:</strong> El modelo alcanza un AUC de <strong>{m['roc_auc']:.3f}</strong>,
    lo que significa que discrimina correctamente entre siniestros fraudulentos y legítimos en más del
    {m['roc_auc']*100:.1f}% de los casos. El Recall de <strong>{m['recall']:.1%}</strong> indica que se detecta
    ese porcentaje de los fraudes reales. Las variables de mayor poder predictivo son el importe del siniestro,
    la antigüedad del cliente y la severidad del incidente.
    </p></div>""", unsafe_allow_html=True)

    # ── 6. IA Generativa ─────────────────────────────────────────────────
    st.markdown("<div class='sec-header'>6 · Explicación con IA Generativa</div>", unsafe_allow_html=True)

    prompt_claims = f"""
    Nuestro modelo de Random Forest para detección de siniestros de alto riesgo ha obtenido los siguientes resultados:
    - Accuracy: {m['accuracy']:.3f}
    - Precision: {m['precision']:.3f} (de cada 10 alertas, {m['precision']*10:.0f} son reales)
    - Recall: {m['recall']:.3f} (detectamos el {m['recall']*100:.0f}% de los fraudes reales)
    - F1 Score: {m['f1']:.3f}
    - ROC-AUC: {m['roc_auc']:.3f}
    - Falsos Negativos (fraudes no detectados): {cm_c[1][0]:,}
    - Falsos Positivos (legítimos marcados como fraude): {cm_c[0][1]:,}
    - Variables más importantes: {", ".join(feat_imp_c.head(5).index.tolist())}
    
    Explica qué significan estos resultados para el Comité de Dirección de Seguros El Corte Inglés.
    Incluye el impacto económico estimado y las implicaciones operativas para el equipo de peritos.
    """

    if st.button("🤖 Explícame estos resultados como si no supiera de datos", key="btn_claims_ai"):
        if not api_key:
            st.warning("Introduce tu API key de OpenAI en el panel lateral.")
        else:
            with st.spinner("Generando explicación ejecutiva..."):
                expl = get_openai_explanation(prompt_claims, api_key)
            st.markdown(f"<div class='ai-box'><p>{expl}</p></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — CASO 2: ANTIFRAUDE
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔵  Caso 2 · Antifraude":

    df_f = load_fraud()

    # ── 1. Hipótesis ──────────────────────────────────────────────────────
    st.markdown("<div class='sec-header'>1 · Hipótesis y Objetivo de Negocio</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <p><strong>Hipótesis:</strong> Usando variables del siniestro, el perfil del conductor y del vehículo,
    podemos predecir la probabilidad de fraude <em>antes de autorizar el pago</em>, activando 
    un flujo de investigación diferenciado para los expedientes de mayor riesgo.</p>
    <p style='margin-top:0.8rem;'><strong>Impacto:</strong> Cada punto porcentual de mejora en la detección de fraude
    supone ahorros directos en la cuenta de resultados. El coste de un falso negativo (fraude no detectado)
    es entre 5x y 15x mayor que el coste operativo de investigar un falso positivo.</p>
    </div>
    """, unsafe_allow_html=True)

    fraud_rate_f = df_f["fraud_reported"].mean()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Total Reclamaciones</div>
        <div class='kpi-value'>{len(df_f):,}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='kpi-card gold'><div class='kpi-label'>Tasa de Fraude</div>
        <div class='kpi-value'>{fraud_rate_f:.1%}</div></div>""", unsafe_allow_html=True)
    with col3:
        avg_total = df_f["total_claim"].mean()
        st.markdown(f"""<div class='kpi-card light'><div class='kpi-label'>Reclamación Media</div>
        <div class='kpi-value'>${avg_total:,.0f}</div></div>""", unsafe_allow_html=True)
    with col4:
        fraud_cost = df_f[df_f["fraud_reported"]==1]["total_claim"].sum()
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Exposición Fraude</div>
        <div class='kpi-value'>${fraud_cost/1e6:.1f}M</div></div>""", unsafe_allow_html=True)

    # ── 2. EDA antifraude ─────────────────────────────────────────────────
    st.markdown("<div class='sec-header'>2 · EDA Enfocado en Detección de Fraude</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Balance y Patrones", "🔍 Variables Clave", "🗂️ Datos"])

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            labels = ["Legítimo (N)", "Fraude (Y)"]
            vals   = df_f["fraud_reported"].value_counts().sort_index().values
            fig_pie = go.Figure(go.Pie(labels=labels, values=vals,
                                       marker_colors=[ECI_GREEN_LIGHT, ECI_GOLD],
                                       hole=0.45))
            fig_pie.update_layout(title="Balance de Clases", **eci_plotly_theme(), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_r:
            fig_box = px.box(df_f, x="fraud_reported", y="total_claim",
                              color="fraud_reported",
                              color_discrete_map={0: ECI_GREEN_LIGHT, 1: ECI_GOLD},
                              labels={"fraud_reported": "Fraude", "total_claim": "Importe ($)"},
                              title="Importe Reclamado: Fraude vs Legítimo")
            fig_box.update_layout(**eci_plotly_theme(), showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # Fraude por canal
        channel_fraud = df_f.groupby("channel")["fraud_reported"].agg(["mean","count"]).reset_index()
        channel_fraud.columns = ["Canal", "Tasa Fraude", "N"]
        fig_ch = px.bar(channel_fraud, x="Canal", y="Tasa Fraude",
                        color="Tasa Fraude",
                        color_continuous_scale=[[0, ECI_GREEN_PALE],[1, ECI_GREEN_DARK]],
                        title="Tasa de Fraude por Canal de Contratación",
                        text="Tasa Fraude")
        fig_ch.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_ch.update_layout(**eci_plotly_theme(), coloraxis_showscale=False)
        st.plotly_chart(fig_ch, use_container_width=True)

    with tab2:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_age = px.histogram(df_f, x="age_of_driver", color="fraud_reported",
                                   barmode="overlay", opacity=0.7,
                                   color_discrete_map={0: ECI_GREEN_LIGHT, 1: ECI_GOLD},
                                   nbins=30,
                                   labels={"age_of_driver": "Edad del Conductor",
                                           "fraud_reported": "Fraude"},
                                   title="Distribución de Edad por Fraude")
            fig_age.update_layout(**eci_plotly_theme())
            st.plotly_chart(fig_age, use_container_width=True)
        with col_r:
            acc_fraud = df_f.groupby("accident_site")["fraud_reported"].mean().reset_index()
            acc_fraud.columns = ["Lugar Accidente", "Tasa Fraude"]
            fig_acc = px.bar(acc_fraud, x="Lugar Accidente", y="Tasa Fraude",
                              color_discrete_sequence=[ECI_GREEN_DARK],
                              title="Tasa de Fraude por Lugar del Accidente")
            fig_acc.update_layout(**eci_plotly_theme())
            st.plotly_chart(fig_acc, use_container_width=True)

        corr_fraud = df_f.select_dtypes(include="number").corr()["fraud_reported"].drop("fraud_reported").abs().sort_values(ascending=True).tail(10)
        fig_corr_f = px.bar(x=corr_fraud.values, y=corr_fraud.index, orientation="h",
                             color=corr_fraud.values,
                             color_continuous_scale=[[0, ECI_GREEN_PALE],[1, ECI_GREEN_DARK]],
                             title="Correlación con Fraude (top 10 variables)")
        fig_corr_f.update_layout(**eci_plotly_theme(), coloraxis_showscale=False)
        st.plotly_chart(fig_corr_f, use_container_width=True)

    with tab3:
        st.dataframe(df_f.head(50), use_container_width=True, height=340)

    # ── 3 & 4. Modelo y Evaluación ─────────────────────────────────────
    st.markdown("<div class='sec-header'>3 · Modelo Antifraude y Evaluación Crítica</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sec-sub'>
    <strong>Algoritmo: Random Forest con ajuste de threshold a 0.35.</strong>  
    El umbral estándar (0.5) prioriza precision; bajarlo a 0.35 incrementa el Recall a costa de más
    falsos positivos, estrategia óptima en fraude donde el coste de no detectar un fraude supera
    al de investigar un expediente legítimo adicional.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Entrenando modelo antifraude..."):
        rf_f, m_std, m_adj, fi_f, fpr_f, tpr_f, cm_f, Xte_f, yte_f, yp_f, yprob_f, resampled = prepare_fraud_model(df_f)

    st.markdown("##### Comparativa: Threshold estándar (0.5) vs Optimizado (0.35)")
    comp_df = pd.DataFrame({
        "Métrica":   ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Th=0.50":   [f"{m_std[k]:.3f}" for k in ["accuracy","precision","recall","f1","roc_auc"]],
        "Th=0.35 ✓": [f"{m_adj[k]:.3f}" for k in ["accuracy","precision","recall","f1","roc_auc"]],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig_cm2, ax2 = plt.subplots(figsize=(5, 4))
        cmap_cm2 = sns.light_palette(ECI_GREEN_DARK, as_cmap=True)
        sns.heatmap(cm_f, annot=True, fmt="d", cmap=cmap_cm2, ax=ax2,
                    xticklabels=["Legítimo", "Fraude"], yticklabels=["Legítimo", "Fraude"],
                    linewidths=1, linecolor="white")
        ax2.set_title("Matriz de Confusión (Th=0.35)", color=ECI_GREEN_DARK, fontfamily="serif", fontsize=11)
        ax2.set_xlabel("Predicho"); ax2.set_ylabel("Real")
        fig_cm2.patch.set_facecolor("white")
        st.pyplot(fig_cm2, use_container_width=True)

    with col_r:
        fig_roc2 = go.Figure()
        fig_roc2.add_trace(go.Scatter(x=fpr_f, y=tpr_f, mode="lines",
                                       name=f"Modelo (AUC={m_adj['roc_auc']:.3f})",
                                       line=dict(color=ECI_GREEN_DARK, width=3)))
        fig_roc2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatorio",
                                       line=dict(dash="dot", color=ECI_GOLD, width=1.5)))
        fig_roc2.update_layout(title="Curva ROC — Modelo Antifraude",
                                xaxis_title="FPR", yaxis_title="TPR",
                                **eci_plotly_theme(), height=350)
        st.plotly_chart(fig_roc2, use_container_width=True)

    top_fi_f = fi_f.head(12).reset_index()
    top_fi_f.columns = ["Feature", "Importancia"]
    fig_fi2 = px.bar(top_fi_f, x="Importancia", y="Feature", orientation="h",
                     color="Importancia",
                     color_continuous_scale=[[0, ECI_GREEN_PALE],[1, ECI_GREEN_DARK]],
                     title="Variables Más Relevantes para Detectar Fraude")
    fig_fi2.update_layout(**eci_plotly_theme(), yaxis=dict(autorange="reversed"),
                           coloraxis_showscale=False, height=400)
    st.plotly_chart(fig_fi2, use_container_width=True)

    # Impacto económico
    fn = cm_f[1][0]
    avg_fraud_claim = df_f[df_f["fraud_reported"]==1]["total_claim"].mean()
    economic_loss = fn * avg_fraud_claim
    st.markdown(f"""<div class='insight-box'><p>
    <strong>Impacto económico:</strong> Con el modelo actual, {fn:,} fraudes no son detectados en el conjunto
    de test. Estimando un importe medio de fraude de ${avg_fraud_claim:,.0f}, esto representa una exposición
    de aproximadamente <strong>${economic_loss:,.0f}</strong> en pagos potencialmente indebidos. 
    Cada mejora de 5 puntos en Recall evita ~${avg_fraud_claim*fn*0.05:,.0f} adicionales en pago fraudulento.
    </p></div>""", unsafe_allow_html=True)

    # ── 5. IA Generativa ─────────────────────────────────────────────────
    st.markdown("<div class='sec-header'>5 · Explicación con IA Generativa</div>", unsafe_allow_html=True)

    prompt_fraud = f"""
    Nuestro sistema de detección de fraude antifraude ha obtenido los siguientes resultados (threshold optimizado 0.35):
    - ROC-AUC: {m_adj['roc_auc']:.3f}
    - Recall (fraudes detectados): {m_adj['recall']:.1%} de los fraudes reales
    - Precision: {m_adj['precision']:.3f}
    - Fraudes no detectados (falsos negativos): {cm_f[1][0]:,} casos
    - Exposición económica estimada no cubierta: ${economic_loss:,.0f}
    - Principales señales de fraude detectadas: {", ".join(fi_f.head(5).index.tolist())}
    
    Explica al Comité de Dirección:
    1. Qué significa este sistema en la práctica diaria del tramitador de siniestros
    2. Por qué detectar el {m_adj['recall']*100:.0f}% de los fraudes es un avance significativo
    3. Los riesgos éticos de un sistema así y cómo mitigarlos
    4. El retorno económico esperado
    Usa lenguaje ejecutivo, claro y sin tecnicismos.
    """

    if st.button("🤖 Explícame el modelo antifraude en lenguaje sencillo", key="btn_fraud_ai"):
        if not api_key:
            st.warning("Introduce tu API key de OpenAI en el panel lateral.")
        else:
            with st.spinner("Generando análisis ejecutivo..."):
                expl2 = get_openai_explanation(prompt_fraud, api_key)
            st.markdown(f"<div class='ai-box'><p>{expl2}</p></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — PLAN 30-60-90 DÍAS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🟣  Plan 30-60-90 días":

    st.markdown("<div class='sec-header'>Hoja de Ruta: Departamento de IA · Seguros El Corte Inglés</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sec-sub'>
    Plan estructurado de implantación del nuevo departamento de Inteligencia Artificial, 
    en el marco de la alianza estratégica Mutua Madrileña · El Corte Inglés. 
    Combina IA tradicional, IA generativa, gobierno del dato y cultura de innovación.
    </div>""", unsafe_allow_html=True)

    # Timeline visual
    fig_timeline = go.Figure()
    phases = [("Días 1-30", 0, 30, ECI_GREEN_LIGHT, "Comprensión y Diagnóstico"),
              ("Días 31-60", 30, 60, ECI_GREEN_MID, "Primeros Pilotos"),
              ("Días 61-90", 60, 90, ECI_GREEN_DARK, "Escalado Estratégico")]
    for name, x0, x1, color, desc in phases:
        fig_timeline.add_shape(type="rect", x0=x0, x1=x1, y0=0.2, y1=0.8,
                               fillcolor=color, opacity=0.85, line_width=0)
        fig_timeline.add_annotation(x=(x0+x1)/2, y=0.5, text=f"<b>{name}</b><br>{desc}",
                                    showarrow=False, font=dict(color="white", size=11))
    fig_timeline.update_layout(xaxis=dict(range=[-2,92], showticklabels=False, showgrid=False),
                                yaxis=dict(showticklabels=False, showgrid=False),
                                height=140, margin=dict(l=0,r=0,t=10,b=10),
                                plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── DÍAS 1-30 ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class='plan-card p30'>
        <span class='plan-tag'>Días 1 – 30 · Comprensión y Diagnóstico</span>
        <h3>🌱 Fase de Exploración y Arquitectura de Conocimiento</h3>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;'>
            <div>
                <p><strong>📂 Auditoría de datos y sistemas</strong><br>
                Inventario de fuentes de datos disponibles (core asegurador, CRM, llamadas, 
                documentos escaneados). Evaluación de calidad, linaje y accesibilidad. 
                Identificación de brechas críticas de dato.</p>
                <p><strong>🗺️ Mapa de procesos automatizables</strong><br>
                Entrevistas con tramitadores, peritos y área comercial. 
                Identificación de los 10 procesos con mayor volumen de trabajo manual y 
                menor complejidad cognitiva: candidatos a automatización inmediata.</p>
                <p><strong>🏛️ Gobierno del dato</strong><br>
                Definición del modelo de gobernanza: Data Owner, Data Steward, Data Engineer. 
                Inventario de datos personales y adecuación RGPD. 
                </p>
            </div>
            <div>
                <p><strong>📊 Evaluación de madurez IA</strong><br>
                Assessment de capacidades actuales: herramientas, talento, infraestructura cloud. 
                Benchmark vs mejores prácticas del sector asegurador español. 
                Identificación de quick wins de alto impacto y baja complejidad.</p>
                <p><strong>👥 Formación del equipo fundacional</strong><br>
                Identificación de perfiles internos con afinidad analítica. 
                Definición del equipo: Data Scientists, ML Engineers, AI Product Manager. 
                Plan de contratación y alianzas con proveedores tecnológicos.</p>
                <p><strong>🎯 KPIs del departamento</strong><br>
                Definición de métricas de éxito: coste evitado por fraude detectado, 
                NPS cliente, tiempo de tramitación, tasa de automatización documental. 
                Dashboard de seguimiento de adopción IA.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DÍAS 31-60 ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class='plan-card p60'>
        <span class='plan-tag'>Días 31 – 60 · Primeros Pilotos y Demostración de Valor</span>
        <h3>🚀 Fase de Pilotos IA y Generación de Confianza</h3>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;'>
            <div>
                <p><strong>🔍 Piloto 1: Motor Antifraude (ML)</strong><br>
                Despliegue en producción del modelo Random Forest antifraude 
                en un subconjunto de tramitadores voluntarios. Medición de tasa de 
                detección vs grupo de control. Feedback cualitativo del equipo.</p>
                <p><strong>📄 Piloto 2: Clasificación documental (IA Gen.)</strong><br>
                Automatización de lectura y clasificación de partes de accidente, 
                facturas médicas e informes de taller mediante LLMs (GPT-4o / Claude). 
                Objetivo: reducir 40% el tiempo de entrada de datos.</p>
                <p><strong>💬 Piloto 3: Asistente interno para tramitadores</strong><br>
                Agente conversacional con acceso a base de conocimiento de pólizas, 
                coberturas y precedentes. Basado en RAG (Retrieval Augmented Generation). 
                Plataforma: OpenAI Assistants o LangChain.</p>
            </div>
            <div>
                <p><strong>🤖 Automatizaciones low-code</strong><br>
                Implantación de flujos de trabajo automatizados con Power Automate / Zapier 
                para notificaciones, asignación de peritos y escalados automáticos. 
                ROI rápido sin dependencia de equipos de desarrollo.</p>
                <p><strong>📚 Formación y cultura IA</strong><br>
                Programa de sensibilización IA para toda la organización (4h). 
                Taller avanzado para directivos: uso responsable de IA generativa. 
                Comunidad interna de práctica: AI Champions por departamento.</p>
                <p><strong>📈 Revisión de pilotos y métricas</strong><br>
                Presentación al Comité de Dirección de resultados de los primeros 30 días 
                operativos. Decisión de escalar, pivotar o descartar cada piloto. 
                Ajuste del roadmap según aprendizajes.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DÍAS 61-90 ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class='plan-card p90'>
        <span class='plan-tag'>Días 61 – 90 · Escalado y Arquitectura Estratégica</span>
        <h3>🏗️ Fase de Industrialización y Visión a Largo Plazo</h3>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;'>
            <div>
                <p><strong>☁️ Arquitectura Cloud y MLOps</strong><br>
                Diseño del stack tecnológico definitivo: Azure ML / AWS SageMaker / GCP Vertex. 
                Implantación de CI/CD para modelos: versionado, monitorización de drift, 
                reentrenamiento automático. Feature Store corporativo.</p>
                <p><strong>⚖️ Framework de IA Responsable</strong><br>
                Política interna de IA: explicabilidad, equidad, privacidad by design. 
                Alineación con AI Act europeo. Comité de Ética IA con representación 
                jurídica, negocio y tecnología. Auditorías trimestrales de modelos en producción.</p>
                <p><strong>🔗 Escalado de casos exitosos</strong><br>
                Despliegue completo del motor antifraude a toda la cartera de siniestros. 
                Extensión del asistente documental a nuevas tipologías de siniestro. 
                Integración con el ecosistema El Corte Inglés: datos de cliente 360°.</p>
            </div>
            <div>
                <p><strong>🗺️ Roadmap anual (90-365 días)</strong><br>
                Plan de 12 meses con 3 oleadas de casos de uso, presupuesto aprobado 
                y Objetivos y Resultados Clave definidos por trimestre. Presentación al Consejo de Administración 
                de la visión IA a 3 años de Seguros ECI.</p>
                <p><strong>🏛️ Comité IA Corporativo</strong><br>
                Constitución del órgano de gobernanza IA: Chief Data & Technology Officer, Chief Data Officer, Director IA, 
                representantes de negocio y Compliance. Reunión mensual de revisión 
                de cartera de proyectos IA. Priorización estratégica continua.</p>
                <p><strong>📊 Reporting de impacto</strong><br>
                Dashboard ejecutivo de impacto IA: € ahorrados en fraude, 
                horas automatizadas, NPS asociado a procesos IA, tiempo de 
                tramitación medio. Publicación interna de casos de éxito.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if api_key:
        if st.button("🤖 Generar carta de presentación ejecutiva del Plan 30-60-90", key="btn_plan_ai"):
            prompt_plan = """
            Eres el nuevo Director de IA de Seguros El Corte Inglés, que acaba de incorporarse 
            al nuevo departamento de Inteligencia Artificial creado en el marco de la alianza estratégica 
            con Mutua Madrileña. 
            Redacta una carta ejecutiva de presentación del plan de los primeros 90 días para el 
            Comité de Dirección. Debe ser ambiciosa, realista, orientada al negocio asegurador, 
            y transmitir liderazgo, innovación y rigor. Máximo 300 palabras. Tono ejecutivo y profesional.
            """
            with st.spinner("Generando carta ejecutiva con IA..."):
                expl3 = get_openai_explanation(prompt_plan, api_key)
            st.markdown(f"<div class='ai-box'><p>{expl3}</p></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — 8 PROPUESTAS ESTRATÉGICAS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🟠  8 Propuestas Estratégicas":

    st.markdown("<div class='sec-header'>8 Casos de Uso Estratégicos de IA para Seguros El Corte Inglés</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sec-sub'>
    Propuestas concretas y viables para el despliegue de IA y IA Generativa en el ecosistema 
    asegurador de El Corte Inglés, aprovechando la alianza con Mutua Madrileña y el acceso 
    privilegiado al cliente premium del retail español.
    </div>""", unsafe_allow_html=True)

    casos = [
        {
            "num": "01",
            "emoji": "💰",
            "title": "Motor de Pricing Dinámico con ML",
            "tags": ["Machine Learning", "Pricing", "Rentabilidad"],
            "problema": "Las tarifas actuales se calculan con modelos actuariales estáticos que no incorporan señales de comportamiento en tiempo real ni el contexto de cliente El Corte Inglés.",
            "solucion": "Modelo de Gradient Boosting que combina variables clásicas (siniestralidad, perfil) con datos de comportamiento: frecuencia de compra en ECI, historial de claims, zona geográfica dinámica. Actualización trimestral del modelo con reentrenamiento automático (MLOps).",
            "impacto": "Reducción del ratio combinado en 2-4 puntos porcentuales. Captación de nuevos segmentos con pricing más competitivo. Retención de clientes de bajo riesgo actualmente sobretarifados.",
            "complejidad": "🟡 Media-Alta · 6-9 meses",
            "riesgos": "Regulatorio (Dirección General de Seguros). Equidad algorítmica. Riesgo de selección adversa si el pricing se hace público."
        },
        {
            "num": "02",
            "emoji": "🤖",
            "title": "Asistente Generativo para Tramitadores",
            "tags": ["IA Generativa", "LLM", "Operaciones"],
            "problema": "Los tramitadores de siniestros invierten el 35-40% de su tiempo en consultar manuales, bases de conocimiento y precedentes para resolver casos complejos o poco frecuentes.",
            "solucion": "Agente conversacional basado en RAG (Retrieval Augmented Generation) con acceso a: pólizas digitalizadas, manual de cobertura, resoluciones históricas, normativa ICEA. Integrado en el escritorio del tramitador. Basado en GPT-4o con fine-tuning sobre vocabulario asegurador.",
            "impacto": "Reducción del 30-40% en tiempo de tramitación de consultas complejas. Homogeneización de criterios entre delegaciones. Reducción de errores de cobertura.",
            "complejidad": "🟢 Media · 3-5 meses",
            "riesgos": "Alucinaciones del modelo en normativa específica. Necesidad de supervisión humana en decisiones de cobertura. Gestión del cambio."
        },
        {
            "num": "03",
            "emoji": "🛍️",
            "title": "Sistema de Recomendación Cross-Selling Mutua + ECI",
            "tags": ["ML", "Personalización", "Revenue"],
            "problema": "El cliente de El Corte Inglés tiene un perfil de alto valor conocido (tarjeta ECI, historial de compra) pero este dato no se utiliza para personalizar la oferta aseguradora.",
            "solucion": "Motor de recomendación collaborative filtering + content-based que cruza: productos adquiridos en ECI, ciclo de vida familiar, zona de residencia, y cartera aseguradora actual. Propone el siguiente mejor producto de seguro en el canal más adecuado (app, email, agente).",
            "impacto": "Incremento del ratio de productos por cliente del 1.4 al 2.1. Mejora del CLV estimada en 18-25%. Diferenciación competitiva única en el mercado español.",
            "complejidad": "🟡 Media · 4-6 meses",
            "riesgos": "Privacidad y consentimiento RGPD para cruzar datos retail-seguro. Coordinación con El Corte Inglés corporativo. Riesgo de percepción intrusiva por el cliente."
        },
        {
            "num": "04",
            "emoji": "🚪",
            "title": "Predicción de Abandono de Cliente (Churn)",
            "tags": ["ML", "Retención", "CRM"],
            "problema": "La tasa de no-renovación en el sector asegurador supera el 15% anual. Actualmente no existe un sistema predictivo que identifique clientes en riesgo de abandono antes de que llegue la fecha de vencimiento.",
            "solucion": "Modelo de clasificación (XGBoost) que predice probabilidad de no-renovación con 90 días de antelación, usando: ratio de siniestralidad, interacciones con el servicio de atención, variaciones de prima, score de satisfacción NPS. Activa flujo automatizado de retención personalizado.",
            "impacto": "Reducción de la tasa de churn en 3-5 puntos. Cada punto porcentual equivale a ~€2-4M en primas retenidas. ROI del proyecto estimado en 8-12x en el primer año.",
            "complejidad": "🟢 Media · 3-4 meses",
            "riesgos": "Calidad del dato de interacciones (llamadas, app). Riesgo de fatiga si las acciones de retención son percibidas como agresivas."
        },
        {
            "num": "05",
            "emoji": "📄",
            "title": "Automatización Documental con LLMs",
            "tags": ["IA Generativa", "NLP", "Automatización"],
            "problema": "El proceso de tramitación de siniestros implica la lectura, clasificación y extracción de datos de múltiples documentos: partes de accidente, facturas, informes médicos, presupuestos de taller. Este proceso es 100% manual y propenso a errores.",
            "solucion": "Pipeline de procesamiento documental con visión por computador + LLM: (1) OCR + clasificación automática del documento, (2) extracción de entidades clave (importes, fechas, CIF, diagnósticos) mediante GPT-4o Vision, (3) pre-relleno automático del expediente, (4) flag de anomalías para revisión humana.",
            "impacto": "Reducción del 60-70% del tiempo de entrada de datos. Eliminación de errores de transcripción. Capacidad de tramitar 3x más expedientes con el mismo equipo.",
            "complejidad": "🟢 Media · 4-5 meses",
            "riesgos": "Precisión del OCR en documentos de baja calidad. Responsabilidad legal de datos extraídos automáticamente. Auditoría de los casos no supervisados."
        },
        {
            "num": "06",
            "emoji": "🔎",
            "title": "Detección de Anomalías en Reembolsos Médicos",
            "tags": ["ML", "Antifraude", "Salud"],
            "problema": "En los seguros de salud, los reembolsos de gastos médicos son un vector de fraude sofisticado: facturas duplicadas, diagnósticos inflados, proveedores ficticios. La revisión manual cubre menos del 5% del volumen.",
            "solucion": "Modelo de detección de anomalías no supervisado (Isolation Forest + Autoencoder) que analiza patrones de facturación por proveedor, diagnóstico, importe y frecuencia. Genera un score de riesgo por expediente y activa revisión automática por encima del umbral.",
            "impacto": "Detección estimada de fraude médico adicional del 15-20% sobre el actual. Ahorro potencial de €1.5-3M anuales en la cartera de salud. Identificación de redes de proveedores fraudulentos.",
            "complejidad": "🔴 Alta · 8-12 meses",
            "riesgos": "Complejidad médica del dominio. Riesgo de denegar reembolsos legítimos. Marco ético y legal en salud especialmente sensible."
        },
        {
            "num": "07",
            "emoji": "🧠",
            "title": "Agente de Conocimiento Asegurador Interno",
            "tags": ["IA Generativa", "Agentes", "Knowledge Management"],
            "problema": "El conocimiento experto en Seguros ECI está distribuido: documentos, emails, decisiones pasadas, formaciones. Los nuevos empleados tardan 6-12 meses en ser operativos. El conocimiento crítico reside en pocas personas.",
            "solucion": "Agente IA con memoria persistente y acceso a base de conocimiento corporativa (SharePoint, correos internos con consentimiento, actas de reuniones). Responde preguntas complejas sobre productos, normativa, precedentes. Se enriquece continuamente con las interacciones. Desplegado en Teams/Slack.",
            "impacto": "Reducción del 50% en tiempo de onboarding. Democratización del conocimiento experto. Reducción de dependencia de personas clave. Ahorro en formación externa.",
            "complejidad": "🟡 Media-Alta · 5-7 meses",
            "riesgos": "Gestión de versiones del conocimiento. Riesgo de propagar información desactualizada. Privacidad de comunicaciones internas utilizadas como fuente."
        },
        {
            "num": "08",
            "emoji": "👑",
            "title": "Segmentación Avanzada del Cliente Premium ECI",
            "tags": ["ML", "Segmentación", "Estrategia"],
            "problema": "La segmentación actual de clientes es binaria (cliente activo / inactivo) y no captura la riqueza del perfil del cliente El Corte Inglés: valor de vida, propensión a adquirir seguros, sensibilidad al precio, preferencia de canal.",
            "solucion": "Modelo de clustering avanzado (K-Means + UMAP para visualización) que combina: datos aseguradores, comportamiento en ECI (con acuerdo de data sharing), variables socioeconómicas y geográficas. Genera 6-8 arquetipos de cliente con propuestas de valor diferenciadas para cada segmento.",
            "impacto": "Personalización de producto, precio y comunicación por segmento. Incremento del ratio de conversión en campañas del 25-40%. Identificación del segmento de mayor CLV para proteger y priorizar.",
            "complejidad": "🟢 Media · 3-4 meses",
            "riesgos": "Calidad y completitud del dato de cliente. Necesidad de acuerdo legal para usar datos del retail. Riesgo de segmentación discriminatoria."
        }
    ]

    for caso in casos:
        st.markdown(f"""
        <div class='caso-card'>
            <h4>{caso['emoji']} Caso {caso['num']} · {caso['title']}</h4>
            <div>
                {''.join([f"<span class='caso-tag'>{t}</span>" for t in caso['tags']])}
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.2rem; margin-top:1rem;'>
                <div>
                    <p style='font-size:0.78rem; font-weight:700; color:{ECI_GREEN_DARK}; text-transform:uppercase; 
                               letter-spacing:0.8px; margin-bottom:0.3rem;'>🎯 Problema</p>
                    <p style='font-size:0.88rem; line-height:1.6; margin:0;'>{caso['problema']}</p>
                </div>
                <div>
                    <p style='font-size:0.78rem; font-weight:700; color:{ECI_GREEN_DARK}; text-transform:uppercase;
                               letter-spacing:0.8px; margin-bottom:0.3rem;'>⚙️ Solución IA</p>
                    <p style='font-size:0.88rem; line-height:1.6; margin:0;'>{caso['solucion']}</p>
                </div>
                <div>
                    <p style='font-size:0.78rem; font-weight:700; color:{ECI_GREEN_DARK}; text-transform:uppercase;
                               letter-spacing:0.8px; margin-bottom:0.3rem;'>📈 Impacto</p>
                    <p style='font-size:0.88rem; line-height:1.6; margin:0;'>{caso['impacto']}</p>
                    <p style='font-size:0.78rem; color:{ECI_GRAY}; margin-top:0.5rem;'><strong>Complejidad:</strong> {caso['complejidad']}</p>
                    <p style='font-size:0.78rem; color:{ECI_GRAY}; margin:0;'><strong>⚠️ Riesgos:</strong> {caso['riesgos']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Matriz de priorización
    st.markdown("<div class='sec-header'>Matriz de Priorización Estratégica</div>", unsafe_allow_html=True)

    casos_matrix = pd.DataFrame({
        "Caso": [f"0{i+1}" for i in range(8)],
        "Nombre": [c["title"][:35]+"..." if len(c["title"])>35 else c["title"] for c in casos],
        "Impacto (1-10)": [8, 9, 8, 7, 9, 7, 8, 6],
        "Complejidad (1-10)": [7, 4, 5, 3, 4, 9, 6, 3],
        "Tiempo estimado": ["6-9m", "3-5m", "4-6m", "3-4m", "4-5m", "8-12m", "5-7m", "3-4m"],
    })

    fig_matrix = px.scatter(
        casos_matrix, x="Complejidad (1-10)", y="Impacto (1-10)",
        text="Caso", size=[50]*8,
        color="Impacto (1-10)",
        color_continuous_scale=[[0, ECI_GREEN_PALE],[1, ECI_GREEN_DARK]],
        hover_data=["Nombre", "Tiempo estimado"],
        title="Matriz Impacto vs Complejidad"
    )
    fig_matrix.update_traces(textposition="top center", textfont=dict(size=12, color=ECI_GREEN_DARK))
    fig_matrix.add_hline(y=7.5, line_dash="dot", line_color=ECI_GOLD, line_width=1.5)
    fig_matrix.add_vline(x=5, line_dash="dot", line_color=ECI_GOLD, line_width=1.5)
    fig_matrix.add_annotation(x=2.5, y=9.5, text="QUICK WINS ★", showarrow=False,
                               font=dict(color=ECI_GREEN_DARK, size=10, family="serif"))
    fig_matrix.add_annotation(x=7.5, y=9.5, text="PROYECTOS ESTRATÉGICOS", showarrow=False,
                               font=dict(color=ECI_GOLD, size=10, family="serif"))
    fig_matrix.update_layout(**eci_plotly_theme(), coloraxis_showscale=False, height=450)
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("""<div class='insight-box'><p>
    <strong>Recomendación de secuencia:</strong> Iniciar con los casos 02 (Asistente tramitadores), 
    04 (Churn), 05 (Automatización documental) y 08 (Segmentación) por su alta relación 
    impacto/esfuerzo. En paralelo, preparar la arquitectura de datos para habilitar los casos 
    01 (Pricing dinámico) y 03 (Cross-selling ECI), que requieren integración de datos retail. 
    Los casos 06 y 07 son más complejos y deben planificarse para el segundo semestre.
    </p></div>""", unsafe_allow_html=True)

    if api_key:
        if st.button("🤖 Generar pitch ejecutivo de las 8 propuestas", key="btn_casos_ai"):
            prompt_casos = """
            Como Director de IA de Seguros El Corte Inglés, debes presentar en 3 minutos 
            ante el Consejo de Administración las 8 propuestas de IA que has diseñado para 
            transformar la compañía. Los casos incluyen: motor antifraude, asistente generativo 
            para tramitadores, cross-selling con datos ECI, predicción de abandono, 
            automatización documental, detección de anomalías médicas, agente de conocimiento 
            interno, y segmentación avanzada de cliente premium.
            Redacta el pitch de apertura: impacto transformador, ventaja competitiva única 
            gracias a la alianza con El Corte Inglés, y llamada a la acción. 
            Máximo 250 palabras. Tono: ambicioso, ejecutivo, inspirador.
            """
            with st.spinner("Generando pitch ejecutivo..."):
                expl4 = get_openai_explanation(prompt_casos, api_key)
            st.markdown(f"<div class='ai-box'><p>{expl4}</p></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='eci-footer'>
    <strong>Seguros El Corte Inglés</strong> · Centro de Innovación en Inteligencia Artificial<br>
    En alianza estratégica con <strong>Mutua Madrileña</strong> · Departamento IA & Transformación Digital<br>
    <span style='color:rgba(255,255,255,0.35); font-size:0.75rem;'>
        Uso interno y confidencial · © 2025 El Corte Inglés, S.A.
    </span>
</div>
""", unsafe_allow_html=True)