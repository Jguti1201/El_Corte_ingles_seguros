# 🌿 Centro de Innovación en IA — Seguros El Corte Inglés

> Aplicación de análisis predictivo y estrategia de Inteligencia Artificial desarrollada en el contexto de la alianza estratégica entre **Seguros El Corte Inglés** y **Mutua Madrileña**.

---

## 📋 Descripción

Esta aplicación Streamlit simula una herramienta interna de alto nivel para el nuevo **Departamento de Inteligencia Artificial** de Seguros El Corte Inglés. Combina análisis de datos real, modelos de Machine Learning, IA Generativa (OpenAI) y visión estratégica de negocio asegurador.

Está construida con identidad visual corporativa de El Corte Inglés (verde institucional, tipografía Playfair Display, diseño ejecutivo) y orientada a perfiles tanto técnicos como directivos.

---

## 🗂️ Estructura de la aplicación

La app tiene **4 páginas** accesibles desde el menú lateral:

### 🟢 Caso 1 · Predicción de Siniestros (`insurance_claims.csv`)
- Definición del problema de negocio e hipótesis
- EDA completo: distribuciones, análisis fraude vs legítimo, mapa de correlaciones
- Feature Engineering con Label Encoding y balanceo de clases
- Modelo **Random Forest Classifier** (holdout 80/20 estratificado)
- Evaluación: Accuracy, Precision, Recall, F1, ROC-AUC, Matriz de Confusión, Curva ROC
- Importancia de variables (Top 12)
- Explicación ejecutiva generada por **GPT-4o-mini**

### 🔵 Caso 2 · Detección de Fraude (`insurance_fraud_data.csv`)
- Hipótesis de detección de fraude pre-pago
- EDA enfocado: balance de clases, patrones por canal, lugar del accidente, edad del conductor
- Modelo **Random Forest** con ajuste de threshold (0.50 → 0.35) para maximizar Recall
- Comparativa de métricas estándar vs optimizadas
- Cálculo de impacto económico estimado (fraudes no detectados en €)
- Explicación ejecutiva con análisis ético generada por **GPT-4o-mini**

### 🟣 Plan 30-60-90 días
- Timeline visual interactivo de las tres fases
- **Días 1-30:** Auditoría de datos, mapa de procesos, gobierno del dato, evaluación de madurez IA
- **Días 31-60:** Pilotos antifraude, clasificación documental, asistente RAG, automatizaciones low-code
- **Días 61-90:** Arquitectura cloud/MLOps, framework IA responsable, comité IA, roadmap anual
- Generación de carta ejecutiva de presentación con IA

### 🟠 8 Propuestas Estratégicas de IA
- 8 casos de uso detallados con problema, solución, impacto, complejidad y riesgos
- Matriz de priorización interactiva Impacto vs Complejidad
- Pitch ejecutivo generado por IA para presentación ante el Consejo de Administración

---

## 🛠️ Tecnologías utilizadas

| Categoría | Librerías |
|-----------|-----------|
| Framework web | `streamlit` |
| Machine Learning | `scikit-learn` |
| Balanceo de clases | `imbalanced-learn` (SMOTE) |
| Datos | `pandas`, `numpy` |
| Visualización | `plotly`, `matplotlib`, `seaborn` |
| IA Generativa | `openai` (GPT-4o-mini) |

---

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/eci-mutua-ia-app.git
cd eci-mutua-ia-app
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Preparar los archivos de datos

Coloca los siguientes archivos en la raíz del proyecto:

```
📁 eci-mutua-ia-app/
├── eci_mutua_ia_app.py
├── insurance_claims.csv          ← obligatorio
├── insurance_fraud_data.csv      ← obligatorio
├── elcorteingles.png             ← opcional (logo en sidebar)
├── requirements.txt
└── README.md
```

### 4. Ejecutar la aplicación

```bash
streamlit run eci_mutua_ia_app.py
```

---

## 🔑 Configuración de la API Key de OpenAI

La app usa **GPT-4o-mini** para generar explicaciones ejecutivas en lenguaje no técnico. Hay dos formas de configurar la API key:

**Opción A — Desde la interfaz** (más rápido para demos):
Introduce tu API key directamente en el campo del panel lateral al abrir la app.

**Opción B — Desde `secrets.toml`** (recomendado para producción):

Crea el archivo `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-proj-..."
```

> ⚠️ Nunca subas tu API key a GitHub. El archivo `.streamlit/secrets.toml` está incluido en `.gitignore`.

---

## 📁 Archivos necesarios

| Archivo | Descripción | Obligatorio |
|---------|-------------|-------------|
| `insurance_claims.csv` | Dataset de siniestros con variable `fraud_reported` | ✅ Sí |
| `insurance_fraud_data.csv` | Dataset de reclamaciones con detección de fraude | ✅ Sí |
| `elcorteingles.png` | Logo corporativo para el sidebar | ❌ Opcional |

---

## ⚙️ Compatibilidad

| Requisito | Versión mínima |
|-----------|----------------|
| Python | 3.9+ |
| pandas | 2.0+ |
| streamlit | 1.32+ |
| scikit-learn | 1.3+ |

> **Nota sobre pandas 2.x:** El código usa `select_dtypes(include=["object"])` en lugar de `["object", "str"]` para compatibilidad con pandas 2.0+.

---

## 🏗️ Arquitectura del modelo

### Caso 1 — Clasificación de fraude en siniestros
- **Algoritmo:** Random Forest Classifier
- **Preprocesado:** Label Encoding de categóricas, eliminación de identificadores y fechas
- **Balanceo:** `class_weight='balanced'`
- **Validación:** Holdout estratificado 80/20
- **Métricas principales:** ROC-AUC, Recall, F1

### Caso 2 — Detección de fraude en reclamaciones
- **Algoritmo:** Random Forest Classifier
- **Preprocesado:** Label Encoding, imputación de `age_of_vehicle`
- **Balanceo:** SMOTE (si disponible) + `class_weight='balanced'`
- **Ajuste de threshold:** 0.35 (optimizado para maximizar Recall)
- **Validación:** Holdout estratificado 80/20

---

## 🤖 IA Generativa — Sistema de Explicaciones

El sistema prompt está diseñado para traducir resultados técnicos a lenguaje ejecutivo. Cada explicación sigue una estructura fija de 5 secciones:

1. **🎯 Qué hemos construido** — Descripción accesible del modelo
2. **📊 Qué nos dicen los resultados** — Métricas traducidas a consecuencias de negocio
3. **💡 Por qué funciona** — Patrones aprendidos en contexto asegurador
4. **⚠️ Limitaciones honestas** — Casos no cubiertos y riesgos
5. **🚀 Próximo paso recomendado** — Acción concreta y accionable

---

## 🎨 Identidad visual

La aplicación implementa la paleta corporativa de El Corte Inglés mediante CSS personalizado:

| Color | Hex | Uso |
|-------|-----|-----|
| Verde oscuro | `#1a5c38` | Header, sidebar, botones, bordes |
| Verde medio | `#2e7d4f` | Elementos secundarios, hover |
| Verde claro | `#4caf7d` | Acentos, gráficos |
| Verde pálido | `#e8f5ee` | Fondos de tarjetas |
| Dorado | `#c8a84b` | Acento premium, KPIs destacados |

Tipografía: **Playfair Display** (títulos) + **Source Sans 3** (cuerpo)

---

## 📄 Licencia

Proyecto desarrollado con fines demostrativos en el contexto de un proceso de selección para el Departamento de IA de Seguros El Corte Inglés. No contiene datos reales de clientes ni información confidencial de la compañía.

---

## 👤 Autor

**Jaime Gutiérrez de Calderón**  
Senior Data Scientist · Especialista en IA aplicada al sector asegurador  
[LinkedIn](https://linkedin.com/in/tu-perfil) · [GitHub](https://github.com/tu-usuario)
