import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="ML Predictor", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;600&family=Mulish:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Mulish', sans-serif; }
.main { background: #080b12; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    color: #ffffff;
    line-height: 1.1;
    letter-spacing: -1.5px;
}
.hero-accent { color: #00ffcc; }
.hero-sub {
    color: #4a5568;
    font-size: 0.95rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 6px;
}
.step-badge {
    display: inline-block;
    background: #00ffcc22;
    color: #00ffcc;
    border: 1px solid #00ffcc44;
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.card {
    background: #0d1117;
    border: 1px solid #1a2035;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}
.metric-card {
    background: #0d1117;
    border: 1px solid #1a2035;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #3a4a5a;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #00ffcc;
}
.predict-result {
    background: linear-gradient(135deg, #00ffcc11, #0088ff11);
    border: 1px solid #00ffcc33;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.predict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #00ffcc99;
}
.predict-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
}
div[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 2px dashed #1a2a3a;
    border-radius: 12px;
    padding: 16px;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { color: #8899aa !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">No-Code <span class="hero-accent">ML</span> Predictor</div>
<div class="hero-sub">// upload → configure → train → predict</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
for key in ["model", "feature_cols", "target_col", "encoders", "scaler", "task_type", "df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── STEP 1: Upload ─────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 01 — Data</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state.df = df
    st.success(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 2: Configure ──────────────────────────────────────
    st.markdown('<div class="step-badge">Step 02 — Configure</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        target = st.selectbox("🎯 Target column (what to predict)", df.columns.tolist())
    with col_b:
        all_features = [c for c in df.columns if c != target]
        features = st.multiselect("🧩 Feature columns", all_features, default=all_features)
    with col_c:
        task = st.selectbox("📐 Task type", ["Auto-detect", "Classification", "Regression"])

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05,
                          help="Fraction of data held out for evaluation")

    model_options = {
        "Classification": ["Random Forest", "Logistic Regression", "Decision Tree"],
        "Regression":     ["Random Forest", "Linear Regression", "Decision Tree"],
    }

    # Auto-detect task
    if task == "Auto-detect":
        if df[target].dtype == object or df[target].nunique() <= 10:
            detected_task = "Classification"
        else:
            detected_task = "Regression"
        st.caption(f"Auto-detected: **{detected_task}**")
    else:
        detected_task = task

    model_choice = st.selectbox("🤖 Algorithm", model_options[detected_task])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 3: Train ──────────────────────────────────────────
    st.markdown('<div class="step-badge">Step 03 — Train</div>', unsafe_allow_html=True)

    if st.button("🚀 Train Model", use_container_width=True):
        if not features:
            st.error("Please select at least one feature column.")
        else:
            with st.spinner("Training..."):
                try:
                    work = df[features + [target]].dropna().copy()
                    encoders = {}

                    # Encode categoricals in features
                    for col in work[features].select_dtypes(["object", "category"]).columns:
                        le = LabelEncoder()
                        work[col] = le.fit_transform(work[col].astype(str))
                        encoders[col] = le

                    # Encode target if classification
                    if detected_task == "Classification" and work[target].dtype == object:
                        le_target = LabelEncoder()
                        work[target] = le_target.fit_transform(work[target].astype(str))
                        encoders[f"__target_{target}"] = le_target

                    X = work[features].values
                    y = work[target].values

                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    # Pick model
                    if detected_task == "Classification":
                        models = {
                            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                            "Logistic Regression": LogisticRegression(max_iter=500),
                            "Decision Tree": DecisionTreeClassifier(random_state=42),
                        }
                    else:
                        models = {
                            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                            "Linear Regression": LinearRegression(),
                            "Decision Tree": DecisionTreeRegressor(random_state=42),
                        }

                    model = models[model_choice]
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # Store in session
                    st.session_state.model = model
                    st.session_state.feature_cols = features
                    st.session_state.target_col = target
                    st.session_state.encoders = encoders
                    st.session_state.scaler = scaler
                    st.session_state.task_type = detected_task

                    # ── Results ───────────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="step-badge">Results</div>', unsafe_allow_html=True)

                    if detected_task == "Classification":
                        acc = accuracy_score(y_test, preds)
                        m1, m2, m3 = st.columns(3)
                        m1.markdown(f'<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{acc:.1%}</div></div>', unsafe_allow_html=True)
                        m2.markdown(f'<div class="metric-card"><div class="metric-label">Test Samples</div><div class="metric-value">{len(y_test)}</div></div>', unsafe_allow_html=True)
                        m3.markdown(f'<div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">{len(np.unique(y))}</div></div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Confusion matrix
                        cm = confusion_matrix(y_test, preds)
                        fig_cm = px.imshow(cm, text_auto=True, template="plotly_dark",
                                           color_continuous_scale="Teal",
                                           title="Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)

                    else:
                        mae = mean_absolute_error(y_test, preds)
                        r2 = r2_score(y_test, preds)
                        m1, m2, m3 = st.columns(3)
                        m1.markdown(f'<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">{r2:.3f}</div></div>', unsafe_allow_html=True)
                        m2.markdown(f'<div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value">{mae:.2f}</div></div>', unsafe_allow_html=True)
                        m3.markdown(f'<div class="metric-card"><div class="metric-label">Test Samples</div><div class="metric-value">{len(y_test)}</div></div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        fig_pred = px.scatter(x=y_test, y=preds, template="plotly_dark",
                                              labels={"x": "Actual", "y": "Predicted"},
                                              title="Actual vs Predicted",
                                              color_discrete_sequence=["#00ffcc"])
                        fig_pred.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                                           x1=y_test.max(), y1=y_test.max(),
                                           line=dict(color="#ffffff44", dash="dash"))
                        st.plotly_chart(fig_pred, use_container_width=True)

                    # Feature importance
                    if hasattr(model, "feature_importances_"):
                        fi = pd.DataFrame({
                            "Feature": features,
                            "Importance": model.feature_importances_
                        }).sort_values("Importance", ascending=True)
                        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                                        template="plotly_dark", title="Feature Importance",
                                        color="Importance", color_continuous_scale="Teal")
                        st.plotly_chart(fig_fi, use_container_width=True)

                    st.success("✅ Model trained! Scroll down to make predictions.")

                except Exception as e:
                    st.error(f"Training failed: {e}")

    # ── STEP 4: Predict ────────────────────────────────────────
    if st.session_state.model is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="step-badge">Step 04 — Predict</div>', unsafe_allow_html=True)
        st.markdown("Enter values for each feature to get a prediction:")

        input_vals = {}
        cols = st.columns(min(len(st.session_state.feature_cols), 3))
        for i, feat in enumerate(st.session_state.feature_cols):
            with cols[i % 3]:
                orig_col = st.session_state.df[feat]
                if feat in st.session_state.encoders:
                    options = st.session_state.df[feat].dropna().unique().tolist()
                    input_vals[feat] = st.selectbox(feat, options, key=f"pred_{feat}")
                elif pd.api.types.is_float_dtype(orig_col):
                    input_vals[feat] = st.number_input(feat, value=float(orig_col.median()), key=f"pred_{feat}", format="%.4f")
                else:
                    input_vals[feat] = st.number_input(feat, value=int(orig_col.median()), key=f"pred_{feat}", step=1)

        if st.button("🔮 Predict", use_container_width=True):
            try:
                row = pd.DataFrame([input_vals])
                for col, le in st.session_state.encoders.items():
                    if not col.startswith("__target_") and col in row.columns:
                        row[col] = le.transform(row[col].astype(str))

                row_scaled = st.session_state.scaler.transform(row[st.session_state.feature_cols])
                result = st.session_state.model.predict(row_scaled)[0]

                # Decode target if needed
                target_key = f"__target_{st.session_state.target_col}"
                if target_key in st.session_state.encoders:
                    result = st.session_state.encoders[target_key].inverse_transform([int(result)])[0]

                st.markdown(f"""
                <div class="predict-result">
                    <div class="predict-label">Predicted {st.session_state.target_col}</div>
                    <div class="predict-value">{result}</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar for classification
                if st.session_state.task_type == "Classification" and hasattr(st.session_state.model, "predict_proba"):
                    probs = st.session_state.model.predict_proba(row_scaled)[0]
                    classes = st.session_state.model.classes_
                    target_key = f"__target_{st.session_state.target_col}"
                    if target_key in st.session_state.encoders:
                        classes = st.session_state.encoders[target_key].inverse_transform(classes)
                    prob_df = pd.DataFrame({"Class": classes, "Probability": probs}).sort_values("Probability", ascending=True)
                    fig_prob = px.bar(prob_df, x="Probability", y="Class", orientation="h",
                                     template="plotly_dark", title="Class Probabilities",
                                     color="Probability", color_continuous_scale="Teal",
                                     range_x=[0, 1])
                    st.plotly_chart(fig_prob, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 0; color:#1e2a3a;">
        <div style="font-size:3rem">🔮</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; margin-top:12px; color:#2a3a4a;">
            Upload a CSV to begin
        </div>
    </div>
    """, unsafe_allow_html=True)
