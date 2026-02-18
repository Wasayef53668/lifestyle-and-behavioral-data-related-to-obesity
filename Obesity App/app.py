import streamlit as st
import pandas as pd
import joblib

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Obesity Risk Calculator",
    page_icon="🧠",
    layout="wide"
)

# ---------------- Load artifacts (cached) ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("obesity_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, feature_cols

model, feature_cols = load_artifacts()

# ---------------- Helpers ----------------
def build_row(gender, age, family, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans):
    row = [gender, age, family, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]
    return pd.DataFrame([row], columns=feature_cols)

def interpret_label(label: str) -> str:
    if "Obesity" in label:
        return "High risk"
    if "Overweight" in label:
        return "Moderate risk"
    if "Normal" in label:
        return "Healthy range"
    return "Monitor"

def pick_reasons(family, faf, tue, caec, fcvc, ch2o, favc, calc):
    reasons = []
    if family == 1:
        reasons.append("Family history increases baseline risk.")
    if faf <= 0.7:
        reasons.append("Low physical activity (FAF) is linked with higher obesity levels.")
    if tue >= 1.2:
        reasons.append("Higher screen time (TUE) suggests a more sedentary lifestyle.")
    if caec in ["Frequently", "Always"]:
        reasons.append("Frequent snacking between meals (CAEC) is associated with higher weight categories.")
    if fcvc <= 2.0:
        reasons.append("Lower vegetable intake (FCVC) reduces a protective dietary factor.")
    if ch2o <= 1.7:
        reasons.append("Lower water intake (CH2O) often correlates with less healthy routines.")
    if favc == 1:
        reasons.append("Frequent high-calorie food (FAVC) can increase risk when combined with low activity.")
    if calc in ["Frequently", "Always"]:
        reasons.append("Higher alcohol consumption (CALC) is associated with increased overweight risk in the data.")
    return reasons[:4]

# ---------------- Header ----------------
st.title("Lifestyle Obesity Risk Calculator")
st.caption("Behavior-based risk estimate + optional BMI check (for fun).")

with st.expander("⚠️ Important note"):
    st.write(
        "This is a behavioral decision-support demo, not medical advice. "
        "The dataset includes synthetic/augmented records (e.g., age appears as decimals). "
        "Use outputs for exploration and education."
    )

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["Risk Predictor", "BMI Fun Check"])

# ================= TAB 1: Risk Predictor =================
with tab1:
    st.header("Behavior-Based Risk Prediction")
    st.write("Estimate obesity category risk from lifestyle habits (Height/Weight NOT used).")

    # ---- Sidebar inputs ----
    st.sidebar.header("Inputs (Risk Predictor)")

    # Reset demo values
    if st.sidebar.button("Reset to demo values"):
        st.session_state["age"] = 25
        st.session_state["gender"] = "Female"
        st.session_state["family"] = 1
        st.session_state["favc"] = 1
        st.session_state["fcvc"] = 2.0
        st.session_state["ncp"] = 3.0
        st.session_state["caec"] = "Sometimes"
        st.session_state["smoke"] = 0
        st.session_state["ch2o"] = 2.0
        st.session_state["scc"] = 0
        st.session_state["faf"] = 1.0
        st.session_state["tue"] = 1.0
        st.session_state["calc"] = "Sometimes"
        st.session_state["mtrans"] = "Public_Transportation"

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="gender")
    age = st.sidebar.slider("Age", 14, 61, st.session_state.get("age", 25), key="age")

    family = st.sidebar.selectbox("Family history overweight (0=No, 1=Yes)", [0, 1], key="family")
    favc = st.sidebar.selectbox("High-calorie food frequent? (FAVC)", [0, 1], key="favc")

    fcvc = st.sidebar.slider("Vegetable consumption (FCVC: 1–3)", 1.0, 3.0, float(st.session_state.get("fcvc", 2.0)), 0.1, key="fcvc")
    ncp = st.sidebar.slider("Meals per day (NCP: 1–4)", 1.0, 4.0, float(st.session_state.get("ncp", 3.0)), 0.1, key="ncp")

    caec = st.sidebar.selectbox("Snacking between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"], key="caec")

    smoke = st.sidebar.selectbox("Smoking (0=No, 1=Yes)", [0, 1], key="smoke")
    ch2o = st.sidebar.slider("Water intake (CH2O: 1–3)", 1.0, 3.0, float(st.session_state.get("ch2o", 2.0)), 0.1, key="ch2o")

    scc = st.sidebar.selectbox("Monitor calories (SCC) (0=No, 1=Yes)", [0, 1], key="scc")
    faf = st.sidebar.slider("Physical activity (FAF: 0–3)", 0.0, 3.0, float(st.session_state.get("faf", 1.0)), 0.1, key="faf")

    tue = st.sidebar.slider("Screen time (TUE: 0–2)", 0.0, 2.0, float(st.session_state.get("tue", 1.0)), 0.1, key="tue")

    calc = st.sidebar.selectbox("Alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"], key="calc")
    mtrans = st.sidebar.selectbox(
        "Transportation (MTRANS)",
        ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"],
        key="mtrans"
    )

    # ---- Main layout ----
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Your inputs (model features)")
        data = build_row(gender, age, family, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans)
        st.dataframe(data, use_container_width=True)

    with right:
        st.subheader("Prediction")
        if st.button("Predict"):
            pred = model.predict(data)[0]
            proba = model.predict_proba(data)[0]

            risk_band = interpret_label(str(pred))

            # CatBoost sometimes returns ['label'] as a list; make it string-safe
            pred_str = pred[0] if isinstance(pred, (list, tuple)) else str(pred)

            if risk_band == "High risk":
                st.error(f"Predicted category: **{pred_str}**  |  Risk band: **{risk_band}**")
            elif risk_band == "Moderate risk":
                st.warning(f"Predicted category: **{pred_str}**  |  Risk band: **{risk_band}**")
            elif risk_band == "Healthy range":
                st.success(f"Predicted category: **{pred_str}**  |  Risk band: **{risk_band}**")
            else:
                st.info(f"Predicted category: **{pred_str}**  |  Risk band: **{risk_band}**")

            prob_df = pd.DataFrame({"Category": model.classes_, "Probability": proba})
            prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

            st.markdown("### Top 3 probabilities")
            st.table(prob_df.head(3))

            st.markdown("### Probability distribution")
            st.bar_chart(prob_df.set_index("Category"))

            st.markdown("### Why this result?")
            reasons = pick_reasons(family, faf, tue, caec, fcvc, ch2o, favc, calc)
            if reasons:
                for r in reasons:
                    st.write("• " + r)
            else:
                st.write("No strong rule-based risk signals detected from the selected inputs.")

    st.caption("Model uses behavior-only features (no Height/Weight) to avoid BMI leakage.")

# ================= TAB 2: BMI Fun Check =================
with tab2:
    st.header("Optional BMI Fun Check 😄")
    st.write("Enter height & weight to calculate BMI — just for fun, not medical advice.")

    c1, c2 = st.columns(2)
    with c1:
        height_cm = st.number_input("Height (cm)", min_value=120, max_value=230, value=170, step=1)
    with c2:
        weight_kg = st.number_input("Weight (kg)", min_value=35, max_value=200, value=70, step=1)

    if st.button("Check my BMI"):
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)

        st.subheader(f"Your BMI: {bmi:.1f}")

        # Fun comments (light + non-shaming)
        if bmi < 18.5:
            st.info("You're lighter than a feather 🪶 — maybe grab a sandwich and a smoothie!")
        elif bmi < 25:
            st.success("Perfectly shaped 😎✨ — your body called, it says 'keep it up!'")
        elif bmi < 30:
            st.warning("You're in the 'extra cuddle mode' zone 🧸 — a bit more movement could help!")
        else:
            st.error("You're in 'boss-level mass' mode 🦍 — consider healthier habits (and maybe fewer midnight snacks).")

        st.caption("BMI is a simple metric and doesn’t reflect muscle mass, body composition, or overall health.")
