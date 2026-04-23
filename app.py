

import os
import pickle
import numpy as np
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chronotype Predictor",
    page_icon="🕰️",
    layout="centered"
)

# ── Custom style ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 850px;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-text {
    color: #666;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}
.result-box {
    padding: 1.2rem;
    border-radius: 16px;
    background-color: #f7f7f9;
    border: 1px solid #e6e6e6;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.small-note {
    font-size: 0.92rem;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model_data = load_model()
    theta = model_data["theta"]
    x_mean = model_data["x_mean"]
    x_std = model_data["x_std"]
except FileNotFoundError:
    st.error("⚠️ model.pkl was not found. Put it in the same folder as app.py.")
    st.stop()
except KeyError:
    st.error("⚠️ model.pkl does not contain the expected keys: theta, x_mean, x_std.")
    st.stop()

# ── Helpers ────────────────────────────────────────────────────────────────────
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_chronotype(wake_up, bed_time, productivity_time, caffeine):
    raw = np.array([[wake_up, bed_time, productivity_time, caffeine]], dtype=float)
    scaled = (raw - x_mean) / x_std
    x_new = np.concatenate((np.ones((1, 1)), scaled), axis=1)
    prob_night = float(sigmoid(np.matmul(x_new, theta)).item())
    pred = int(prob_night >= 0.5)
    return pred, prob_night

def confidence_label(prob):
    strength = abs(prob - 0.5) * 2
    if strength < 0.2:
        return "Low confidence"
    elif strength < 0.5:
        return "Moderate confidence"
    else:
        return "High confidence"

# ── Mappings ───────────────────────────────────────────────────────────────────
wake_map = {
    "Before 6:00 AM": 0,
    "6:00 AM - 8:00 AM": 1,
    "8:00 AM - 10:00 AM": 2,
    "After 10:00 AM": 3
}

bed_map = {
    "Before 10:00 PM": 0,
    "10:00 PM - 12:00 AM": 1,
    "After 12:00 AM": 2
}

prod_map = {
    "Morning (6 AM - 12 PM)": 0,
    "Afternoon (12 PM - 6 PM)": 1,
    "Evening (6 PM - 9 PM)": 2,
    "Night (9 PM - 12 AM)": 3
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts whether a user is more likely to be a "
        "**morning person** or a **night person** based on daily habits."
    )
    st.write("**Features used:**")
    st.write("- Wake-up time")
    st.write("- Bed time")
    st.write("- Most productive time")
    st.write("- Coffee consumption")
    st.info("Prediction is based on your trained logistic regression model.")

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🕰️ Chronotype Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Answer a few questions about your daily habits to estimate whether you are more likely a morning person or a night person.</div>',
    unsafe_allow_html=True
)

with st.form("chronotype_form"):
    st.subheader("Daily Habit Questions")

    wake_up_label = st.selectbox(
        "1. What time do you usually wake up?",
        list(wake_map.keys())
    )

    bed_time_label = st.selectbox(
        "2. What time do you usually go to bed?",
        list(bed_map.keys())
    )

    productivity_time_label = st.selectbox(
        "3. When do you feel most productive during the day?",
        list(prod_map.keys())
    )

    caffeine_label = st.slider(
        "4. How many cups of coffee do you drink per day?",
        min_value=0,
        max_value=10,
        value=1
    )

    submitted = st.form_submit_button("🔍 Predict Chronotype", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    wake_up_value = wake_map[wake_up_label]
    bed_time_value = bed_map[bed_time_label]
    productivity_value = prod_map[productivity_time_label]

    pred, prob_night = predict_chronotype(
        wake_up_value,
        bed_time_value,
        productivity_value,
        caffeine_label
    )

    prob_morning = 1 - prob_night
    confidence = confidence_label(prob_night)

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("Morning Probability", f"{prob_morning * 100:.1f}%")
    col2.metric("Night Probability", f"{prob_night * 100:.1f}%")

    if pred == 0:
        st.success("🌞 **Prediction: Morning Person**")
        st.markdown("""
<div class="result-box">
You appear to follow habits that are more aligned with a <b>morning chronotype</b>.
People in this group often wake earlier and may feel more productive earlier in the day.
</div>
""", unsafe_allow_html=True)
        st.progress(prob_morning)
    else:
        st.warning("🌙 **Prediction: Night Person**")
        st.markdown("""
<div class="result-box">
Your habits appear to be more aligned with a <b>night chronotype</b>.
People in this group often sleep later and may feel more productive in the evening or at night.
</div>
""", unsafe_allow_html=True)
        st.progress(prob_night)

    st.markdown(f"**Confidence level:** {confidence}")

    st.subheader("Your Input Summary")
    st.write(f"**Wake-up time:** {wake_up_label}")
    st.write(f"**Bed time:** {bed_time_label}")
    st.write(f"**Most productive time:** {productivity_time_label}")
    st.write(f"**Coffee per day:** {caffeine_label} cup(s)")

    st.markdown(
        '<div class="small-note">This prediction is only based on the features included in your dataset and model.</div>',
        unsafe_allow_html=True
    )