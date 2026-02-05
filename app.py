import streamlit as st
import numpy as np
import joblib
import pandas as pd
from fpdf import FPDF
import streamlit.components.v1 as components

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

h1, h2, h3, h4 {
    color: #ffffff;
    text-align: center;
}

section[data-testid="stSidebar"] {
    background-color: #161b22;
}

.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1em;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #2ea043;
    color: white;
}

.stSlider label {
    color: #ffffff !important;
}

.stMarkdown, .stText, .stSubheader {
    color: #ffffff !important;
}

div[data-testid="stDataFrame"] {
    background-color: #161b22;
}
</style>
""", unsafe_allow_html=True)

import os
import zipfile

# Unzip model if not already extracted
if not os.path.exists("student_performance_model.pkl"):
    with zipfile.ZipFile("student_performance_model.zip", "r") as zip_ref:
        zip_ref.extractall()

if not os.path.exists("label_encoder.pkl"):
    with zipfile.ZipFile("label_encoder.zip", "r") as zip_ref:
        zip_ref.extractall()


# ---------------- LOAD MODEL ----------------
# Load model and encoder
@st.cache_resource
def load_models():
    model = joblib.load("student_performance_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_models()



# ---------------- TITLE ----------------
st.title("🎓 Student Performance Predictor")
st.markdown("### AI-Based Student Performance Prediction System")
st.write("Predicts student performance using study habits, attendance, and participation.")

# ---------------- INPUTS ----------------
study_hours = st.slider("Weekly Self Study Hours", 0, 40, 10)
attendance = st.slider("Attendance Percentage", 0, 100, 75)
participation = st.slider("Class Participation Score", 0, 10, 5)

# ---------------- CHART ----------------
st.subheader("📊 Student Activity Overview")
chart_data = pd.DataFrame({
    "Activity": ["Study Hours", "Attendance", "Participation"],
    "Value": [study_hours, attendance, participation]
})
st.bar_chart(chart_data.set_index("Activity"))
st.markdown(
    """
    <style>
    canvas {
        background-color: #0e1117 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PREDICTION ----------------
if st.button("Predict Performance"):

    input_data = np.array([[study_hours, attendance, participation]])

    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)[0]

    st.success(f"🎯 Predicted Performance: **{result}**")

    # Confidence
    probability = model.predict_proba(input_data)
    confidence = round(max(probability[0]) * 100, 2)
    st.info(f"🔍 Confidence Level: **{confidence}%**")

    # Performance Meter
    performance_score = {"Low": 30, "Medium": 65, "High": 90}
    score = performance_score[result]

    components.html(f"""
    <div style="width:100%;background-color:#333;border-radius:10px;">
      <div style="width:{score}%;background-color:#4CAF50;color:white;padding:5px;border-radius:10px;text-align:center;">
        Performance Meter: {score}%
      </div>
    </div>
    """, height=50)

    # Tips
    if result == "Low":
        st.warning("⚠ Student needs to increase study hours and class participation.")
    elif result == "Medium":
        st.info("👍 Good performance, but improving study consistency can help reach high level.")
    else:
        st.success("🌟 Excellent performance! Keep up the great work!")

    # PDF Download
    if st.button("📄 Download Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Student Performance Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Study Hours: {study_hours}", ln=True)
        pdf.cell(200, 10, txt=f"Attendance: {attendance}", ln=True)
        pdf.cell(200, 10, txt=f"Participation: {participation}", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Performance: {result}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {confidence}%", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button("⬇ Download PDF", f, "Student_Report.pdf")

# ---------------- HISTORY TABLE ----------------
if st.session_state.history:
    st.subheader("📁 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))
