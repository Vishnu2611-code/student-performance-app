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

# ---------------- LOAD MODEL ----------------
from huggingface_hub import hf_hub_download
import os

@st.cache_resource
def load_models():
    model_path = hf_hub_download(
        repo_id="Vishnu2611/student_performance_model",
        filename="student_performance_model.pkl"
    )

    encoder_path = hf_hub_download(
        repo_id="Vishnu2611/student_performance_model",
        filename="label_encoder.pkl"
    )

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return model, le

# 🚨 THIS LINE IS IMPORTANT
model, le = load_models()



if model is None or le is None:
    st.error("Model failed to load. Check HuggingFace files.")
    st.stop()


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

    st.session_state.history.append({
    "Study Hours": study_hours,
    "Attendance": attendance,
    "Participation": participation,
    "Prediction": result,
    "Confidence %": confidence
})

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
   report_data = f"""
   Student Performance Report

   Study Hours: {study_hours}
   Attendance: {attendance}
   Participation: {participation}
   Predicted Performance: {result}
   Confidence: {confidence}%
   """

   pdf = FPDF()
   pdf.add_page()
   pdf.set_font("Arial", size=12)
 
   for line in report_data.split("\n"):
       pdf.cell(200, 8, txt=line, ln=True)

   pdf.output("report.pdf")

   with open("report.pdf", "rb") as f:
       st.download_button("📄 Download Report", f, "Student_Report.pdf")


# ---------------- HISTORY TABLE ----------------
if st.session_state.history:
    st.subheader("📁 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))







