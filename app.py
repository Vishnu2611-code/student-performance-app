import streamlit as st
import numpy as np
import joblib
import pandas as pd
from fpdf import FPDF
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download
import sqlite3

# ---------------- DATABASE SETUP ----------------
conn = sqlite3.connect("predictions.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS history
             (study_hours INT, attendance INT, participation INT,
              prediction TEXT, confidence REAL)''')
conn.commit()

# ---------------- THEME TOGGLE ----------------
theme = st.sidebar.radio("Choose Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#0e1117"
    text_color = "white"
else:
    bg_color = "#f5f5f5"
    text_color = "black"

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}
h1, h2, h3, h4 {{
    color: {text_color};
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- ADMIN LOGIN ----------------
admin_password = "admin123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

password = st.sidebar.text_input("Admin Password", type="password")
if password == admin_password:
    st.session_state.logged_in = True
    st.sidebar.success("Admin Logged In")

# ---------------- LOAD MODEL ----------------
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
    return joblib.load(model_path), joblib.load(encoder_path)

model, le = load_models()

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Predictor")
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

# ---------------- BULK CSV PREDICTION ----------------
st.subheader("📂 Bulk Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    df["Predicted Performance"] = le.inverse_transform(preds)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Results CSV", csv, "bulk_predictions.csv")

# ---------------- SINGLE PREDICTION ----------------
if st.button("Predict Performance"):

    input_data = np.array([[study_hours, attendance, participation]])
    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)[0]

    probability = model.predict_proba(input_data)
    confidence = round(max(probability[0]) * 100, 2)

    st.success(f"🎯 Predicted Performance: **{result}**")
    st.info(f"🔍 Confidence Level: **{confidence}%**")

    # Save to DB
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?)",
              (study_hours, attendance, participation, result, confidence))
    conn.commit()

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
        st.warning("⚠ Student needs to increase study hours and participation.")
    elif result == "Medium":
        st.info("👍 Good performance, try improving consistency.")
    else:
        st.success("🌟 Excellent performance! Keep it up!")

    # ---------------- PDF REPORT ----------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    report_data = f"""
Student Performance Report

Study Hours: {study_hours}
Attendance: {attendance}
Participation: {participation}

Predicted Performance: {result}
Confidence Level: {confidence}%
"""

    for line in report_data.split("\n"):
        pdf.cell(200, 8, txt=line, ln=True)

    pdf.output("report.pdf")

    with open("report.pdf", "rb") as f:
        st.download_button("📄 Download Report", f, "Student_Report.pdf")

# ---------------- ADMIN DATABASE VIEW ----------------
if st.session_state.logged_in:
    st.subheader("📁 Database History (Admin Only)")
    db_df = pd.read_sql_query("SELECT * FROM history", conn)
    st.dataframe(db_df)
