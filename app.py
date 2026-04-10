import streamlit as st
import pickle
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #4facfe, #00f2fe);
        color: white;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stNumberInput input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    with open("Second_Model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align: center;'>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict Diabetes Risk using Machine Learning</p>", unsafe_allow_html=True)

st.divider()

# -------------------- INPUT SECTION --------------------
st.subheader("📋 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1, step=1)

st.divider()

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Diabetes"):
    try:
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])

        prediction = model.predict(input_data)

        with st.spinner("Analyzing patient data..."):
            st.balloons()

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align: center;'>⚡ Built with Streamlit | ML Project</p>", unsafe_allow_html=True)
