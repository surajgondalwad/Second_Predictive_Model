import streamlit as st
import pickle
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stTextInput>div>div>input {
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
st.markdown("<h1 style='text-align: center;'>🤖 ML Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Simple, Fast & Clean Prediction Tool</p>", unsafe_allow_html=True)

st.divider()

# -------------------- INPUT SECTION --------------------
st.subheader("🔢 Enter Input Features")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)

# Add/remove features based on your model

# -------------------- PREDICTION --------------------
st.divider()

if st.button("🚀 Predict"):
    try:
        input_data = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(input_data)

        st.success("✅ Prediction Complete!")

        # Animation-like effect
        with st.spinner("Analyzing data..."):
            st.balloons()

        st.markdown(f"""
        <h2 style='text-align: center; color: #00ffcc;'>
        🎯 Result: {prediction[0]}
        </h2>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
