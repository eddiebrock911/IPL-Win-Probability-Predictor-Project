import streamlit as st
import pickle
import numpy as np
import pandas as pd

# streamlit run app.py

# Load trained pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🏏 IPL Win Probability Predictor",
    page_icon="🏆",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #141E30, #243B55);
            color: #FFFFFF;
        }
        .stApp {
            background: linear-gradient(120deg, #1e3c72, #2a5298);
        }
        h1 {
            text-align: center;
            color: #FFD700;
            font-family: 'Trebuchet MS', sans-serif;
            text-shadow: 2px 2px #000;
        }
        h2, h3, h4 {
            color: #00FFCC;
        }
        .stButton>button {
            background-color: #FFD700;
            color: #000;
            font-weight: bold;
            border-radius: 12px;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FFA500;
            color: white;
            transform: scale(1.05);
        }
        .css-1d391kg p {
            font-size: 16px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("🏏 IPL Win Probability Predictor")
st.markdown("<h4 style='text-align:center; color:#EEE;'>Batting Team की जीत की संभावना देखें!</h4>", unsafe_allow_html=True)

# ---------- INPUT SECTION ----------
st.markdown("### 🌆 Select Teams & City")
teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

cities = [
    'Mumbai', 'Kolkata', 'Delhi', 'Bangalore', 'Chennai',
    'Hyderabad', 'Jaipur', 'Ahmedabad', 'Pune'
]

col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams)
with col2:
    bowling_team = st.selectbox("🎯 Bowling Team", [t for t in teams if t != batting_team])
with col3:
    city = st.selectbox("🌍 Match City", cities)

# ---------- MATCH STATS ----------
st.markdown("### 📊 Match Stats")
c1, c2, c3 = st.columns(3)
with c1:
    total_runs_x = st.number_input("🎯 Target Runs", min_value=0, max_value=300, value=150)
    runs_left = st.number_input("🏃‍♂️ Runs Left", min_value=0, max_value=300, value=50)
with c2:
    balls_left = st.number_input("⚾ Balls Left", min_value=0, max_value=120, value=60)
    wickets = st.number_input("🧤 Wickets Left", min_value=0, max_value=10, value=5)
with c3:
    crr = st.number_input("📈 Current Run Rate (CRR)", min_value=0.0, max_value=20.0, value=7.5)
    rrr = st.number_input("📉 Required Run Rate (RRR)", min_value=0.0, max_value=20.0, value=8.0)

# ---------- PREPARE INPUT ----------
feature_names = [
    'batting_team', 'bowling_team', 'city',
    'total_runs_x', 'runs_left', 'balls_left',
    'wickets', 'crr', 'rrr'
]

input_data = pd.DataFrame([[
    batting_team, bowling_team, city,
    total_runs_x, runs_left, balls_left,
    wickets, crr, rrr
]], columns=feature_names)

for col in ['total_runs_x', 'runs_left', 'balls_left', 'wickets', 'crr', 'rrr']:
    input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

# ---------- PREDICT BUTTON ----------
st.markdown("---")
if st.button("🔎 Predict Win Probability"):
    prob = model.predict_proba(input_data)[0][1] * 100
    pred = model.predict(input_data)[0]

    st.balloons()
    st.markdown(f"<h2 style='text-align:center; color:#FFD700;'>🏆 Win Probability: {prob:.2f}%</h2>", unsafe_allow_html=True)

    if pred == 1:
        st.success(f"🎉 **{batting_team} is Likely to WIN!**")
    else:
        st.error(f"⚠️ **{batting_team} might LOSE.**")

# ---------- FOOTER ----------
st.markdown("""
    <hr>
    <p style='text-align:center; color:#AAAAAA; font-size:12px;'>
    Made with ❤️ by Ankit Kumar | IPL Win Predictor
    </p>
""", unsafe_allow_html=True)

