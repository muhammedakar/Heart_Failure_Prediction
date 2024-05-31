import streamlit as st
from streamlit import components
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(
    page_title="HEART FAILURE PREDICTION",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

model = joblib.load("final_model.pkl")
st.title('Heart Failure Prediction')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', step=1, min_value=18)

with col2:
    rest = st.number_input('Rest', min_value=80)

with col3:
    cholesterol = st.number_input('cholesterol', min_value=80)

cols1, cols2, cols3 = st.columns(3)

gender = ['Female', 'Male']

with cols1:
    MaxHr = st.number_input('MaxHr', min_value=60)

with cols2:
    oldpeak = st.number_input('Old Peak', min_value=-3)

with cols3:
    sex = st.selectbox('Sex', gender)

cola1, cola2, cola3, cola4, cola5 = st.columns(5)

chest = ['ATA', 'NAP', 'ASY', 'TA']

resta = ['Normal', 'ST', 'LVH']
angina = ['N', 'Y']
sts = ['Up', 'Flat', 'Down']

with cola1:
    chest_pain = st.selectbox('Chest Pain Type', chest)

with cola2:
    restingEcg = st.selectbox('Resting ECG', resta)

with cola3:
    ExerciseAngina = st.selectbox('Exercise Angina', angina)

with cola4:
    ST_Slope = st.selectbox('ST Slope', sts)

with cola5:
    FastingBS = st.selectbox('Fasting BS', [0, 1])

if sex == 'Male':
    sex = [1]
elif sex == 'Female':
    sex = [0]
else:
    st.title('Hata!')

if chest_pain == 'ATA':
    chest_pain = [1, 0, 0]
elif chest_pain == 'NAP':
    chest_pain = [0, 1, 0]
elif chest_pain == 'TA':
    chest_pain = [0, 0, 1]
else:
    chest_pain = [0, 0, 0]

if restingEcg == 'Normal':
    restingEcg = [1, 0]
elif restingEcg == 'ST':
    restingEcg = [0, 1]
else:
    restingEcg = [0, 0]

if ExerciseAngina == 'Y':
    ExerciseAngina = [1]
else:
    ExerciseAngina = [0]

if ST_Slope == 'Flat':
    ST_Slope = [1, 0]
elif ST_Slope == 'Up':
    ST_Slope = [0, 1]
else:
    ST_Slope = [0, 0]

if FastingBS == '1':
    FastingBS = [1]
else:
    FastingBS = [0]

if age > 64:
    agecat = [0, 1]
elif age < 45:
    agecat = [0, 0]
else:
    agecat = [1, 0]

if MaxHr > 150:
    maxcat = [1, 0]
elif MaxHr < 60:
    maxcat = [0, 0]
else:
    maxcat = [1, 0]

if oldpeak > 1.5:
    oldpeakcat = [0, 1]
elif oldpeak < -2.65:
    oldpeakcat = [0, 0]
else:
    oldpeakcat = [1, 0]

if rest > 120:
    restcat = [0, 1]
elif rest < 80:
    restcat = [0, 0]
else:
    restcat = [1, 0]

chocat = []

if cholesterol > 280:
    chocat = [0, 0, 1]
elif cholesterol > 240:
    chocat = [0, 1, 0]
elif cholesterol > 200:
    chocat = [1, 0, 0]
else:
    chocat = [0, 0, 0]


def predict_review_score(Age, RestingBP, Cholesterol, MaxHR, Oldpeak, Sex_M, ChestPainType_ATA, ChestPainType_NAP,
                         ChestPainType_TA, RestingECG_Normal, RestingECG_ST, ExerciseAngina_Y, ST_Slope_Flat, ST_Slope_Up,
                         FastingBS_1, AgeCat_Middle, AgeCat_Senior, MaxHRCat_Moderate, MaxHRCat_High, RestingBPCat_Normal,
                         RestingBPCat_High, OldpeakCat_Normal, OldpeakCat_High, CholesterolCat_Normal, CholesterolCat_High,
                         CholesterolCat_Very_High):
    features = [Age, RestingBP, Cholesterol, MaxHR, Oldpeak, Sex_M, ChestPainType_ATA, ChestPainType_NAP,
                         ChestPainType_TA, RestingECG_Normal, RestingECG_ST, ExerciseAngina_Y, ST_Slope_Flat, ST_Slope_Up,
                         FastingBS_1, AgeCat_Middle, AgeCat_Senior, MaxHRCat_Moderate, MaxHRCat_High, RestingBPCat_Normal,
                         RestingBPCat_High, OldpeakCat_Normal, OldpeakCat_High, CholesterolCat_Normal, CholesterolCat_High,
                         CholesterolCat_Very_High]

    prediction = model.predict([features])

    return prediction[0]


if st.button('Predict'):
    predicted_score = predict_review_score(age, rest, cholesterol, MaxHr,
                                           oldpeak, *sex, *chest_pain, *restingEcg,
                                           *ExerciseAngina, *ST_Slope, *FastingBS,
                                           *agecat, *maxcat, *restcat, *oldpeakcat, *chocat)
    if predicted_score == 0:
        predicted_score = 'You are in safe :D'
    else:
        predicted_score = 'You should seen to a doctor'
    st.success(f"{predicted_score}")

