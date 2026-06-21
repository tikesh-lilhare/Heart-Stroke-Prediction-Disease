import streamlit as st
import joblib
import pandas as pd

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Stroke Disease Prediction")
st.markdown("Provide the Following Details")

age = st.slider("Age" , 18,100,40)
sex = st.selectbox("Sex",['M','F'])
chest_pain = st.selectbox("Chest Pain Type",["ATA" , "NAP" , "TA" , "ASY"])
resting_bp = st.number_input("Resting Blood Pressure(mm Hg)",80,200,120)
cholesterol = st.number_input("Cholesterol (mg/dL)",100,600,200)
fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL",[0,1])
restinecg = st.selectbox("Resting ECG" , ["Normal","ST","LVH"])
maxHR = st.slider("Max Heart Rate",60,220,150)
exercise_angina = st.selectbox("Exercise-Induced Angina",["Y","N"])
oldpeak=st.slider("OldPeak (ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope",["UP" , "Flat" , "Down"])

if st.button("Predict"):
    raw_inputs = {
    'Age' : age ,
    'RestingBP' : resting_bp,
    'Cholesterol' : cholesterol,
    'FatingBS' : fastingBS,
    'MaxHR' : maxHR,
    'Oldpeak' : oldpeak,
    'Sex_' + sex : 1,
    'ChestPainType_' + chest_pain : 1,
    'RestingECG_' + restinecg : 1,
    'ExerciseAngina_' + exercise_angina : 1,
    'ST_Slope_' + st_slope : 1

    }

    input_df = pd.DataFrame([raw_inputs])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if(prediction == 1):
        st.error("High Risk Of Heart Disease")
    else:
        st.success("Low Risk Of Heart Disease")
