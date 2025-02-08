import pickle
import streamlit as st
import pandas as pd
import os

# Load the model
model_path = os.path.join('diabetes_model.pkl')
model = pickle.load(open(model_path, 'rb'))

# Function to handle one-hot encoding of smoking history
def encode_smoking_history(smoking_status):
    smoking_current = 0
    smoking_former = 0
    smoking_never = 0

    if smoking_status == "Current":
        smoking_current = 1
    elif smoking_status == "Former":
        smoking_former = 1
    elif smoking_status == "Never":
        smoking_never = 1

    return smoking_current, smoking_former, smoking_never

# Streamlit app
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title('Diabetes Prediction Web App')
st.info("An easy and user-friendly application to predict diabetes based on your health data.")

# Sidebar for user input
st.sidebar.header('Diabetic Prediction Project')
st.sidebar.image(os.path.join('diabetes.PNG'))
st.sidebar.write('This project is using Diabetic Dataset from kaggle with 90% Accuracy till now')
st.sidebar.write("")
st.sidebar.markdown('Made with 🍬 By Eng. [Ramez Mohamed](https://www.linkedin.com/in/ramezmo1/)')








# Gender selection (with default value)
gender_options = ["Female", "Male"]
gender = st.selectbox("Gender", gender_options, index=0)

# Input fields for numerical features (with default values and floating-point numbers)
age = st.slider('Age', 18, 100, 45)  # Age range from 18 to 100 with a default value of 45
bmi = st.slider('BMI', 10.0, 60.0, 25.0, step=0.1)  # BMI range with floating-point precision (default 25.0)
HbA1c_level = st.slider('HbA1c Level', 4.0, 15.0, 6.0, step=0.1)  # HbA1c level range with floating-point precision (default 6.0)
blood_glucose_level = st.slider('Blood Glucose Level (mg/dL)', 50, 300, 100)  # Glucose range from 50 to 300 mg/dL with a default value of 100

# Input fields for categorical features (with default values)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'], index=0)
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'], index=0)
smoking_history = st.selectbox("Smoking Status", ["Current", "Former", "Never"], index=2)  # Default to "Never"

# Manual encoding for gender
gender_mapping = {"Female": 0, "Male": 1}
gender_encoded = gender_mapping[gender]

# One-hot encoding for smoking status
smoking_current, smoking_former, smoking_never = encode_smoking_history(smoking_history)

# Prepare the data as a DataFrame
df = pd.DataFrame({
    'gender': [gender_encoded],
    'age': [age],
    'hypertension': [1 if hypertension == "Yes" else 0],
    'heart_disease': [1 if heart_disease == "Yes" else 0],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level],
}, index=[0])

# Add one-hot encoded smoking history columns to the DataFrame
df['smoking_history_current'] = smoking_current
df['smoking_history_former'] = smoking_former
df['smoking_history_never'] = smoking_never

# Reorder the columns to match the model's expected order
df = df[['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 
         'blood_glucose_level', 'smoking_history_current', 'smoking_history_former', 
         'smoking_history_never']]

# Fill NaN values with the column mean (if any)
df = df.fillna(df.mean())

# Button to confirm and predict
confirm = st.button('Predict Diabetes')

# Show the original dataframe for debugging
#st.write("Original DataFrame (before prediction):")
#st.write(df)

if confirm:
    # Display a loading spinner while the model is predicting
    with st.spinner('Predicting...'):
        result = model.predict(df)
    
    # Display the result with a colored message
    if result[0] == 1:
        st.markdown("<h3 style='color: red;'>Prediction: Diabetic</h3>", unsafe_allow_html=True)
        st.write("It appears that you are at risk for diabetes. Consider consulting a healthcare provider for further analysis.")
    else:
        st.markdown("<h3 style='color: green;'>Prediction: Non-Diabetic</h3>", unsafe_allow_html=True)
        st.write("You are not at high risk for diabetes based on the input provided. Keep maintaining a healthy lifestyle!")

# Add custom styling for better presentation
st.markdown("""
    <style>
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 20px;
            border-radius: 10px;
            padding: 15px;
        }
        .stButton>button:hover {
            background-color: #47a0f5;
        }
        .stTitle {
            font-size: 28px;
            font-weight: bold;
        }
        .stInfo {
            font-size: 18px;
        }
        .stSelectbox, .stSlider {
            font-size: 16px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)
