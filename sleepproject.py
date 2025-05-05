import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Define constants
MODEL_FILE = 'sleep_model_nodisorder.pkl'
FEATURES = [
    'Age', 'Gender', 'BMI Category',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure',
    'Heart Rate', 'Daily Steps', 'Physical Activity Level',
    'Sleep Duration', 'Stress Level'
]

def load_or_train_model():
    # Check if the model file exists
    if os.path.exists(MODEL_FILE):
        model_data = joblib.load(MODEL_FILE)
        return model_data['model'], model_data['encoders']

    # If the model doesn't exist, train a new model
    data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    data.dropna(subset=['Quality of Sleep', 'Gender', 'BMI Category', 'Blood Pressure'], inplace=True)

    # Preprocess blood pressure
    bp_split = data['Blood Pressure'].str.split('/', expand=True)
    data['Systolic Blood Pressure'] = pd.to_numeric(bp_split[0], errors='coerce')
    data['Diastolic Blood Pressure'] = pd.to_numeric(bp_split[1], errors='coerce')

    # Drop rows with missing critical values
    data.dropna(subset=[
        'Systolic Blood Pressure', 'Diastolic Blood Pressure',
        'Heart Rate', 'Daily Steps', 'Physical Activity Level',
        'Sleep Duration', 'Stress Level', 'Age'
    ], inplace=True)

    # Encode categorical variables
    encoders = {}
    for col in ['Gender', 'BMI Category']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # Features and target variable
    X = data[FEATURES]
    y = data['Quality of Sleep']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump({'model': model, 'encoders': encoders}, MODEL_FILE)
    return model, encoders

def preprocess_input(user_input, encoders):
    df = pd.DataFrame([user_input])
    try:
        bp_parts = df['Blood Pressure'].str.split("/", expand=True)
        df['Systolic Blood Pressure'] = pd.to_numeric(bp_parts[0], errors='coerce')
        df['Diastolic Blood Pressure'] = pd.to_numeric(bp_parts[1], errors='coerce')
    except:
        raise ValueError("Blood Pressure must be in format like 120/80")

    # Encode categorical features
    for col in ['Gender', 'BMI Category']:
        df[col] = encoders[col].transform([df[col][0]])

    return df[FEATURES]

def display_prediction(pred):
    # Map prediction to readable quality
    quality_map = {
        1: "Very Poor", 2: "Poor", 3: "Fair",
        4: "Good", 5: "Very Good", 6: "Excellent",
        7: "Outstanding", 8: "Exceptional", 9: "Perfect", 10: "Ideal"
    }
    label = quality_map.get(pred, str(pred))
    st.success(f"💤 Predicted Sleep Quality: {label}")

    # Display quality advice
    if pred >= 7:
        st.info("✅ Excellent sleep quality. Keep it up!")
    elif pred >= 5:
        st.warning("🙂 Moderate sleep quality. You can improve it with lifestyle changes.")
    else:
        st.error("⚠️ Poor sleep quality. Consider consulting a sleep expert.")

def main():
    # Set up Streamlit page
    st.set_page_config("Sleep Quality Predictor", layout="centered")
    st.title("😴 Sleep Quality Prediction App")

    # Load or train the model
    model, encoders = load_or_train_model()

    with st.form("sleep_form"):
        st.subheader("👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 100, 30)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col3:
            bmi = st.selectbox("BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"])

        st.subheader("❤️ Health Metrics")
        bp = st.text_input("Blood Pressure (e.g., 120/80)", "120/80")
        hr = st.number_input("Heart Rate (bpm)", 40, 120, 72)
        steps = st.number_input("Daily Steps", 0, 30000, 5000)
        activity = st.slider("Physical Activity Level", 1, 10, 5)

        st.subheader("🛏 Sleep & Stress")
        duration = st.number_input("Sleep Duration (hrs)", 3.0, 12.0, 7.0, 0.1)
        stress = st.slider("Stress Level", 1, 10, 5)

        submitted = st.form_submit_button("🔍 Predict Sleep Quality")

        if submitted:
            try:
                user_input = {
                    'Age': age,
                    'Gender': gender,
                    'BMI Category': bmi,
                    'Blood Pressure': bp,
                    'Heart Rate': hr,
                    'Daily Steps': steps,
                    'Physical Activity Level': activity,
                    'Sleep Duration': duration,
                    'Stress Level': stress
                }

                # Preprocess user input
                X_input = preprocess_input(user_input, encoders)

                # Predict sleep quality
                prediction = model.predict(X_input)[0]

                # Display the prediction result
                display_prediction(prediction)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
