import gradio as gr
import pandas as pd
import joblib

model = joblib.load("model.pkl")

columns = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI','DiabetesPedigreeFunction','Age','BMI_Age'
]

def predict(*inputs):
    df = pd.DataFrame([inputs], columns=columns)
    prob = model.predict_proba(df)[0][1]
    label = "Diabetes" if prob >= 0.5 else "No Diabetes"
    return f"{label} (Probability: {prob:.2f})"

interface = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label=c) for c in columns],
    outputs="text",
    title="Diabetes Prediction System"
)

interface.launch()