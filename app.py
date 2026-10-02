
import gradio as gr
import pandas as pd
import joblib

model = joblib.load("insurance_model.pkl")

# These are the columns used when training the model
feature_columns = [
    'age', 'bmi', 'children',
    'sex_male', 'smoker_yes',
    'region_northwest', 'region_southeast', 'region_southwest'
]

def predict_insurance(age, bmi, smoker, children, gender, region):

    input_data = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': gender,
        'smoker': smoker,
        'region': region
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=['sex', 'smoker', 'region'],
        drop_first=True
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    return f"Estimated Insurance Cost: ${prediction:,.2f}"


app = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="BMI"),
        gr.Dropdown(["yes", "no"], label="Smoking Status"),
        gr.Number(label="Number of Children"),
        gr.Dropdown(["male", "female"], label="Gender"),
        gr.Dropdown(
            ["northeast", "northwest", "southeast", "southwest"],
            label="Region"
        )
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Medical Insurance Cost Predictor",
    description="Enter your details to estimate medical insurance charges."
)

app.launch()
