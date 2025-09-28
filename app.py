import gradio as gr
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Prediction function (takes 30 features as inputs)
def predict(*features):
    data_input = [list(features)]
    pred_class = clf.predict(data_input)[0]
    pred_proba = clf.predict_proba(data_input)[0]
    result = data.target_names[pred_class]

    proba_df = pd.DataFrame({
        "Class": data.target_names,
        "Probability": pred_proba
    })

    return result, proba_df

# Create sliders dynamically for all 30 features
sliders = []
for feature_name in X.columns:
    col_min = float(X[feature_name].min())
    col_max = float(X[feature_name].max())
    step = (col_max - col_min) / 100  # reasonable step size
    sliders.append(gr.Slider(minimum=col_min, maximum=col_max, value=(col_min + col_max)/2, step=step, label=feature_name))

# Example input - use the first data point as example
example = list(X.iloc[0])

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🦠 Breast Cancer Classifier 🦠")
    gr.Markdown(
        "This model predicts whether a tumor is **malignant** or **benign** based on 30 features "
        "computed from a digitized image of a fine needle aspirate (FNA) of a breast mass."
    )
    
    with gr.Row():
        with gr.Column(scale=2, min_width=600):
            for slider in sliders:
                slider.render()
            btn = gr.Button("Classify 🩺")
        with gr.Column(scale=1, min_width=400):
            label = gr.Textbox(label="Predicted Tumor Type", interactive=False)
            chart = gr.BarPlot(label="Class Probabilities", x="Class", y="Probability", width=400, height=300)

    btn.click(
        fn=predict,
        inputs=sliders,
        outputs=[label, chart]
    )

    gr.Examples(
        examples=[example],
        inputs=sliders
    )

demo.launch()
