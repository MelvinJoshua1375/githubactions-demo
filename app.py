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

# Prediction function
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

# Group features by their suffix
def group_features(columns):
    groups = {"mean": [], "se": [], "worst": []}
    for col in columns:
        if col.endswith("mean"):
            groups["mean"].append(col)
        elif col.endswith("se"):
            groups["se"].append(col)
        elif col.endswith("worst"):
            groups["worst"].append(col)
        else:
            groups.setdefault("other", []).append(col)
    return groups

groups = group_features(X.columns)

# Slider creation helper
def create_sliders(feature_list):
    sliders = []
    for feature_name in feature_list:
        col_min = float(X[feature_name].min())
        col_max = float(X[feature_name].max())
        step = (col_max - col_min) / 100
        sliders.append(gr.Slider(minimum=col_min, maximum=col_max, value=(col_min + col_max)/2, step=step, label=feature_name))
    return sliders

mean_sliders = create_sliders(groups.get("mean", []))
se_sliders = create_sliders(groups.get("se", []))
worst_sliders = create_sliders(groups.get("worst", []))

# All sliders concatenated for prediction input
all_sliders = mean_sliders + se_sliders + worst_sliders

# Example input from the dataset (first row)
example = list(X.iloc[0])

# Function to color the prediction
def color_label(label):
    if label == "malignant":
        return f"<span style='color: red; font-weight: bold;'>{label.upper()}</span>"
    else:
        return f"<span style='color: green; font-weight: bold;'>{label.upper()}</span>"

# Wrapper prediction to colorize label
def predict_colored(*features):
    label, proba_df = predict(*features)
    return color_label(label), proba_df

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🦠 Breast Cancer Classifier 🦠")
    gr.Markdown(
        "Predict if a tumor is **malignant** or **benign** based on 30 numeric features "
        "extracted from a digitized image of a breast mass fine needle aspirate (FNA)."
    )
    
    with gr.Row():
        with gr.Column(scale=2, min_width=600):
            with gr.Accordion("Mean Features", open=True):
                gr.Markdown("Features computed as mean values for the tumor cells.")
                for slider in mean_sliders:
                    slider.render()
            with gr.Accordion("Standard Error (SE) Features", open=False):
                gr.Markdown("Standard error of mean values for tumor cells.")
                for slider in se_sliders:
                    slider.render()
            with gr.Accordion("Worst Features", open=False):
                gr.Markdown("Worst or largest values recorded for tumor cells.")
                for slider in worst_sliders:
                    slider.render()
            btn = gr.Button("Classify 🩺")

        with gr.Column(scale=1, min_width=400):
            label = gr.HTML(label="Predicted Tumor Type")
            chart = gr.BarPlot(label="Class Probabilities", x="Class", y="Probability", width=400, height=300)

    btn.click(
        fn=predict_colored,
        inputs=all_sliders,
        outputs=[label, chart]
    )

    gr.Examples(
        examples=[example],
        inputs=all_sliders
    )

demo.launch()
