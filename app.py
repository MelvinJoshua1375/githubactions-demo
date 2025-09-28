import gradio as gr
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred_class = clf.predict(data)[0]
    pred_proba = clf.predict_proba(data)[0]
    result = iris.target_names[pred_class]
    
    # Probability bar chart
    proba_df = pd.DataFrame({
        "Class": iris.target_names,
        "Probability": pred_proba
    })

    return result, proba_df

# Define the interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌸 Iris Flower Classifier")
    gr.Markdown(
        "This model classifies iris flowers into one of three species: **setosa**, **versicolor**, or **virginica**, "
        "based on measurements of their sepals and petals."
    )
    
    with gr.Row():
        with gr.Column():
            sepal_length = gr.Slider(4.0, 8.0, step=0.1, label="Sepal Length (cm)")
            sepal_width = gr.Slider(2.0, 5.0, step=0.1, label="Sepal Width (cm)")
            petal_length = gr.Slider(1.0, 7.0, step=0.1, label="Petal Length (cm)")
            petal_width = gr.Slider(0.1, 2.5, step=0.1, label="Petal Width (cm)")
            btn = gr.Button("Classify 🌺")
        with gr.Column():
            label = gr.Textbox(label="Predicted Species", interactive=False)
            chart = gr.BarPlot(label="Class Probabilities", x="Class", y="Probability", width=400, height=300)

    btn.click(
        fn=predict,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=[label, chart]
    )

    gr.Examples(
        examples=[
            [5.1, 3.5, 1.4, 0.2],
            [6.0, 2.2, 4.0, 1.0],
            [6.9, 3.1, 5.4, 2.1]
        ],
        inputs=[sepal_length, sepal_width, petal_length, petal_width]
    )

demo.launch()
