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

def predict(sepal_length, sepal_width, petal_length, petal_width):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = clf.predict(data)[0]
    return iris.target_names[pred]

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(minimum=4, maximum=8, step=0.1, label="Sepal Length"),
        gr.Slider(minimum=2, maximum=5, step=0.1, label="Sepal Width"),
        gr.Slider(minimum=1, maximum=7, step=0.1, label="Petal Length"),
        gr.Slider(minimum=0.1, maximum=2.5, step=0.1, label="Petal Width"),
    ],
    outputs="text",
    title="🌸 Demonstration of Simple Classification"
)

iface.launch()
