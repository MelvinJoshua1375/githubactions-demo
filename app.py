import gradio as gr
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Train simple model
iris = load_iris()
clf = RandomForestClassifier().fit(iris.data, iris.target)

def predict(sepal_length, sepal_width, petal_length, petal_width):
    pred = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return iris.target_names[pred]

# UI
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(4.3, 7.9, 5.1, label="Sepal Length"),
        gr.Slider(2.0, 4.4, 3.5, label="Sepal Width"),
        gr.Slider(1.0, 6.9, 1.4, label="Petal Length"),
        gr.Slider(0.1, 2.5, 0.2, label="Petal Width"),
    ],
    outputs="text",
    title="Demonstration of Simple Classification"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
