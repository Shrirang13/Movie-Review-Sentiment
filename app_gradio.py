import os
import numpy as np
import gradio as gr

import joblib

from train_and_evaluate import clean_text


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Run train_and_evaluate.py / colab notebook first to generate it."
        )
    return joblib.load(model_path)


def predict_sentiment_gradio(review_text: str, model_path: str, show_cleaned: bool):
    model = load_model(model_path)

    if review_text is None or not str(review_text).strip():
        return "Please enter a movie review.", ""

    cleaned = clean_text(str(review_text))
    pred = int(model.predict([cleaned])[0])  # 0/1

    # Confidence estimate (best-effort)
    confidence_str = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        confidence = float(proba[1])  # probability of Positive
        confidence_str = f"\nConfidence: {confidence*100:.1f}%"
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function([cleaned])[0])
        confidence = 1.0 / (1.0 + np.exp(-score))
        confidence_str = f"\nConfidence: {confidence*100:.1f}%"

    sentiment = "Positive" if pred == 1 else "Negative"
    cleaned_out = cleaned if show_cleaned else ""

    return f"Predicted Sentiment: {sentiment}{confidence_str}", cleaned_out


def launch_gui(
    model_path: str = "models/best_sentiment_pipeline.joblib",
    share: bool = True,
):
    """
    Launches Gradio UI.
    In Colab, share=True provides a public link ("Go Live") for screenshots.
    """
    with gr.Blocks(title="Movie Review Sentiment Analysis") as demo:
        gr.Markdown("# 🎬 Movie Review Sentiment Analysis")
        gr.Markdown("Enter a review, click **Predict**, and get **Positive / Negative** sentiment.")

        with gr.Row():
            with gr.Column(scale=3):
                review_in = gr.Textbox(
                    label="Movie Review",
                    placeholder="Type a movie review here...",
                    lines=8,
                )
                show_cleaned = gr.Checkbox(
                    label="Show cleaned text (preprocessing output)",
                    value=False,
                )
                model_path_in = gr.Textbox(
                    label="Model path (use default)",
                    value=model_path,
                    interactive=False,
                )

                btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=2):
                out = gr.Textbox(label="Prediction", interactive=False, lines=4)
                cleaned_out = gr.Textbox(
                    label="Cleaned Text (if enabled)",
                    interactive=False,
                    lines=6,
                )

        btn.click(
            fn=predict_sentiment_gradio,
            inputs=[review_in, model_path_in, show_cleaned],
            outputs=[out, cleaned_out],
        )

    demo.launch(share=share)


if __name__ == "__main__":
    launch_gui()

