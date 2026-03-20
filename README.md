# Movie Review Sentiment Analysis (NLP)

This project classifies **movie reviews** as **Positive** or **Negative** using **TF-IDF** features and multiple **machine learning** models. It includes:
- Dataset loading + text preprocessing
- Feature extraction (TF-IDF)
- Model training + model comparison
- Best model selection
- Confusion matrix + accuracy results plots
- A **Gradio GUI** for live sentiment prediction

## Files
- `train_and_evaluate.py` - training, preprocessing demo, evaluation, plots, and saving the best model
- `app_gradio.py` - Gradio GUI that loads the saved best model
- `colab_notebook_fixed.ipynb` - ready-made notebook for Google Colab (screenshots-ready)
- `colab_notebook.ipynb` is included but `colab_notebook_fixed.ipynb` is the recommended one.
- `requirements.txt` - Python dependencies

## How to use in Google Colab (recommended)
1. Upload this whole folder to Google Colab.
2. Open `colab_notebook.ipynb` and run all cells.
3. Take the required screenshots from the notebook output:
   - Text preprocessing output
   - Model comparison table
   - Accuracy results
   - Confusion matrix
   - Best model selection text
   - GUI interface
   - GUI prediction output

## Trained model file
During training, the notebook/script saves:
- `models/best_sentiment_pipeline.joblib`

After training finishes, you can commit the generated model file to GitHub (required by the assignment).

## GUI ("Go Live")
The notebook launches Gradio with `share=True`, which gives you a public link you can open and screenshot.

