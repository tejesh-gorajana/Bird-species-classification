# Bird Species Classification Using Deep Learning

## Description

This project classifies bird species from images using a **pretrained EfficientNet-B2 model**
(`dennisjooo/Birds-Classifier-EfficientNetB2` from the Hugging Face Hub), which was created with transfer
learning. A Jupyter notebook demonstrates the model, and a simple **Streamlit** web app lets you upload a
bird photo and see the predicted species with its confidence and the top 5 predictions.

The model is used directly. It is not retrained in this project.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- EfficientNet-B2 (pretrained)
- Streamlit
- Pillow
- Matplotlib

## Project Structure

```text
tej_project/
│
├── bird_species_classification.ipynb
├── app.py
├── requirements.txt
└── README.md
```

There are **no subfolders**. No model file is needed in the folder: the model is downloaded automatically
on the first run and stored in your computer's Hugging Face cache (outside the project).

## How to Install

```bash
pip install -r requirements.txt
```

To run the notebook you also need a notebook environment (for example the Jupyter extension in VS Code,
or `pip install jupyter`).

## How to Run the Notebook

1. Open `bird_species_classification.ipynb`.
2. Copy a bird photo into the same folder and set `IMAGE_PATH` in the configuration cell.
3. Run all cells from top to bottom. The notebook loads the model, shows its details, predicts the
   species of your image and plots the top 5 predictions.

## How to Run Streamlit

```bash
streamlit run app.py
```

Then open the local address shown in the terminal (usually http://localhost:8501).

## How Prediction Works

```text
Upload Image
      ↓
Image Preprocessing (resize + normalize, done by the pipeline)
      ↓
EfficientNet-B2
      ↓
Predicted Bird Species
      ↓
Confidence
```

## Important

- The **first run needs an internet connection** to download the model. After that it works offline.
- No `.keras` or other model file is needed in the project folder.
- The model can only name the bird species it was trained on, so photos of other birds or non-birds will
  still get the closest matching species.
