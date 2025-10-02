# Indonesian Fake News Detection

A deep learning project for fake news detection in Indonesian texts using transformer models and NLP techniques. Created as part of the Deep Learning Course assignment.

## Overview

- **Purpose**: Classify Indonesian news articles as valid or hoax
- **Models**: IndoBERT, DistilBERT Multilingual, mBERT
- **Features**: Text analysis, feature engineering, model ensemble
- **Dataset**: 600 labeled Indonesian news articles (balanced)

## Requirements

```txt
torch==2.6.0
transformers
datasets
scikit-learn==1.6.0
optuna
```

## Quick Start


Choose one of these options to run the notebook:

   - **VS Code (Recommended)**:
     - Install "Jupyter" extension in VS Code
     - Open `fake-news-classification.ipynb`
     - Click "Select Kernel" and choose Python environment
     - Run cells using play button or `Shift+Enter`

   - **Google Colab**:
     - Upload `fake-news-classification.ipynb` to Colab
     - Upload dataset to Colab workspace
     - Update dataset path in notebook
     - Run all cells

   - **Kaggle**:
     - Create new notebook
     - Import `fake-news-classification.ipynb`
     - Connect to GPU if needed
     - Update dataset path
     - Run all cells

## Project Structure

- `fake-news-classification.ipynb`: Main analysis notebook 
- `600 news with valid hoax label.csv`: Dataset

## Colab 
  - [View notebook in Google Drive](https://drive.google.com/file/d/1VVbqXJi8hWpIhcIXHqM2Lk4-4_aVVrnK/view?usp=sharing) (backup link if notebook doesn't render in GitHub)