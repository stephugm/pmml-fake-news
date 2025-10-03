import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.set_page_config(
    page_title="Indonesian Fake News Classification",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_models():
    model_paths = {
        "IndoBERT": "./models/indobert/",
        "mBERT": "./models/mbert/",
        "DistilBERT": "./models/distilbert/"
    }
    
    models = {}
    for name, path in model_paths.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            models[name] = {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            st.error(f"Gagal memuat model '{name}'. Pastikan path '{path}' benar. Error: {e}")
            return None
    return models

st.title("📰 Indonesian Fake News Classification App")
st.write(
    "**Model Averaging** dari 3 model (IndoBERT, mBERT, DistilBERT) digunakan untuk memberikan hasil klasifikasi yang lebih akurat."
)

with st.spinner("Memuat semua model ensemble..."):
    models = load_models()

if models is None:
    st.stop()

news_text = st.text_area("Masukkan teks berita di sini:", height=200)
classify_button = st.button("Klasifikasi")

if classify_button and news_text.strip():
    with st.spinner('Menganalisis berita dengan 3 model...'):
        
        all_probabilities = []
        individual_predictions = {}

        for name, components in models.items():
            tokenizer = components["tokenizer"]
            model = components["model"]

            # 1. Tokenisasi
            inputs = tokenizer(news_text, return_tensors="pt", truncation=True, max_length=512, padding=True)

            # 2. Prediksi
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # 3. Hitung probabilitas dengan Softmax
            probabilities = torch.softmax(logits, dim=1)
            all_probabilities.append(probabilities)

            pred_id = torch.argmax(probabilities).item()
            label_name = model.config.id2label[pred_id]
            confidence = probabilities[0, pred_id].item()
            individual_predictions[name] = {"label": label_name, "confidence": confidence}

        stacked_probabilities = torch.stack(all_probabilities)
        avg_probabilities = torch.mean(stacked_probabilities, dim=0)

        final_prediction_id = torch.argmax(avg_probabilities).item()
        final_label = models["IndoBERT"]["model"].config.id2label[final_prediction_id] # Ambil id2label dari salah satu model
        final_confidence = avg_probabilities[0, final_prediction_id].item()

        st.success("Analisis Selesai!")
        
        st.subheader("Hasil Akhir")
        col1, col2 = st.columns(2)
        col1.metric(label="**Clas Prediction**", value=final_label)
        col2.metric(label="**Final Confidence Score**", value=f"{final_confidence:.2%}")
        
        with st.expander("Prediksi dari Setiap Model"):
            for name, prediction in individual_predictions.items():
                st.write(f"**{name}:**")
                st.json({
                    "Prediksi": prediction["label"],
                    "Confidence Score": f"{prediction['confidence']:.2%}"
                })

elif classify_button and not news_text.strip():
    st.warning("Mohon masukkan teks berita terlebih dahulu.")