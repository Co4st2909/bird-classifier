import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
# Caminho do modelo salvo
model_path = 'mobilenet_best_model_random_search.h5'

# Carregar o modelo
model = load_model(model_path)

# Dimensão da entrada do modelo
img_size = 224

# Classes do dataset
classes_selecionadas = [
    'Brown-Headed-Barbet',
    'Cattle-Egret',
    'Common-Kingfisher',
    'Common-Rosefinch',
    'Hoopoe'
]
# Função para pré-processar a imagem
def preprocess_image(image, target_size):
    # Converter para RGB (evita erros com PNG RGBA)
    image = image.convert("RGB")
    img = image.resize((target_size, target_size))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Personalização da página
st.set_page_config(
    page_title="Classificação de Aves",
    page_icon="🐦",
    layout="centered",
)

# CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .title {
        color: #343a40;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .instructions {
        color: #6c757d;
        font-size: 1.2rem;
        text-align: center;
    }
    .result {
        font-size: 1.5rem;
        font-weight: bold;
        color: #495057;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        background-color: #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título
st.markdown("<div class='title'>Classificação de Espécies de Aves</div>", unsafe_allow_html=True)

# Instruções
st.markdown(
    "<div class='instructions'>Faça o upload de uma imagem para descobrir qual das 5 espécies corresponde:<br>"
    "<strong>Brown-Headed-Barbet</strong>, <strong>Cattle-Egret</strong>, "
    "<strong>Common-Kingfisher</strong>, <strong>Common-Rosefinch</strong>, "
    "<strong>Hoopoe</strong>.</div>",
    unsafe_allow_html=True,
)

# Upload
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Mostrar a imagem
    st.image(image, caption="Imagem enviada", use_column_width=True)
    st.write("Classificando...")

    # Pré-processamento
    processed_image = preprocess_image(image, img_size)

    # Previsão
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes_selecionadas[predicted_class_index]
    certainty = predictions[0][predicted_class_index] * 100

    # Resultado
    st.markdown(
        f"<div class='result'>"
        f"Espécie prevista: <strong>{predicted_class}</strong><br>"
        f"Certeza: <strong>{certainty:.2f}%</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )
