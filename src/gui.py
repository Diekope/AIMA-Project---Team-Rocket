import cv2
import numpy as np
import streamlit as st
from pipeline import split_bodies_and_save_faces

# Titre
st.title('AI-Models & Applications - Team Rocket')

## Partie image (sélection + affichage)
st.subheader('Insérez une image')

# Affichage de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    converted_image = cv2.imdecode(file_bytes, 1)
    split_bodies_and_save_faces(converted_image, file_name)

    st.image(uploaded_file)


