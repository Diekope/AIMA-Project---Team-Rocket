import cv2
import json
import numpy as np
import streamlit as st
from pathlib import Path
from face_recognition import pipeline_recognition
from pipeline import split_bodies_and_save_faces, get_results

# Titre
st.title('AI-Models & Applications - Team Rocket')

## Partie image (sélection + affichage)
st.subheader('Insérez une image')

model = st.radio(
    "Choisissez un modèle",
    ["vgg16", "resnet50", "senet50"],
    horizontal=True
)

# Affichage de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    converted_image = cv2.imdecode(file_bytes, 1)
    
    faces_and_bodies_folder = split_bodies_and_save_faces(converted_image, file_name)
    faces_list = get_results(faces_and_bodies_folder)

    # Section résultats
    st.subheader('Résultats de l\'analyse')
    
    # Boucle sur les fichiers JSON
    for idx, json_path in enumerate(faces_list, 1):
        # Lecture du fichier JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Récupération des informations
        person_id = data.get('face_id')
        image_path = data.get('path') + '/' + data.get('original')
        age = data.get('age')
        emotion = data.get('emotion')
        gender = data.get('gender')
        
        name, prob = pipeline_recognition(image_path, model)
        prob_percent = f"{prob*100:.2f}"

        # Création de colonnes pour un affichage côte à côte
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Affichage de l'image
            if image_path and Path(image_path).exists():
                st.image(image_path, use_container_width=True)
            else:
                st.warning(f"Image non trouvée: {image_path}")
        
        with col2:
            st.write("## Résultats des prédictions")
            # Affichage des informations
            st.write(f"**Id: {idx}**")
            st.write(f"**Celebrity guessed: {name}**")
            st.write(f"**Confidence: {prob_percent}% ({prob})**")
            st.write(f"**Age:** {age}")
            st.write(f"**Emotion:** {emotion}")
            st.write(f"**Genre:** {gender}")
        
        # Séparateur entre les résultats
        st.divider()


