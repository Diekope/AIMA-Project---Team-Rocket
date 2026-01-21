# AIMA-Project---Team-Rocket

## Comment utiliser le projet

## Création de l'environnement de travail
Commencez par créer un evnironnement de travail sur python 3.10 pour pouvoir installer toutes les dépendances nécessaires à l'exécution du script.
```bash
conda create -n env_name python=3.10
```

Ensuite activez cet environnement :
```bash
conda activate env_name
```

Par la suite, installez les librairies suivante via pip et conda :
```bash
conda install -c conda-forge tensorflow=2.11
pip install git+https://github.com/rcmalli/keras-vggface.git
pip install opencv-python
```

Enfin, installez les librairies restantes :
```bash
pip install -r src/requirements.txt
```

Il y aura surement un message d'erreur de dépendances mais ne faites pas attention à cela.

Et après toutes ces étapes, vous pourrez enfin exécuter le script voulu.

```bash
streamlit run src/gui.py
```