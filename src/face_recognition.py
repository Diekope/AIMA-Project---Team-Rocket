import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions


def predict_celebrity(face_path, model, name_model):
    """
    To give a list of predicted celebrities for a face
    """
    img = cv2.imread(face_path)
    if img is None:
        raise FileNotFoundError(face_path)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img.astype(np.float32), axis=0)

    if name_model =="vgg16":
        x = preprocess_input(x, version=1)  
    else:
        x = preprocess_input(x, version=2)

    preds = model.predict(x)
    decoded = decode_predictions(preds)[0]
    print(decoded)
    name = decoded[0][0].replace("b'","").replace("'","")
    name = name.replace("_", " ")

    max_element = max(decoded, key=lambda x: x[1])
    prob = max_element[1]
    return name, prob

def model_choice(name_model):
    """
    Loads and return the selected model
    """
    if name_model =="vgg16":
        model = VGGFace(model="vgg16", include_top=True)
    elif name_model =="resnet50":
        model = VGGFace(model="resnet50", include_top=True)
    elif name_model =="senet50":
        model = VGGFace(model="senet50", include_top=True)
    else :
        raise ValueError("The model name is wrong")
    return model

def pipeline_recognition(img, model_):
    """
    Load a model and predict the celebirty of a given image
    """
    model = model_choice(model_)
    name, prob = predict_celebrity(img, model, model_)

    return name, prob

def main():
    # name_model = "vgg16"
    name_model = "senet50"
    # name_model = "senet50"

    model = model_choice(name_model)
    print(predict_celebrity("/Users/ValQuiTravaille/Projects/AIMA-Project---Team-Rocket/img/working/b/b-person-0-bb-14-36-163-267_face_1.jpg",model,name_model))

# main()