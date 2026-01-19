import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions


def predict_celebrity(face_path, model, name_model):

    img = cv2.imread(face_path)
    if img is None:
        raise FileNotFoundError(face_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    return name

def model_choice(name_model):
    if name_model =="vgg16":
        model = VGGFace(model="vgg16", include_top=True)
    elif name_model =="resnet50":
        model = VGGFace(model="resnet50", include_top=True)
    elif name_model =="senet50":
        model = VGGFace(model="senet50", include_top=True)
    else :
        raise ValueError("The model name is wrong")
    return model

def main():
    # name_model = "vgg16"
    name_model = "senet50"
    # name_model = "senet50"

    model = model_choice(name_model)
    print(predict_celebrity("./src/visage.jpg",model,name_model))
main()