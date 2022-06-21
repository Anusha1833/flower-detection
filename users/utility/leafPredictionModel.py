from keras.preprocessing import image
# from PIL import image
import numpy as np
from .deeplearning import graph, model, output_list
import base64
import os
from django.conf import settings


def predict_leaf(filepath):
    print(f"Works is Good {filepath}")
    # print("MyFileeeeeee:",myfile)
    # b64_img = base64.b64encode(myfile.read()).decode('ascii')
    # print("b64_img:",b64_img)
    myfile = os.path.join(settings.MEDIA_ROOT, filepath)
    img = image.load_img(myfile, target_size=(224, 224))
    # print("IMG1:",img)
    img = image.img_to_array(img)
    # print("IMG2:",img)
    img = np.expand_dims(img, axis=0)
    # print("IMG3:",img)
    img = img / 255
    # print("IMG4:",img)
    with graph.as_default():
        prediction = model.predict(img)
        print("prediction:", prediction)
    prediction_flatten = prediction.flatten()
    print("prediction_flatten:", prediction_flatten)
    max_val_index = np.argmax(prediction_flatten)
    print("max_val_index:", max_val_index)
    result = output_list[max_val_index]
    print("result:", result)
    return result, filepath

