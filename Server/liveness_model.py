from tensorflow import keras
import cv2
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

model = keras.models.load_model(os.path.join(package_directory, 'my_model.h5'))

def get_scores(X):
    classes = ['Attack', 'Real']
    scores = model.predict(X)[0]
    sc_d = {k:str(v) for k,v in zip(classes, scores)}
    print(sc_d)
    return sc_d