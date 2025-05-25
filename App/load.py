import numpy as np
import keras.models
from keras.models import model_from_json
from PIL import Image
import tensorflow as tf

def init():
    json_file= open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model= model_from_json(loaded_model_json)

    loaded_model.load_weights("model.weights.h5")
    print("Loaded Model from disk")

    loaded_model.compile(loss='categorical_crossentropy')

    return loaded_model

def loadVGG16():
    json_file= open('modelEffcientB0.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model= model_from_json(loaded_model_json)

    loaded_model.load_weights("modelEffcientB0.weights.h5")
    print("Loaded Model from disk")

    loaded_model.compile(loss='categorical_crossentropy')

    return loaded_model

def load_base_model():
    json_file = open('baseEfficientNetB0.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("baseEfficientNetB0.weights.h5")
    print("Loaded base EfficientNetB0 model from disk")

    return loaded_model
