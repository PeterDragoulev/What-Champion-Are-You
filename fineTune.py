import model
import subSet
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate, Input, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pathlib
from tensorflow.keras import layers
import heatMapPremade

Load = True
createSubSets = True
createNewModel = True
imageGenerate = False
fineTune = False 

if createSubSets:
    subSet.makeSubset()

loaded_model, base_model = model.makeTransferModel()

if createNewModel:
    num_epoch = 5 
    model.runModel(loaded_model, num_epoch)

if fineTune:
    num_epoch = 10
    loaded_model = model.fine_tune_model(loaded_model, base_model, fine_tune_at=100)
    model.runModel(loaded_model, num_epoch)

if imageGenerate:
    model.getModelResults()
else:
    print("******NO MODEL LOADED******")

heatMapPremade.getHeatMap(base_model, loaded_model)
