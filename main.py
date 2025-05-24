import subSet
import model
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate, Input, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pathlib
from tensorflow.keras import layers

createSubSets = True
loadPretrainedModel = True
extendModel = True
createNewModel = False
runKfold = True
generateVisualizations = True
saveOriginalExtendedModel = True 

PRETRAINED_MODEL_PATH = "convnet_from_scratch_with_augmentation.keras"
EXTENDED_MODEL_PATH = "convnet_extended_base.keras"
KFOLD_BEST_MODEL_PATH = "convnet_best_kfold.keras"
NUM_EPOCHS = 30
K_FOLDS = 5

if createSubSets:
    print("Creating data subsets...")
    subSet.makeSubset()

current_model = None

if loadPretrainedModel:
    print(f"Loading pre-trained model from {PRETRAINED_MODEL_PATH}...")
    try:
        base_model = keras.models.load_model(PRETRAINED_MODEL_PATH)
        base_model.summary()
        current_model = base_model

        if extendModel:
            print("Extending the model with additional layers...")
            x = base_model.layers[-4].output
            extended_model = keras.Model(inputs=base_model.input, outputs=outputs)
            extended_model.compile(loss="sparse_categorical_crossentropy",
                                   optimizer="rmsprop",
                                   metrics=["accuracy"])
            extended_model.summary()
            current_model = extended_model
            if saveOriginalExtendedModel:
                print(f"Saving extended base model to {EXTENDED_MODEL_PATH}...")
                extended_model.save(EXTENDED_MODEL_PATH)
                print("Extended model saved successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        current_model = None

if createNewModel:
    print("Creating new model from scratch...")
    current_model = model.makeModel()

if runKfold and current_model is not None:
    print(f"Running {K_FOLDS}-fold cross-validation for {NUM_EPOCHS} epochs...")
    model.runModelKfold(current_model, NUM_EPOCHS, K_FOLDS)

    print(f"Loading best model from K-fold validation: {KFOLD_BEST_MODEL_PATH}")
    try:
        current_model = keras.models.load_model(KFOLD_BEST_MODEL_PATH)
        print("Successfully loaded best model from K-fold validation")
    except Exception as e:
        print(f"Error loading best K-fold model: {e}")
        print("Continuing with previously loaded model")

if generateVisualizations and current_model is not None:
    print("Generating model visualizations...")
    model.getModelResults()
else:
    print("******NO MODEL LOADED OR VISUALIZATIONS DISABLED******")

print("Script execution complete.")
