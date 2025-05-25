from tensorflow import keras

MODEL_PATH = "Models/convnet_EffcientB0.keras"

model = keras.models.load_model(MODEL_PATH)
model.summary()

model_json = model.to_json()
with open("modelEffcientB0.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("modelEffcientB0.weights.h5")
print("Saved model to disk")


