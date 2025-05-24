from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.utils import image_dataset_from_directory
import generatePhotos as gf
from sklearn.model_selection import KFold

img_w = 256
img_l = 256

def makeTransferModel():
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(img_w, img_l, 3),
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(img_w, img_l, 3))
    x = data_augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model

def fine_tune_model(mod, base_model, fine_tune_at=100):
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False 

    mod.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return mod


def makeModel():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
    )

    inputs = keras.Input(shape=(img_w, img_l, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1. / 255)(x)

    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.Conv2D(512, 3, padding="same", name='last_conv_layer')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    model.summary()

    return model


def runModel(model, num_epoch):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="convnet.keras",
            save_best_only=True,
            monitor="val_loss")
    ]

    new_base_dir = pathlib.Path("league_champs_small")

    train_dataset = image_dataset_from_directory(
        new_base_dir / "train",
        image_size=(img_w, img_l),
        batch_size=8)
    validation_dataset = image_dataset_from_directory(
        new_base_dir / "validation",
        image_size=(img_w, img_l),
        batch_size=8)

    history = model.fit(
        train_dataset,
        epochs=num_epoch,
        validation_data=validation_dataset,
        callbacks=callbacks)

    gf.trainingValidationMetrics(history)

    best_epoch = np.argmin(history.history["val_loss"])
    print(f"\nBest model was saved at epoch {best_epoch + 1} (0-based index + 1).")

    best_model = keras.models.load_model("convnet.keras")
    loss, accuracy = best_model.evaluate(validation_dataset)
    print(f"\nBest Model Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


def load_images_and_labels(directory, image_size):
    dataset = image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=1,
        shuffle=False
    )

    images = []
    labels = []
    for img, label in dataset:
        images.append(img.numpy()[0])
        labels.append(label.numpy()[0])

    return np.array(images), np.array(labels), dataset.class_names


def runModelKfold(model, num_epoch, k=5):
    data_dir = pathlib.Path("temp") 
    images, labels, class_names = load_images_and_labels(data_dir, (img_w, img_l))

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_no = 1
    all_scores = []
    best_val_accuracy = 0
    best_fold = 0

    for train_idx, val_idx in kfold.split(images):
        print(f"\nTraining on fold {fold_no}...")

        model = makeModel()

        fold_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=f"convnet_fold_{fold_no}.keras",
            save_best_only=True,
            monitor="val_loss"
        )

        x_train, x_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        history = model.fit(
            x_train, y_train,
            epochs=num_epoch,
            validation_data=(x_val, y_val),
            verbose=1,
            callbacks=[fold_checkpoint]
        )

        best_fold_model = keras.models.load_model(f"convnet_fold_{fold_no}.keras")

        loss, accuracy = best_fold_model.evaluate(x_val, y_val, verbose=0)
        print(f"Fold {fold_no} — Best model loss: {loss:.4f}, accuracy: {accuracy:.4f}")

        all_scores.append(accuracy)

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_fold = fold_no
            best_fold_model.save("convnet_best_kfold.keras")
            print(f"New best model saved from fold {fold_no} with accuracy: {accuracy:.4f}")

        fold_no += 1

    print(f"\nAverage validation accuracy over {k} folds: {np.mean(all_scores):.4f}")
    print(f"Best model was from fold {best_fold} with accuracy: {best_val_accuracy:.4f}")
    print("Best model saved as 'convnet_best_kfold.keras'")

    return class_names


def getModelResults():
    model = keras.models.load_model("convnet_best_kfold.keras")
    print("Loaded best model from k-fold cross-validation")

    lux_img_path = pathlib.Path(
        "temp") / "Lux" / "battle-academia-lux-prestige-edition-splash-art-lol-uhdpaper.com-4K-75-2235998235.jpg"
    img_tensor = gf.get_img_array(lux_img_path, target_size=(img_w, img_l))

    new_base_dir = pathlib.Path("league_champs_small")
    train_dataset = image_dataset_from_directory(
        new_base_dir / "train",
        image_size=(img_w, img_l),
        batch_size=8)
    class_names = train_dataset.class_names

    gf.getExamples(model, img_w, img_l)
    gf.getLayerMaps(model, img_tensor, class_names)
    gf.getHeatMap(model, img_tensor, class_names)
