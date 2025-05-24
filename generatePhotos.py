from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.utils import image_dataset_from_directory
import cv2

import os
import subprocess
import io
import base64

global_gc_img = 1
global_gc_heat = 1

def remove_zone_identifier(file_path):
    ads_path = f"{file_path}:Zone.Identifier"
    try:
        if os.name == 'nt': 
            subprocess.run(['cmd', '/c', f'del "{ads_path}"'], check=True)
        elif os.name == 'posix':
            subprocess.run(['rm', f'{ads_path}'], check=True)
        print(f"Removed Zone.Identifier from {file_path}")
    except subprocess.CalledProcessError:
        print(f"No Zone.Identifier found or failed to remove for {file_path}")

def get_prediction_label(model, img_tensor, class_names):
    predictions = model.predict(img_tensor)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    predicted_label = class_names[predicted_class_index]
    return predicted_label, confidence

def getPred(model, img_tensor, class_names):
    global global_gc_img
    predicted_label, confidence = get_prediction_label(model, img_tensor, class_names)
    print(f"Predicted class: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(img_tensor[0].astype("uint8"))
    plt.title(f"Predicted: {predicted_label} ({confidence:.2f})")
    plt.savefig(f"prediction_result_{global_gc_img}.png", bbox_inches='tight')
    global_gc_img += 1


def generate_heatmap(model, img_tensor, class_names, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    img = img_tensor[0].astype("uint8")
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap_colored * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    return heatmap, superimposed_img

def get_img_array(img_path, target_size):
    img = keras.utils.load_img(
        img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def getExamples(model, img_w, img_l):
    original_dir = pathlib.Path("league_champs_small")

    train_dataset = image_dataset_from_directory(
        original_dir / "train",
        image_size=(img_w, img_l),
        batch_size=4
    )
    class_names = train_dataset.class_names

    for category in class_names:
        dir_fetch = original_dir / "validation" / category
        onlyFiles = [f for f in dir_fetch.iterdir() if f.is_file()]

        if not onlyFiles:
            continue

        example_file = onlyFiles[0]
        print(f"Using file: {example_file.name}")
        img_tensor = get_img_array(example_file, target_size=(img_w, img_l))
        getHeatMap(model, img_tensor, class_names)
        getPred(model, img_tensor, class_names)

def getLayerMaps(model, img_tensor, class_names):
    predicted_label, confidence = get_prediction_label(model, img_tensor, class_names)
    print(f"Predicted class: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(img_tensor[0].astype("uint8"))
    plt.title(f"Predicted: {predicted_label} ({confidence:.2f})")
    plt.savefig("prediction_result_lux.png", bbox_inches='tight')
    layer_outputs = []
    layer_names = []
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros(((size + 1) * n_cols - 1,
                                 images_per_row * (size + 1) - 1))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_index = col * images_per_row + row
                channel_image = layer_activation[0, :, :, channel_index].copy()
                if channel_image.sum() != 0:
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[
                col * (size + 1): (col + 1) * size + col,
                row * (size + 1): (row + 1) * size + row] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.axis("off")
        plt.imshow(display_grid, aspect="auto", cmap="viridis")
        plt.savefig(f"{layer_name}_activation.png", bbox_inches='tight')
        plt.close()

def trainingValidationMetrics(history):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, "bo-", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r.-", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "r.-", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png", bbox_inches='tight')
    plt.show()

def getHeatMap(model, img_tensor, class_names ):
    global global_gc_heat
    last_conv_layer_name = "last_conv_layer"

    predicted_class_index = np.argmax(model.predict(img_tensor)[0])

    heatmap, superimposed_img = generate_heatmap(
        model,
        img_tensor,
        class_names,
        last_conv_layer_name,
        pred_index=predicted_class_index
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_tensor[0].astype("uint8"))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    predicted_label, confidence = get_prediction_label(model, img_tensor, class_names)

    plt.subplot(1, 3, 3)
    plt.title(f"Heatmap Overlay\nPredicted: {predicted_label} ({confidence:.2f})")
    plt.imshow(superimposed_img)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"heatmap_analysis_{global_gc_heat}.png", bbox_inches='tight')
    global_gc_heat += 1


def getHeatMapWeb(model, img_tensor, class_names):
    last_conv_layer_name = "last_conv_layer"
    predicted_class_index = np.argmax(model.predict(img_tensor)[0])

    heatmap, superimposed_img = generate_heatmap(
        model,
        img_tensor,
        class_names,
        last_conv_layer_name,
        pred_index=predicted_class_index
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(superimposed_img)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    img_bytes = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_bytes