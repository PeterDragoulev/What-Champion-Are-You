import tensorflow as tf
from tensorflow.keras.applications.vgg16 import  preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import models
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate, Input, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pathlib
from tensorflow.keras import layers
import subprocess
import io
import base64
import cv2

class_names = ['Ahri', 'Akali', 'Blitzcrank', 'Caitlyn', 'Draven', 'Ezreal', 'LeeSin', 'Lux', 'MissFortune',
               'TwistedFate']

def generateHeatmapPre(base_model, loaded_model, file):
    last_conv_layer = None
    for layer in base_model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        print("No Conv2D layer found in base model. Cannot generate heatmap.")
        return

    heatmap_model = Model(inputs=base_model.input, outputs=last_conv_layer.output)
    image = load_img(file, target_size=(256, 256))
    img_tensor = tf.keras.preprocessing.image.img_to_array(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    predictions = loaded_model.predict(img_tensor)
    pred_index = np.argmax(predictions[0])
    predicted_class = class_names[pred_index]
    confidence = predictions[0][pred_index]
    feature_maps = heatmap_model.predict(img_tensor)
    attention_map = np.mean(feature_maps[0], axis=-1)
    attention_map = (attention_map - np.min(attention_map)) / (
            np.max(attention_map) - np.min(attention_map) + K.epsilon())
    attention_resized = tf.image.resize(
        np.expand_dims(attention_map, axis=-1),
        (img_tensor.shape[1], img_tensor.shape[2])
    ).numpy()
    attention_resized = attention_resized[:, :, 0]

    original_img = np.array(tf.keras.preprocessing.image.array_to_img(img_tensor[0]))
    height, width = original_img.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(original_img)
    x_positions = np.linspace(0, width - 1, 50)
    y_positions = np.linspace(0, height - 1, 50)
    X, Y = np.meshgrid(x_positions, y_positions)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    points = []
    colors = []
    sizes = []

    for x, y in zip(X_flat, Y_flat):
        x_idx = int(x * attention_resized.shape[1] / width)
        y_idx = int(y * attention_resized.shape[0] / height)

        if 0 <= x_idx < attention_resized.shape[1] and 0 <= y_idx < attention_resized.shape[0]:
            value = attention_resized[y_idx, x_idx]

            if value > 0.6:
                points.append((x, y))
                norm_value = (value - 0.6) / 0.4
                colors.append(plt.cm.jet(norm_value))
                sizes.append(15 * (0.5 + norm_value))

    if points:
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=sizes, alpha=0.9)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    return fig

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
