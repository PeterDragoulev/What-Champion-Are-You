import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import models
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import model
import subSet
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate, Input, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pathlib
from tensorflow.keras import layers
import heatMapPremade

global_gc_heat = 0

MODEL_PATH = "convnet.keras"

class_names = ['Ahri', 'Akali', 'Blitzcrank', 'Caitlyn', 'Draven', 'Ezreal', 'LeeSin', 'Lux', 'MissFortune',
               'TwistedFate']

dir_fetch = pathlib.Path("test")
onlyFiles = [f for f in dir_fetch.iterdir() if f.is_file()]


def create_dot_heatmap(attention_map, threshold=0.5, dot_size=20, density=100):
    fig, ax = plt.subplots(figsize=(6, 6))
    height, width = attention_map.shape
    x_positions = np.linspace(0, width - 1, density)
    y_positions = np.linspace(0, height - 1, density)
    X, Y = np.meshgrid(x_positions, y_positions)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    points = []
    colors = []
    sizes = []

    for x, y in zip(X_flat, Y_flat):
        x_idx = int(x)
        y_idx = int(y)
        if 0 <= x_idx < width and 0 <= y_idx < height:
            value = attention_map[y_idx, x_idx]
            if value > threshold:
                points.append((x, y))
                norm_value = (value - threshold) / (1 - threshold)
                colors.append(plt.cm.jet(norm_value))
                sizes.append(dot_size * (0.5 + norm_value))

    if points:
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=sizes, alpha=0.7)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    return fig


def getHeatMap(base_model, loaded_model):
    global global_gc_heat
    last_conv_layer = None
    for layer in base_model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        print("No Conv2D layer found in base model. Cannot generate heatmap.")
        return

    heatmap_model = Model(inputs=base_model.input, outputs=last_conv_layer.output)

    for file in onlyFiles:
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
        dot_heatmap_fig = create_dot_heatmap(attention_resized, threshold=0.6, dot_size=15, density=50)
        overlay_fig, overlay_ax = plt.subplots(figsize=(6, 6))
        overlay_ax.imshow(original_img)
        height, width = original_img.shape[:2]


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
            overlay_ax.scatter(points[:, 0], points[:, 1], c=colors, s=sizes, alpha=0.7)

        overlay_ax.set_xlim(0, width)
        overlay_ax.set_ylim(height, 0)
        overlay_ax.axis('off')

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(original_img.astype("uint8"))
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(original_img.astype("uint8"))

        if points.size > 0:
            plt.scatter(points[:, 0], points[:, 1], c=colors, s=sizes, alpha=0.7)

        plt.title("Dot Heatmap Overlay")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(np.zeros_like(original_img))

        if points.size > 0:
            plt.scatter(points[:, 0], points[:, 1], c=colors, s=sizes, alpha=0.9)

        plt.title(f"Pure Dot Heatmap\nPredicted: {predicted_class} ({confidence:.2f})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"heatmap_analysis_{global_gc_heat}.png", bbox_inches='tight')
        global_gc_heat += 1
        plt.close('all')


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