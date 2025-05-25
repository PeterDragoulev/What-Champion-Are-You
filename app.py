from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import os
import sys

from matplotlib import pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from heatMapWeb import getHeatMapWeb, generateHeatmapPre
import io
import base64

sys.path.append(os.path.abspath('./model'))
from load import *

class_names = ['Ahri', 'Akali', 'Blitzcrank', 'Caitlyn', 'Draven', 'Ezreal', 'LeeSin', 'Lux', 'MissFortune',
               'TwistedFate']

app = Flask(__name__)
model1 = init()
model2 = loadVGG16()
base_efficientnet = load_base_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    modelnum = request.form.get('modelnum')
    file = request.files['image']

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file.save('out.jpg')
    heatmap_base64 = None

    if modelnum == '1':
        img = Image.open('out.jpg').convert('RGB')
        img = img.resize((256, 256))
        x = np.expand_dims(np.array(img), axis=0)

        selected_model = model1
        prediction = selected_model.predict(x)
        class_result = class_names[np.argmax(prediction)]
        heatmap_base64 = getHeatMapWeb(selected_model, x, class_names)

    else:
        img = Image.open('out.jpg').convert('RGB')
        img = img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        x = np.expand_dims(img_array, axis=0)

        selected_model = model2
        prediction = selected_model.predict(x)
        class_result = class_names[np.argmax(prediction)]

        heatmap_image = generateHeatmapPre(base_efficientnet, selected_model, 'out.jpg')

        if heatmap_image:
            buffer = io.BytesIO()
            heatmap_image.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(heatmap_image)

    return jsonify({
        "prediction": class_result,
        "model": modelnum,
        "heatmap": heatmap_base64
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

