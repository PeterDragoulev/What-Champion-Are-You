# League Champion Image Guesser

This project consists of two main parts: a deep learning model for guessing which League of Legends champion, character, a user's uploaded photo resembles, and a Flask web application to interact with these models.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Information](#model-creation-details)
- [Limitations](#limitations)
- [Closing Thoughts](#closing-thoughts)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Project Overview

The core idea is to provide a fun and engaging experience where users can upload their photo and see which League of Legends champion the model predicts they look most like. This involves:

1.  **Model Creation:**
    * **"From Scratch" Champion Guesser:** A custom-built convolutional neural network (CNN) designed to take an image as input and output a prediction of the most similar League of Legends champion. This model was developed from the ground up, focusing on understanding the underlying principles of image classification.
    * **EfficientNet Extension:** An additional model that leverages the powerful EfficientNet architecture, a pre-trained deep learning model known for its efficiency and accuracy in image recognition tasks. This model serves as an extension, potentially offering improved performance or different insights compared to the "from scratch" model.

2.  **Flask Web Application:**
    * A lightweight web application built with Flask, allowing users to easily upload their images.
    * The app integrates with both trained models, enabling users to get predictions from either the "from scratch" model or the EfficientNet-based model.
    * The user interface provides a simple way to upload images and displays the predicted champion.

## Model Creation Details

### "From Scratch" Champion Guesser

This model was built using TensorFlow and trained on a dataset of League of Legends champion images, consisting largely of splash arts, fan art, and other drawings for the different champions. It utilizes a custom CNN architecture consisting of Convolutional Layers, Pooling Layers, Normalization Layers, and dropout layers. To avoid the model from memorizing training images, as we had a very limited quantity of total images which could be used, we employed a few techniques, the first of which was rotating images by 0-10 degrees each time an image was processed, as well as the aforementioned dropout layers. In addition, we used K-fold cross-validation training for our final model iteration to confirm our model was generalizing and not memorizing specific data, and that human intervention had not caused inadvertent fitting to our validation data. The final model was trained on 200 epochs, as part of a 5-fold batch, which each ran 200 epochs. This resulted in the model guessing around 60% of validation tests correctly, well above the baseline of 10% if random guessing was used, as we had around 100 images for each character.

### EfficientNet Extension

This model leverages the pre-trained EfficientNetB0 architecture. The EfficientNet model was fine-tuned on the League of Legends champion image dataset. Transfer learning techniques were employed, where the pre-trained convolutional base was frozen and new classification layers were added and trained on our specific task. Overall, this model far surpassed the home-grown model we made, as could be expected with our data offering limited ability for data abstraction.

## Limitations

This model is meant to be used for fun to see what League of Legends champion you look like; however, the discrepancy between training data, cartoon drawings of champions, and use data, pictures of humans, means a lot of what was learned will not be useful for its predictions. However, testing with images of cosplayers, we were able to get decent results, with our heat maps showing the model had learned key features of game characters, though some will be shown below, some will not be useful for judging normal people.

First, an example showing good abstraction;
(Clarification Model 1 refers to our model, while Model 2 is the extended version of EfficientNetB0)

![Ezreal cosplay](Images/ezrealGood.png)

As seen above, the main area highlighted is Ezreal's short blond hair, which is a feature commonly seen in regular non-cosplay pictures of people.

Next, a feature which is very well defined, but will not be useful in most cases;

![Akali cosplay](Images/AkaliDagger.png)

Here it is clear that the guess is almost solely made off the dagger which Akali has, a distinct sickle, something I doubt many people will have with them. 

Finally, I will showcase the limitations of the model.

![Blitz Bad](Images/BlitzBad.png)
![Blitz Good](Images/BlitzGood.png)

Here we see the considerable shortcoming of noise in the background; this noise is enough to completely make our own model guess wrong, and it also confuses the EfficientNetB0 model, though it still manages to guess through the blocky arm structure.

## Closing thoughts
The performance of the models far surpassed my expectations of what could be achieved, setting out. With our own model guessing the correct champion for a cosplayer 40% of the time, and the EfficientNetB0 model getting a staggering 80% correct. I believe that future refinements, such as filtering images to focus on central parts, or even better, introducing new layers with attention mechanisms to better capture what is "important" in a given image. Additionally, a large distraction from what I have seen in the model results were bright spots, causing overactivation, so a likely solution would have to deal with the issue by mimicking different conditions of lighting. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What you need to install the software:

* Python 3.x
* pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/PeterDragoulev/What-Champion-Are-You.git
    cd What-Champion-Are-You
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Ensure your virtual environment is active.**

2.  **Run the Flask application:**

    ```bash
    python app.py
    ```
3.  **Open your web browser** and navigate to `http://127.0.0.1:5000/` (or the address displayed in your terminal).

## Project Structure

```
What-Champion-Are-You/
│
├── .idea/                           # IDE configuration (optional to version control)
├── App/                             # Web app-related code and models
│   ├── templates/                   # HTML templates
│   ├── __pycache__/                # Cached Python bytecode
│   ├── app.py                       # Flask app entry point
│   ├── baseEfficientNetB0.json      # Base model architecture (EfficientNetB0)
│   ├── baseEfficientNetB0.weights.h5
│   ├── heatMapWeb.py                # Web heatmap generation logic
│   ├── load.py                      # Model loading utilities
│   ├── makeWebModel.py              # Lightweight web model script
│   ├── model.json                   # Alternate model architecture
│   ├── model.weights.h5
│   ├── modelEffcientB0.json         # EfficientNetB0 model variant
│   ├── modelEffcientB0.weights.h5
│   └── out.jpg                      # Sample model output image
│
├── Images/                          # Manually generated/test images
│   ├── AkaliDagger.png
│   ├── BlitzBad.png
│   ├── BlitzGood.png
│   └── ezrealGood.png
│
├── league_champs_small/            # Dataset
│   ├── train/
│   └── validation/
│
├── Models/                          # Saved trained models
│   ├── convnet.keras
│   ├── convnet_best_kfold.keras
│   ├── convnet_EffcientB0.keras
│   ├── convnet_from_scratch_with_augmentation.keras
│   └── convnet_on_other_model.keras
│
├── __pycache__/                    # Cached Python bytecode
│
├── fineTune.py                      # Script for fine-tuning models
├── generatePhotos.py               # Image generation and preprocessing
├── heatMapPremade.py               # Standalone heatmap logic
├── main.py                          # Central experiment or training script
├── model.py                         # Model definition (e.g., CNN architecture)
├── README.md                        # Project overview and instructions
├── requirements.txt                 # Python dependencies
└── subSet.py                        # Script to create dataset subsets

```

## Usage
If anyone has any interest in what was done, feel free to use anything available.

## Acknowledgments
* François Chollet's book Deep Learning with Python was largely what helped me understand how to create this project, so lots of code was inspired by it
* Riot Games for creating all the charcters used

