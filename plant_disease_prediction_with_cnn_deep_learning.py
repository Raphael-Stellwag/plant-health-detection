# -*- coding: utf-8 -*-
"""Plant Disease Prediction with CNN Deep Learning

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/plant-disease-prediction-with-cnn-deep-learning-b81ca2df-f597-4655-bac7-96849b0e5394.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20241221/auto/storage/goog4_request%26X-Goog-Date%3D20241221T064752Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D178a306a190031336dd3e98e6d7c5f320ad17d041f1f1495e6fec79136ce0ab6c064d9265cebd32dd1680f369518946da597f62970ce6090d39f86c1ad66d78839236a289c77392ec13b0aa9498f00b8111fcfdd01e0d46ef7b19aadce260ba056a9dac3cb08bb8be38667feaccee129055c2a434fe922a8ddc7f76c87b6b666a5713b5564e30b5a84ba0321d6aac81727502ee9efe136e4cdf0368cf9ebaa9e9f287bd7bef4d10f2b44dec49e175c9863526ca625479b38e0971d03ec6507e151926f508ae610ac0d2a92e298537eec023670139e8e7bc6faaf763e2d54cc2f9028734f61f0bac06808c40d662507598faf0ef2321a7ae49a9bde8b16ff255e
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
# import kagglehub
# abdallahalidev_plantvillage_dataset_path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')

print('Data source import complete.')

"""# Seeding for reproducibility"""

# Set seeds for reproducibility--设置随机种子
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)

"""#  Importing the Libraries"""

import os
import json
from zipfile import ZipFile
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

print(os.listdir("data/plantvillage dataset"))


print(len(os.listdir("data/plantvillage dataset/segmented")))
print(os.listdir("data/plantvillage dataset/segmented")[:5])

print(len(os.listdir("data/plantvillage dataset/color")))
print(os.listdir("data/plantvillage dataset/color")[:5])

print(len(os.listdir("data/plantvillage dataset/grayscale")))
print(os.listdir("data/plantvillage dataset/grayscale")[:5])

print(len(os.listdir("data/plantvillage dataset/color/Grape___healthy")))
print(os.listdir("data/plantvillage dataset/color/Grape___healthy")[:5])

"""# Data Preprocessing"""

# Dataset Path
base_dir = 'data/plantvillage dataset'

image_path = 'data/plantvillage dataset/color/Apple___Cedar_apple_rust/025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG'

# Read the image
img = mpimg.imread(image_path)

print(img.shape)
# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis numbers
plt.show()

image_path = 'data/plantvillage dataset/color/Apple___Cedar_apple_rust/025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG'

# Read the image
img = mpimg.imread(image_path)

print(img)

# Image Parameters
img_size = 224
batch_size = 32

"""# Train Test Split"""

# Image Data Generators
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use 20% of data for validation
)

# Train Generator
train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

# Validation Generator
validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

"""# Convolutional Neural Network"""

# Model Definition
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))

# model summary
model.summary()

# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""# Model training"""

# Training the Model
history = model.fit(
    train_generator,
    steps_per_epoch=500 ,  # train_generator.samples // batch_size Number of steps per epoch
    epochs=1,  # Number of epochs--迭代数量，原先是5
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size  # Validation steps
)

"""# Model Evaluation"""

# Model Evaluation
print("Evaluating model...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

"""# Building a Predictive System"""

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

# Create a mapping from class indices to class names
class_indices = {v: k for k, v in train_generator.class_indices.items()}

class_indices

# saving the class names as json file
json.dump(class_indices, open('class_indices.json', 'w'))

import json

# Example disease information dictionary
disease_info = {
    'Apple___Apple_scab': 'Apple scab is a common fungal disease that affects apple trees, causing dark, scabby lesions on the leaves, fruit, and stems. It can lead to significant crop loss if not controlled. Management includes pruning, using resistant varieties, and applying fungicides.',
    'Apple___Black_rot': 'Black rot is a fungal disease of apple trees that results in fruit rot, leaf spots, and cankers on branches. It is especially damaging in humid conditions. Control involves removing infected material and applying fungicides.',
    'Apple___Cedar_apple_rust': 'Cedar apple rust is a fungal disease affecting apples and junipers, characterized by yellow-orange spots on apple leaves and galls on junipers. Management includes removing nearby junipers and using resistant apple varieties.',
    'Apple___healthy': 'This apple tree is healthy with no visible signs of disease or pest infestation. Regular monitoring and proper care are essential to maintain tree health.',
    'Blueberry___healthy': 'This blueberry plant is healthy with no visible signs of disease or nutrient deficiencies. Proper care, including correct watering and fertilization, will keep the plant thriving.',
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery mildew is a fungal disease that affects cherry trees, causing a white, powdery coating on leaves, buds, and fruits. It thrives in warm, dry conditions and can be controlled by improving air circulation and applying fungicides.',
    'Cherry_(including_sour)___healthy': 'This cherry tree is healthy and free of disease. Proper pruning, watering, and pest control are key to maintaining tree health.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Gray leaf spot, caused by the fungus Cercospora zeae-maydis, is a serious disease of maize. It causes gray or tan lesions on leaves, reducing photosynthesis and yield. Crop rotation and resistant hybrids are effective management strategies.',
    'Corn_(maize)___Common_rust_': 'Common rust in corn is caused by the fungus Puccinia sorghi, leading to reddish-brown pustules on leaves. It can reduce yield, especially in susceptible hybrids. Fungicides and resistant hybrids help manage the disease.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern leaf blight is a fungal disease of maize that produces long, gray-green lesions on leaves. It can lead to significant yield losses. Management includes using resistant hybrids and applying fungicides.',
    'Corn_(maize)___healthy': 'This maize plant is healthy, showing no signs of disease or nutrient deficiencies. Good agricultural practices, including proper fertilization and pest management, help maintain plant health.',
    'Grape___Black_rot': 'Black rot is a fungal disease of grapes that causes black lesions on leaves, shoots, and fruit. It can severely reduce yield if not controlled. Management includes removing infected plant material and applying fungicides.',
    'Grape___Esca_(Black_Measles)': 'Esca, also known as black measles, is a complex disease affecting grapevines. It causes dark streaks on the wood and black spots on the fruit. The disease is difficult to control, but good vineyard hygiene and pruning practices can help.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight caused by Isariopsis spp. results in dark, angular spots on grape leaves, leading to premature leaf drop. It can reduce the vigor of the vine and fruit quality. Control measures include removing affected leaves and applying fungicides.',
    'Grape___healthy': 'This grapevine is healthy and free from diseases. Regular pruning, pest control, and proper irrigation are essential to maintain vine health.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Huanglongbing (HLB), also known as citrus greening, is a bacterial disease that affects citrus trees. It causes yellowing of the leaves, green misshapen fruit, and eventual tree decline. There is no cure, and management involves controlling the psyllid vector and removing infected trees.',
    'Peach___Bacterial_spot': 'Bacterial spot is a disease of peach trees caused by Xanthomonas spp. It leads to dark spots on leaves, fruit, and twigs, causing defoliation and reduced fruit quality. Copper-based sprays and resistant varieties are used for management.',
    'Peach___healthy': 'This peach tree is healthy, with no signs of disease or pests. Regular care, including proper fertilization, watering, and pruning, is important to maintain tree health.',
    'Pepper,_bell___Bacterial_spot': 'Bacterial spot in bell peppers is caused by Xanthomonas campestris, leading to dark, water-soaked spots on leaves and fruit. It can severely reduce yield and fruit quality. Management includes using resistant varieties and applying bactericides.',
    'Pepper,_bell___healthy': 'This bell pepper plant is healthy and free from disease. Proper care, including watering, fertilization, and pest management, will help maintain plant health.',
    'Potato___Early_blight': 'Early blight is a fungal disease of potatoes caused by Alternaria solani. It causes dark, concentric spots on leaves and can lead to significant yield loss. Crop rotation, resistant varieties, and fungicides are key management strategies.',
    'Potato___Late_blight': 'Late blight, caused by Phytophthora infestans, is a devastating disease of potatoes. It leads to water-soaked lesions on leaves and tubers, often resulting in total crop loss. Immediate removal of infected plants and fungicide applications are essential for control.',
    'Potato___healthy': 'This potato plant is healthy and shows no signs of disease. Good agricultural practices, including crop rotation and proper fertilization, are essential to maintain plant health.',
    'Raspberry___healthy': 'This raspberry plant is healthy and free from disease. Regular pruning, pest control, and proper irrigation are important for maintaining plant health.',
    'Soybean___healthy': 'This soybean plant is healthy with no visible signs of disease. Good agricultural practices, including crop rotation and proper fertilization, are essential for maintaining plant health.',
    'Squash___Powdery_mildew': 'Powdery mildew is a common fungal disease of squash, causing a white, powdery growth on leaves. It can reduce plant vigor and yield. Management includes improving air circulation, removing infected leaves, and applying fungicides.',
    'Strawberry___Leaf_scorch': 'Leaf scorch in strawberries is caused by the fungus Diplocarpon earliana. It results in dark, angular spots on leaves, leading to leaf drop and reduced plant vigor. Removing infected leaves and applying fungicides can help manage the disease.',
    'Strawberry___healthy': 'This strawberry plant is healthy, showing no signs of disease or pests. Proper care, including regular watering and fertilization, is key to maintaining plant health.',
    'Tomato___Bacterial_spot': 'Bacterial spot is a serious disease of tomatoes caused by Xanthomonas spp. It leads to dark, water-soaked spots on leaves, stems, and fruit. The disease can reduce yield and fruit quality. Management includes using resistant varieties and applying bactericides.',
    'Tomato___Early_blight': 'Early blight is a common fungal disease of tomatoes caused by Alternaria solani. It causes dark, concentric spots on leaves and stems, reducing plant vigor and yield. Crop rotation, resistant varieties, and fungicides are effective control measures.',
    'Tomato___Late_blight': 'Late blight, caused by Phytophthora infestans, is a destructive disease of tomatoes. It leads to water-soaked lesions on leaves and fruit, often causing total crop loss. Immediate removal of infected plants and fungicide applications are essential for control.',
    'Tomato___Leaf_Mold': 'Leaf mold is a fungal disease of tomatoes caused by Cladosporium fulvum. It results in yellowing of leaves and a fuzzy, grayish-brown growth on the underside of leaves. Improved air circulation and fungicides can help manage the disease.',
    'Tomato___Septoria_leaf_spot': 'Septoria leaf spot is a fungal disease of tomatoes caused by Septoria lycopersici. It leads to small, circular spots on leaves, causing defoliation and reducing yield. Control includes removing infected leaves and applying fungicides.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider mites are tiny pests that feed on tomato plants, causing yellowing and stippling of leaves. Severe infestations can reduce plant vigor and yield. Management includes using miticides and promoting natural predators.',
    'Tomato___Target_Spot': 'Target spot is a fungal disease of tomatoes caused by Corynespora cassiicola. It leads to circular, target-like spots on leaves and fruit, reducing yield and fruit quality. Management includes removing infected material and applying fungicides.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato yellow leaf curl virus (TYLCV) is a viral disease transmitted by whiteflies. It causes yellowing and curling of leaves, stunting of plants, and reduced fruit production. Control involves managing whitefly populations and using resistant varieties.',
    'Tomato___Tomato_mosaic_virus': 'Tomato mosaic virus (TMV) is a viral disease that causes mottled leaves, stunted growth, and reduced fruit quality. It is managed by using resistant varieties and practicing good sanitation to prevent the spread of the virus.',
    'Tomato___healthy': 'This tomato plant is healthy, with no visible signs of disease or pest infestation. Regular care, including proper watering, fertilization, and pest management, is essential for maintaining plant health.'
}

# Convert the dictionary to a JSON file
with open('disease_info.json', 'w') as file:
    json.dump(disease_info, file, indent=4)

print("Disease information has been successfully written to disease_info.json")

"""# Prediction"""

import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the disease information from the JSON file
with open('disease_info.json', 'r') as file:
    disease_info = json.load(file)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

# Example Usage
image_path = 'data/plantvillage dataset/color/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/00a20f6f-e8bd-4453-9e25-36ea70feb626___RS_GLSp 4655.JPG'
predicted_class_name = predict_image_class(model, image_path, class_indices)

# Output the result
print("Predicted Class Name:", predicted_class_name)

# Get the corresponding disease information
disease_info_text = disease_info.get(predicted_class_name, "No information available for this class.")
print("Disease Information:", disease_info_text)

# Display the image
img = Image.open(image_path)
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis('off')  # Turn off axis numbers
plt.title(f'Predicted: {predicted_class_name}')
plt.show()

import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the disease information from the JSON file
with open('disease_info.json', 'r') as file:
    disease_info = json.load(file)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

# Example Usage
image_path = 'data/plantvillage dataset/segmented/Tomato___Early_blight/01861c93-ea8b-4820-aaa8-cc6003b3e75b___RS_Erly.B 7855_final_masked.jpg'
predicted_class_name = predict_image_class(model, image_path, class_indices)

# Output the result
print("Predicted Class Name:", predicted_class_name)

# Get the corresponding disease information
disease_info_text = disease_info.get(predicted_class_name, "No information available for this class.")
print("Disease Information:", disease_info_text)

# Display the image
img = Image.open(image_path)
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis('off')  # Turn off axis numbers
plt.title(f'Predicted: {predicted_class_name}')
plt.show()

model.save('plant_disease_prediction_model.h5')