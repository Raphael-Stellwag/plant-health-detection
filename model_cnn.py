import json

import numpy as np


class ModelCNN:
    name = "CNN"
    file = "models/plant_disease_prediction_model_cnn.h5"

    def train(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.file)

    def predict(self, img):

        target_size = (224, 224)

        # Resize the image
        img = img.resize(target_size)
        # Convert the image to a numpy array
        img_array = np.array(img)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Scale the image values to [0, 1]
        img_array = img_array.astype('float32') / 255.

        predictions = self.model.predict(img_array)
        print(predictions)

        predicted_class_index = np.argmax(predictions, axis=1)[0]
        class_indices = json.load(open("class_indices.json"))
        predicted_class_name = class_indices[str(predicted_class_index)]
        return predicted_class_name