import numpy as np


class ModelTemplate:
    name = "Model Template"

    def train(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

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


        # Load the model
        # predictions = model.predict(img_array)
        # predicted_class_index = np.argmax(predictions, axis=1)[0]
        # predicted_class_name = class_indices[predicted_class_index]
        # return predicted_class_name

        return "Prediction"