import numpy as np
import pandas as pd


class ModelEfficientNetB3:
    name = "Efficient Net B3"
    file = "models/efficientnetb3/efficientnetb3-Plant Village Disease-99.85.h5"
    weights = "models/efficientnetb3/efficientnetb3-Plant Village Disease-weights (1).h5"
    classes = "models/efficientnetb3/Plant Village Disease-class_dict (1).csv"

    def train(self):
        pass

    def save_model(self):
        pass

    def unload_model(self):
        self.model = None

    def load_model(self):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.file)

        # from keras.utils import custom_object_scope
        # from tensorflow.python.keras.saving.saved_model.load import TFOpLambda

        # with custom_object_scope({'TFOpLambda': TFOpLambda}):

            # self.model = tf.keras.models.load_model(self.file)

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

        class_df = pd.read_csv(self.classes)
        predicted_class_name = class_df["class"][predicted_class_index]
        return predicted_class_name