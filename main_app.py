import streamlit as st
from PIL import Image

from ResNet9 import ResNet9
from model_EfficientNet import EfficientNetV2SModel
from model_ResNet import ResNetModel
from model_cnn import ModelCNN
from model_template import ModelTemplate

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    last_selected_model: ModelTemplate = ModelTemplate()
    last_selected_model.load_model()

    models = [ModelCNN(), ResNetModel(), EfficientNetV2SModel()]
    model_names = list()

    for model in models:
        model_names.append(model.name)

    st.title('Plant Disease Classification')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    option = st.selectbox(
        "What model do you like to use?",
        (model_names)
    )

    if uploaded_file is not None and option is not None:

        selected_model = models[model_names.index(option)]

        if last_selected_model != selected_model:
            last_selected_model.unload_model()
            selected_model.load_model()
            last_selected_model = selected_model

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        with st.empty():
            # Display "Classifying..." initially
            st.write("Classifying...")

            prediction = selected_model.predict(image)

            # Update the placeholder with the prediction result
            st.write(f"Prediction of {selected_model.name}:")

        st.write(f"Class: {prediction}")
        splitted = prediction.split("___")
        st.write(f"Plant: {splitted[0]}")
        st.write(f"Disease: {splitted[1]}")
        st.write(f"Healthy: {splitted[1] == 'healthy'}")